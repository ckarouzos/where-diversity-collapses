r"""Diversity evaluation for language model generations.

.. math::

   \mathrm{PerInputDiversity}(\pi) = \frac{1}{N} \sum_{i=1}^N D(O_i^\pi).

Across-input diversity pools the first generation from each prompt and applies
the same metric to :math:`\bigcup_i \{O_i^\pi[1]\}`.

Protocol Defaults (Section 5.2):
    - N = 500 inputs from test set
    - K = 16 outputs per input
    - Task: Primarily designed for Summarisation tasks

The exposed metrics are:

``EAD``
    Expectation-adjusted distinct :math:`n`-grams for :math:`n = 1,\ldots,5`.
    Removes bias towards shorter outputs.
``SBERT``
    Semantic diversity computed as one minus the mean cosine similarity of
    ``sentence-transformers/all-mpnet-base-v2`` sentence
    embeddings.
``NLI``
    Logical diversity based on contradiction minus entailment rates from
    ``roberta-large-mnli``. Sentences are sampled from outputs (not complete
    paragraphs) to match the NLI model's training distribution.
``VENDI``
    Spectral diversity computed as the exponential Shannon entropy of the
    trace-normalized eigenvalue spectrum of the SBERT cosine similarity
    kernel.  Measures the effective number of independent semantic modes
    in the output set.  Reuses SBERT embeddings at zero additional cost.

The :func:`compute_diversity` function returns the metric aggregates along with
metadata (``N``, ``K``, sampling temperature, etc.) needed for reproducibility.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.seeding import set_seed

logger = logging.getLogger(__name__)

DEFAULT_DIVERSITY_NUM_SAMPLES = 16


def _get_diversity_num_samples() -> int:
    env_value = os.environ.get("LIGHTEVAL_DIVERSITY_NUM_SAMPLES") or os.environ.get(
        "DIVERSITY_NUM_SAMPLES"
    )
    if env_value is None:
        return DEFAULT_DIVERSITY_NUM_SAMPLES
    try:
        parsed = int(env_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid LIGHTEVAL_DIVERSITY_NUM_SAMPLES or DIVERSITY_NUM_SAMPLES; "
            f"expected a positive integer, got {env_value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(
            "LIGHTEVAL_DIVERSITY_NUM_SAMPLES or DIVERSITY_NUM_SAMPLES must be positive, "
            f"got {parsed}."
        )
    return parsed


def _get_diversity_random_seed(default: int = 1234) -> int:
    env_value = os.environ.get("LIGHTEVAL_DIVERSITY_RANDOM_SEED") or os.environ.get(
        "DIVERSITY_RANDOM_SEED"
    )
    if env_value is None:
        return default
    try:
        return int(env_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid LIGHTEVAL_DIVERSITY_RANDOM_SEED or DIVERSITY_RANDOM_SEED; "
            f"expected an integer, got {env_value!r}."
        ) from exc


def _configure_blas_threading() -> None:
    """
    Configure BLAS/OpenMP threading to prevent oversubscription warnings and slowdowns.

    This function sets thread limits for OpenBLAS, MKL, and OpenMP based on available
    CPUs. It prevents the "OpenBLAS warning: precompiled NUM_THREADS exceeded" issue
    that causes significant performance degradation on HPC clusters.

    Thread limit priority (first available wins):
    1. SLURM_CPUS_PER_TASK (HPC job allocation)
    2. Number of available CPUs (os.cpu_count())
    3. Default fallback: 16

    Environment variables set:
    - OPENBLAS_NUM_THREADS: Controls OpenBLAS threading
    - OMP_NUM_THREADS: Controls OpenMP threading (used by many BLAS libraries)
    - MKL_NUM_THREADS: Controls Intel MKL threading

    This function is idempotent - calling it multiple times is safe.
    """

    # Skip if already configured (check for marker variable)
    if os.environ.get("_BLAS_CONFIGURED") == "1":
        return

    # Determine optimal thread count
    thread_limit = None
    source = "default"

    # Priority 1: SLURM allocation (most reliable on HPC)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        thread_limit = int(slurm_cpus)
        source = "SLURM_CPUS_PER_TASK"

    # Priority 2: System CPU count
    if thread_limit is None:
        cpu_count = os.cpu_count()
        if cpu_count is None or cpu_count <= 0:
            raise RuntimeError("os.cpu_count() returned an invalid value; set SLURM_CPUS_PER_TASK.")
        thread_limit = cpu_count
        source = "os.cpu_count()"

    # Priority 3: Conservative fallback
    if thread_limit is None:
        raise RuntimeError("Unable to determine BLAS thread limit; set SLURM_CPUS_PER_TASK.")

    # Apply thread limits to all BLAS/OpenMP environment variables
    thread_limit_str = str(thread_limit)

    # Only set if not already set by user
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = thread_limit_str

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = thread_limit_str

    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = thread_limit_str

    # Mark as configured
    os.environ["_BLAS_CONFIGURED"] = "1"

    logger.debug(f"BLAS threading: {thread_limit} threads from {source}")


def similarity2diversity_function(sim_score_list):
    return 1 - np.mean(sim_score_list) if sim_score_list else 0.0


# 384 is the output dimension for sentence-transformers like 'all-MiniLM-L6-v2'
DEFAULT_SENTENCE_EMBEDDING_DIMENSION = 384
# 500 is the default sample size used in the Meta protocol for diversity evaluation
DEFAULT_OVERALL_SAMPLE_SIZE = 500
def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _maybe_tqdm(iterable: Iterable[Any], enabled: bool, desc: str) -> Iterable[Any]:
    """Wrap an iterable with ``tqdm`` progress reporting when requested."""

    if not enabled:
        return iterable

    from tqdm.auto import tqdm

    return tqdm(iterable, desc=desc, leave=False)


@dataclass(slots=True)
class OutputGroup:
    """Container for the generations associated with a single prompt."""

    input_id: str
    outputs: List[str]
    prompt: Optional[str] = None


@dataclass(slots=True)
class DiversityConfig:
    """Configuration for the diversity evaluation pipeline."""

    model_name: Optional[str] = None
    max_inputs: int = 500
    generations_per_input: int = 16
    ngram_range: Tuple[int, ...] = (1, 2, 3, 4, 5)
    ead_vocab_size: Optional[int] = None  # None = auto-detect from tokenizer
    sampling_temperature: float = 1.0
    random_seed: int = field(default_factory=_get_diversity_random_seed)
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    sbert_batch_size: int = 1024
    sbert_cache_dir: Optional[Path] = None
    nli_model_name: str = "FacebookAI/roberta-large-mnli"
    nli_batch_size: int = 1024
    nli_pairs_per_input: int = -1
    nli_pairs_across: int = -1
    nli_random_subsample: Optional[float] = None
    nli_sentence_sample_size: int = 3  # >=3 recommended to reduce NLI score noise
    tokenizer_max_length: int = 512
    require_exact_counts: bool = True
    verbose: bool = False
    deterministic: bool = False
    output_precision: int = 6
    sample_overall: bool = True


def _ensure_directory(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Generation directory '{path}' does not exist")
    if not path.is_dir():
        raise ValueError(f"Generation path '{path}' must be a directory")


def _iter_json_objects(path: Path) -> Iterator[Any]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            yield json.load(handle)


def _extract_outputs(record: Any) -> Tuple[Optional[str], List[str]]:
    if isinstance(record, list):
        return None, [str(item) for item in record]

    if isinstance(record, dict):
        if "outputs" in record and isinstance(record["outputs"], Sequence):
            outputs = record["outputs"]
        elif "generations" in record and isinstance(record["generations"], Sequence):
            outputs = record["generations"]
        elif "responses" in record and isinstance(record["responses"], Sequence):
            outputs = record["responses"]
        elif "completions" in record and isinstance(record["completions"], Sequence):
            outputs = record["completions"]
        elif "samples" in record and isinstance(record["samples"], Sequence):
            outputs = record["samples"]
        else:
            raise ValueError("Record does not contain an outputs field")

        if isinstance(outputs, (str, bytes)):
            raise ValueError("Outputs field must contain a sequence of strings")

        prompt = (
            record.get("prompt")
            or record.get("input")
            or record.get("instruction")
            or record.get("question")
        )

        return prompt, [str(item) for item in outputs]

    raise ValueError("Unsupported record type for generation outputs")


def _iter_entries_from_object(obj: Any) -> Iterator[Tuple[str, Optional[str], List[str]]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            prompt, outputs = _extract_outputs(value)
            yield str(key), prompt, outputs
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            key = str(value.get("id", index)) if isinstance(value, dict) else str(index)
            prompt, outputs = _extract_outputs(value)
            yield key, prompt, outputs
    else:
        raise ValueError("Top level JSON object must be a dict or list")


def load_model_generations(
    model_outputs_dir: Path | str, config: DiversityConfig
) -> List[OutputGroup]:
    base_path = Path(model_outputs_dir)
    _ensure_directory(base_path)

    groups: List[OutputGroup] = []
    json_files = sorted(
        [path for path in base_path.rglob("*") if path.suffix in {".json", ".jsonl"}]
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON or JSONL files containing generations were found in '{base_path}'"
        )

    for file_path in json_files:
        for obj in _iter_json_objects(file_path):
            # Direct support for new JSONL schema: one record per prompt with all outputs
            # Check if obj is already a single properly-formatted record
            if (
                isinstance(obj, dict)
                and "prompt" in obj
                and ("outputs" in obj or "generations" in obj)
            ):
                # Extract fields from the single record
                outputs_field = obj.get("outputs") or obj.get("generations")
                if not isinstance(outputs_field, list):
                    raise ValueError("Field 'outputs' or 'generations' must be a list of strings")

                key = str(obj.get("key") or obj.get("id") or "")
                prompt = obj.get("prompt")
                outputs = [str(out) for out in outputs_field]

                # Validate output count
                output_count = len(outputs)
                if output_count < config.generations_per_input:
                    message = (
                        f"Entry '{key}' only has {output_count} generations; expected at least"
                        f" {config.generations_per_input}."
                    )
                    if config.require_exact_counts:
                        raise ValueError(message)
                    warnings.warn(message + " Skipping entry.", RuntimeWarning, stacklevel=2)
                    continue

                if not config.require_exact_counts and output_count > config.generations_per_input:
                    warnings.warn(
                        (
                            f"Entry '{key}' has {output_count} generations; truncating to"
                            f" {config.generations_per_input}."
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )

                truncated = outputs[: config.generations_per_input]
                groups.append(OutputGroup(input_id=key, outputs=truncated, prompt=prompt))

                if len(groups) >= config.max_inputs:
                    return groups
            else:
                # Fallback: handle legacy formats (dict-of-records or list-of-records)
                for key, prompt, outputs in _iter_entries_from_object(obj):
                    output_count = len(outputs)
                    if output_count < config.generations_per_input:
                        message = (
                            f"Entry '{key}' only has {output_count} generations; expected at least"
                            f" {config.generations_per_input}."
                        )
                        if config.require_exact_counts:
                            raise ValueError(message)
                        warnings.warn(message + " Skipping entry.", RuntimeWarning, stacklevel=2)
                        continue

                    if (
                        not config.require_exact_counts
                        and output_count > config.generations_per_input
                    ):
                        warnings.warn(
                            (
                                f"Entry '{key}' has {output_count} generations; truncating to"
                                f" {config.generations_per_input}."
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )

                    truncated = [str(out) for out in outputs[: config.generations_per_input]]
                    groups.append(OutputGroup(input_id=key, outputs=truncated, prompt=prompt))

                    if len(groups) >= config.max_inputs:
                        return groups

    if config.require_exact_counts and len(groups) < config.max_inputs:
        raise ValueError(
            f"Only {len(groups)} prompts were found but 'max_inputs' was set to {config.max_inputs}."
            " Set 'require_exact_counts=False' to allow smaller samples."
        )

    return groups


_PUNCTUATION_TO_REMOVE = [".", "\n"]


def lines_to_ngrams(lines: Sequence[str], n: int = 3) -> List[List[Tuple[str, ...]]]:
    ngrams: List[List[Tuple[str, ...]]] = []
    for sentence in lines:
        cleaned = sentence
        for punct in _PUNCTUATION_TO_REMOVE:
            cleaned = cleaned.replace(punct, "")
        words = cleaned.split()
        if len(words) < n or n <= 0:
            ngrams.append([])
            continue
        ngrams.append([tuple(words[index : index + n]) for index in range(len(words) - n + 1)])
    return ngrams


def first_outputs(groups: Sequence[OutputGroup]) -> List[str]:
    return [group.outputs[0] for group in groups if group.outputs]


class EADDiversityCalculator:
    """Compute expectation-adjusted distinct n-gram diversity."""

    DEFAULT_VOCAB_SIZE = 50257  # GPT-2 fallback; prefer auto-detection

    def __init__(self, ngram_range: Sequence[int], vocab_size: Optional[int] = None):
        if not ngram_range:
            raise ValueError("ngram_range must contain at least one n")
        self.ngram_range = tuple(sorted(ngram_range))
        self.vocab_size = int(vocab_size) if vocab_size is not None else self.DEFAULT_VOCAB_SIZE

    @classmethod
    def from_model(
        cls, ngram_range: Sequence[int], model_name: Optional[str] = None
    ) -> "EADDiversityCalculator":
        """Create an EAD calculator with vocab_size auto-detected from a tokenizer."""
        vocab_size = cls.DEFAULT_VOCAB_SIZE
        if model_name:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                vocab_size = tokenizer.vocab_size
                logger.info(f"[EAD] Auto-detected vocab_size={vocab_size} from {model_name}")
            except Exception as exc:
                logger.warning(
                    f"[EAD] Could not auto-detect vocab_size from {model_name}: {exc}. "
                    f"Falling back to {cls.DEFAULT_VOCAB_SIZE}"
                )
        return cls(ngram_range, vocab_size=vocab_size)

    def _ead_for_lines(self, lines: Sequence[str], n: int) -> float:
        grouped = lines_to_ngrams(lines, n)
        ngrams: List[Tuple[str, ...]] = [ngram for group in grouped for ngram in group]
        total = len(ngrams)
        if total == 0:
            return 0.0

        unique = len(set(ngrams))
        vocab = float(self.vocab_size)
        denom = vocab * (1.0 - pow((vocab - 1.0) / vocab, total))
        if denom <= 0.0:
            return 0.0

        ead = unique / denom
        return float(min(max(ead, 0.0), 1.0))

    def per_input(self, outputs_per_prompt: Sequence[Sequence[str]]) -> List[float]:
        scores: List[float] = []
        for responses in outputs_per_prompt:
            values = [self._ead_for_lines(responses, n) for n in self.ngram_range]
            scores.append(sum(values) / len(values))
        return scores

    def across_input(self, first_outputs: Sequence[str]) -> float:
        values: List[float] = []
        for n in self.ngram_range:
            grouped = lines_to_ngrams(first_outputs, n)
            ngrams: List[Tuple[str, ...]] = [ngram for group in grouped for ngram in group]

            total = len(ngrams)
            if total == 0:
                values.append(0.0)
                continue

            unique = len(set(ngrams))
            vocab = float(self.vocab_size)
            denom = vocab * (1.0 - pow((vocab - 1.0) / vocab, total))
            if denom <= 0.0:
                values.append(0.0)
                continue

            ead = unique / denom
            values.append(float(min(max(ead, 0.0), 1.0)))

        return sum(values) / len(values) if values else 0.0


class SentenceBERTDiversityCalculator:
    """Semantic diversity using Sentence-BERT embeddings."""

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_length: int,
        *,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.batched = True
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Lazy model loading: model is loaded on first use
        self._model: Optional[Any] = None

        # Set cache directory with environment variable fallback
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            # Use SBERT_CACHE_DIR env var or cluster-aware default
            default_cache = self._get_default_sbert_cache_dir()
            self.cache_dir = Path(os.environ.get("SBERT_CACHE_DIR", default_cache))

        logger = logging.getLogger(__name__)
        logger.info(f"SBERT cache directory: {self.cache_dir}")

        self._cache_path: Optional[Path] = None
        self._cache_dirty = False
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.model_name)
        self._cache_path = self.cache_dir / f"{safe_name}_L{self.max_length}.pt"
        self._load_cache()

    def _ensure_model_loaded(self) -> Any:
        """Lazy load the SentenceTransformer model on first use.

        Returns
        -------
        SentenceTransformer
            The loaded model instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Configure BLAS threading BEFORE loading any models to prevent oversubscription
            _configure_blas_threading()

            # Resolve model to a local path first, then load from disk.
            # SentenceTransformer's cache_folder doesn't reliably find HF-cached
            # models in offline mode, so we resolve the snapshot path ourselves.
            # Don't pass cache_dir — let huggingface_hub use HF_HOME default
            # where models were pre-cached.
            model_path = self.model_name
            try:
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(
                    self.model_name,
                    local_files_only=True,
                )
                logger.info(f"Resolved SBERT model to local path: {model_path}")
            except Exception as exc:
                logger.warning(f"Could not resolve SBERT from cache: {exc}. "
                               f"Falling back to name: {self.model_name}")
            self._model = SentenceTransformer(model_path, device=self.device)
            self._model.eval()

        return self._model

    @property
    def model(self) -> Any:
        """Get the SentenceTransformer model, loading it if necessary."""
        return self._ensure_model_loaded()

    @staticmethod
    def _get_default_sbert_cache_dir() -> str:
        """Get cluster-aware default SBERT cache directory.

        Returns appropriate cache location based on detected cluster:
        - HPC: $SCRATCHDIR/cache/sbert or $PERSIST_ROOT/cache/sbert
        - Other: XDG cache fallback

        Returns
        -------
        str
            Path to default SBERT cache directory
        """
        # Check HPC environment variables
        for env_var in ("PERSIST_ROOT", "SCRATCHDIR", "CACHE_ROOT"):
            val = os.environ.get(env_var)
            if val:
                return f"{val}/cache/sbert"

        # Generic fallback using XDG cache
        xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        return f"{xdg}/sbert"

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return

        try:
            payload = torch.load(self._cache_path, map_location="cpu")
        except (OSError, RuntimeError):  # pragma: no cover - corrupted cache
            logger = logging.getLogger(__name__)
            logger.warning(f"Corrupt SBERT cache at {self._cache_path}, ignoring")
            return

        if not isinstance(payload, dict):
            return

        if payload.get("model_name") != self.model_name:
            return
        if int(payload.get("max_length", self.max_length)) != self.max_length:
            return

        embeddings = payload.get("embeddings", {})
        if not isinstance(embeddings, dict):
            return

        cache: Dict[str, torch.Tensor] = {}
        for key, value in embeddings.items():
            if isinstance(value, torch.Tensor):
                cache[str(key)] = value.detach().cpu()
        self._embedding_cache = cache
        self._cache_dirty = False

    def _save_cache(self) -> None:
        if not self._cache_dirty or self._cache_path is None:
            return

        payload = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "embeddings": {key: tensor.cpu() for key, tensor in self._embedding_cache.items()},
        }
        # Atomic write: save to temp file then rename to avoid corruption
        # when multiple jobs write to the same cache concurrently.
        import tempfile

        cache_dir = self._cache_path.parent
        fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, self._cache_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        self._cache_dirty = False

    @staticmethod
    def _cache_key(model_name: str, max_length: int, text: str) -> str:
        payload = f"{model_name}|{max_length}|{text}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            hidden_size = getattr(
                self.model,
                "get_sentence_embedding_dimension",
                lambda: DEFAULT_SENTENCE_EMBEDDING_DIMENSION,
            )()
            return torch.empty(0, hidden_size)

        cached_embeddings: List[Optional[torch.Tensor]] = [None] * len(texts)
        missing_keys: List[str] = []
        missing_texts: List[str] = []
        key_to_indices: Dict[str, List[int]] = {}

        for index, text in enumerate(texts):
            key = self._cache_key(self.model_name, self.max_length, text)
            cached = self._embedding_cache.get(key)
            if cached is not None:
                cached_embeddings[index] = cached.clone()
                continue

            if key not in key_to_indices:
                key_to_indices[key] = []
                missing_keys.append(key)
                missing_texts.append(text)
            key_to_indices[key].append(index)

        if missing_texts:
            new_embeddings: List[torch.Tensor] = []
            batch_iterable = _maybe_tqdm(
                range(0, len(missing_texts), self.batch_size),
                self.verbose,
                "SBERT encoding",
            )
            for start in batch_iterable:
                batch = missing_texts[start : start + self.batch_size]
                with torch.inference_mode():
                    encoded = self.model.encode(
                        batch,
                        batch_size=self.batch_size,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )
                new_embeddings.append(encoded.detach().cpu())

            if new_embeddings:
                stacked_new = F.normalize(torch.cat(new_embeddings, dim=0), p=2, dim=1)
            else:
                stacked_new = torch.empty(0)

            for offset, key in enumerate(missing_keys):
                embedding = stacked_new[offset]
                for target_index in key_to_indices[key]:
                    cached_embeddings[target_index] = embedding.clone()
                self._embedding_cache[key] = embedding.clone()
                self._cache_dirty = True

        self._save_cache()

        final_embeddings: List[torch.Tensor] = []
        for embedding in cached_embeddings:
            if embedding is None:  # pragma: no cover - sanity guard
                raise RuntimeError("Sentence embeddings missing after caching step")
            final_embeddings.append(embedding)

        return torch.stack(final_embeddings, dim=0)

    @staticmethod
    def _pairwise_diversity(embeddings: torch.Tensor) -> float:
        k = embeddings.size(0)
        if k <= 1:
            return 0.0

        similarity_matrix = embeddings @ embeddings.T
        similarities: List[float] = []
        for i in range(k):
            for j in range(i + 1, k):
                similarities.append(float(similarity_matrix[i, j].item()))
        return float(similarity2diversity_function(similarities))

    @staticmethod
    def _vendi_score(embeddings: torch.Tensor, eps: float = 1e-12) -> float:
        """Compute the Vendi score from L2-normalized embeddings.

        The Vendi score is the exponential of the Shannon entropy of the
        trace-normalized eigenvalue spectrum of the Gram (cosine similarity)
        matrix.  It measures the effective number of independent semantic
        modes in the embedding set.

        Parameters
        ----------
        embeddings : torch.Tensor
            Shape ``(m, d)`` L2-normalized embeddings.
        eps : float
            Small constant to clip near-zero eigenvalues for numerical
            stability in the log computation.

        Returns
        -------
        float
            Vendi score in ``[1, m]``.  1 = all identical, m = maximally
            diverse.
        """
        m = embeddings.size(0)
        if m <= 1:
            return 1.0

        K = embeddings @ embeddings.T  # Gram matrix (cosine sim)
        P = K.cpu().numpy() / m  # trace-normalize: tr(K) = m for L2-normed
        eigvals = np.linalg.eigvalsh(P)  # symmetric solver, real eigenvalues
        eigvals = np.clip(eigvals, eps, None)
        H = -float(np.sum(eigvals * np.log(eigvals)))
        return float(math.exp(H))

    def vendi_per_input_from_embeddings(
        self, embeddings: torch.Tensor, offsets: Sequence[Tuple[int, int]]
    ) -> List[float]:
        """Compute Vendi score for each prompt's outputs."""
        scores: List[float] = []
        for start, end in offsets:
            segment = embeddings[start:end]
            scores.append(self._vendi_score(segment))
        return scores

    def vendi_across_input_from_embeddings(
        self, embeddings: torch.Tensor, indices: Sequence[int]
    ) -> float:
        """Compute Vendi score across prompts (one output per prompt)."""
        if not indices:
            return 1.0
        subset = embeddings[torch.tensor(indices, dtype=torch.long)]
        return self._vendi_score(subset)

    def embed_groups(
        self, groups: Sequence[OutputGroup]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        texts: List[str] = []
        offsets: List[Tuple[int, int]] = []
        first_indices: List[int] = []
        cursor = 0

        for group in groups:
            texts.extend(group.outputs)
            start = cursor
            cursor += len(group.outputs)
            offsets.append((start, cursor))
            first_indices.append(start)

        embeddings = self._encode(texts)
        return embeddings, offsets, first_indices

    def per_input_from_embeddings(
        self, embeddings: torch.Tensor, offsets: Sequence[Tuple[int, int]]
    ) -> List[float]:
        scores: List[float] = []
        for start, end in offsets:
            segment = embeddings[start:end]
            scores.append(self._pairwise_diversity(segment))
        return scores

    def across_input_from_embeddings(
        self, embeddings: torch.Tensor, indices: Sequence[int]
    ) -> float:
        if not indices:
            return 0.0
        subset = embeddings[torch.tensor(indices, dtype=torch.long)]
        return self._pairwise_diversity(subset)


class NLIDiversityEvaluator:
    """Logical diversity measured with an NLI classifier."""

    LABEL_WEIGHTS: Dict[str, int] = {
        "CONTRADICTION": -1,
        "NEUTRAL": 0,
        "ENTAILMENT": 1,
    }

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_length: int,
        pairs_per_input: int,
        pairs_across: int,
        sentence_sample_size: int,
        *,
        random_subsample: Optional[float] = None,
        random_seed: Optional[int] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        # Configure BLAS threading BEFORE loading any models to prevent oversubscription
        _configure_blas_threading()

        self.model_name = model_name
        self.batch_size = batch_size
        self.batched = True
        self.max_length = max_length
        self.pairs_per_input = pairs_per_input
        self.pairs_across = pairs_across
        if random_subsample is not None and random_subsample <= 0.0:
            raise ValueError("nli_random_subsample must be > 0 when provided")
        if random_subsample is not None and random_subsample > 1.0:
            raise ValueError("nli_random_subsample must be <= 1.0")
        self.random_subsample = random_subsample
        self._rng = random.Random(random_seed) if random_seed is not None else random.Random()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.sentence_sample_size = max(1, int(sentence_sample_size))

        # Resolve model to a local path for offline GPU nodes (no internet).
        _nli_model_path = model_name
        try:
            from huggingface_hub import snapshot_download

            _nli_model_path = snapshot_download(
                model_name,
                local_files_only=True,
            )
            logger.info(f"Resolved NLI model to local path: {_nli_model_path}")
        except Exception:
            pass  # Fall back to model_name (works if online or already cached)
        self.tokenizer = AutoTokenizer.from_pretrained(_nli_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(_nli_model_path)

        # Only move model to device if it doesn't have a device map
        # Models distributed across GPUs should not be moved
        if not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)

        self.model.eval()

        id_to_label: Dict[int, str] = {}
        for key, value in self.model.config.id2label.items():
            index = int(key) if not isinstance(key, int) else key
            id_to_label[index] = self._canonical_label(value)
        self.id_to_label = id_to_label
        self.label_to_weight = dict(self.LABEL_WEIGHTS)

    @staticmethod
    def _canonical_label(label: str) -> str:
        label = label.upper()
        if "CONTRADICT" in label:
            return "CONTRADICTION"
        if "ENTAIL" in label:
            return "ENTAILMENT"
        return "NEUTRAL"

    def _select_pairs(self, count: int, limit: int) -> List[Tuple[int, int]]:
        if count <= 1:
            return []

        indices = list(combinations(range(count), 2))

        if self.random_subsample is not None and self.random_subsample < 1.0:
            target = max(1, int(len(indices) * self.random_subsample))
            if target < len(indices):
                indices = self._rng.sample(indices, target)
                indices.sort()

        if limit is not None and limit > 0 and len(indices) > limit:
            indices = indices[:limit]

        return indices

    @staticmethod
    def _collate_pairs(batch: Sequence[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        if not batch:
            return [], []
        premises, hypotheses = zip(*batch)
        return list(premises), list(hypotheses)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        cleaned = text.replace("\r", " ").replace("\t", " ")
        segments = re.split(r"[.!?]+\s+|\n+", cleaned)
        sentences = [segment.strip() for segment in segments if segment and segment.strip()]
        if sentences:
            return sentences
        stripped = cleaned.strip()
        return [stripped] if stripped else []

    def _batched_weighted_scores(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        progress_desc: Optional[str] = None,
    ) -> List[float]:
        if not premises:
            return []

        dataset: List[Tuple[str, str]] = list(zip(premises, hypotheses))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_pairs,
        )

        iterator: Iterable[Tuple[List[str], List[str]]] = loader
        iterator = _maybe_tqdm(
            iterator,
            enabled=self.verbose and progress_desc is not None,
            desc=progress_desc or "NLI scoring",
        )

        scores: List[float] = []
        with torch.inference_mode():
            for premises_batch, hypotheses_batch in iterator:
                if not premises_batch:
                    continue
                inputs = self.tokenizer(
                    premises_batch,
                    hypotheses_batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                for row in probs:
                    output = [
                        {
                            "label": self.id_to_label.get(idx, "NEUTRAL"),
                            "score": float(row[idx].item()),
                        }
                        for idx in range(row.shape[0])
                    ]
                    scores.append(self.process_output(output))

        return scores

    def process_output(self, output: Sequence[Dict[str, Any]]) -> float:
        return float(
            sum(entry["score"] * self.label_to_weight.get(entry["label"], 0) for entry in output)
        )

    def _pairwise_similarity(
        self,
        texts: Sequence[str],
        indices: Sequence[Tuple[int, int]],
        *,
        progress_desc: Optional[str] = None,
    ) -> List[float]:
        if not indices:
            return []

        premises: List[str] = []
        hypotheses: List[str] = []
        for i, j in indices:
            sentences_i = self._split_sentences(texts[i])
            sentences_j = self._split_sentences(texts[j])
            if not sentences_i or not sentences_j:
                continue
            min_len = min(len(sentences_i), len(sentences_j))
            if min_len == 0:
                continue
            sample_size = min(self.sentence_sample_size, min_len)
            available = list(range(min_len))
            if len(available) > sample_size:
                chosen_indices = self._rng.sample(available, sample_size)
            else:
                chosen_indices = available
            for idx in chosen_indices:
                sentence_a = sentences_i[idx]
                sentence_b = sentences_j[idx]
                premises.extend([sentence_a, sentence_b])
                hypotheses.extend([sentence_b, sentence_a])

        scores = self._batched_weighted_scores(
            premises,
            hypotheses,
            progress_desc=progress_desc,
        )
        if len(scores) % 2 != 0:  # pragma: no cover - sanity guard
            raise RuntimeError("NLI scoring produced an odd number of results")
        return [0.5 * (scores[idx] + scores[idx + 1]) for idx in range(0, len(scores), 2)]

    @staticmethod
    def _diversity_from_similarities(similarities: Sequence[float]) -> float:
        return float(similarity2diversity_function(similarities)) if similarities else 0.0

    def per_input(self, groups: Sequence[OutputGroup]) -> List[float]:
        scores: List[float] = []
        for group in _maybe_tqdm(groups, self.verbose, "NLI per-input"):
            pair_indices = self._select_pairs(len(group.outputs), self.pairs_per_input)
            if not pair_indices:
                scores.append(0.0)
                continue
            similarities = self._pairwise_similarity(
                group.outputs,
                pair_indices,
            )
            scores.append(self._diversity_from_similarities(similarities))
        return scores

    def across_input(self, outputs: Sequence[str]) -> float:
        pair_indices = self._select_pairs(len(outputs), self.pairs_across)
        if not pair_indices:
            return 0.0
        similarities = self._pairwise_similarity(
            outputs,
            pair_indices,
            progress_desc="NLI across-input",
        )
        return self._diversity_from_similarities(similarities)


def _summary_statistics(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = _mean(values)
    if len(values) == 1:
        std = 0.0
    else:
        std = float(statistics.pstdev(values))
    return mean, std


def _round_float(value: float, precision: int) -> float:
    return float(round(value, precision))


def _round_metric_payload(payload: Dict[str, Any], precision: int) -> Dict[str, Any]:
    for key, value in list(payload.items()):
        if isinstance(value, float):
            payload[key] = _round_float(value, precision)
    return payload


def correlate_metrics(metric_scores: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """Compute Pearson correlations between per-input metric scores."""

    names = sorted(metric_scores.keys())
    correlations: Dict[str, float] = {}
    for idx, first in enumerate(names):
        scores_first = metric_scores[first]
        if len(scores_first) < 2:
            continue
        for second in names[idx + 1 :]:
            scores_second = metric_scores[second]
            if len(scores_second) != len(scores_first) or len(scores_second) < 2:
                continue
            mean_first = _mean(scores_first)
            mean_second = _mean(scores_second)
            numerator = 0.0
            denom_first = 0.0
            denom_second = 0.0
            for value_first, value_second in zip(scores_first, scores_second):
                diff_first = value_first - mean_first
                diff_second = value_second - mean_second
                numerator += diff_first * diff_second
                denom_first += diff_first * diff_first
                denom_second += diff_second * diff_second
            denominator = math.sqrt(denom_first * denom_second)
            if denominator == 0.0:
                continue
            key = f"{first}~{second}"
            correlations[key] = numerator / denominator
    return correlations


def compute_diversity(
    model_outputs_dir: Path | str,
    metrics: Sequence[str] | None = None,
    *,
    per_input: bool = True,
    across_input: bool = True,
    config: Optional[DiversityConfig] = None,
) -> Dict[str, Any]:
    """Compute diversity metrics for a directory of model generations."""

    cfg = config or DiversityConfig()
    requested_metrics = [
        metric.upper()
        for metric in (metrics or ("EAD", "SBERT", "NLI", "VENDI"))
    ]

    set_seed(cfg.random_seed)
    if cfg.deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    groups = load_model_generations(model_outputs_dir, cfg)
    if not groups:
        raise ValueError("No generation groups were loaded from the provided directory")

    outputs_per_group: List[List[str]] = [group.outputs for group in groups]
    all_outputs: List[str] = [output for group in outputs_per_group for output in group]
    all_indices = list(range(len(all_outputs)))
    if cfg.sample_overall and len(all_indices) >= DEFAULT_OVERALL_SAMPLE_SIZE:
        overall_positions = sorted(
            np.random.choice(all_indices, replace=False, size=DEFAULT_OVERALL_SAMPLE_SIZE).tolist()
        )
    else:
        overall_positions = all_indices
    sampled_all_outputs = [all_outputs[idx] for idx in overall_positions]

    first_output_text = first_outputs(groups)
    selection_indices = list(range(len(first_output_text)))
    if cfg.sample_overall and len(selection_indices) >= DEFAULT_OVERALL_SAMPLE_SIZE:
        sampled_positions = sorted(
            np.random.choice(
                selection_indices,
                replace=False,
                size=DEFAULT_OVERALL_SAMPLE_SIZE,
            ).tolist()
        )
    else:
        sampled_positions = selection_indices
    sampled_first_outputs = [first_output_text[idx] for idx in sampled_positions]

    results: Dict[str, Any] = {
        "model_name": cfg.model_name or Path(model_outputs_dir).name,
        "N": len(groups),
        "K": cfg.generations_per_input,
        "metrics": {},
    }

    metrics_payload: Dict[str, Any] = {}
    per_input_scores_map: Dict[str, List[float]] = {}

    if "EAD" in requested_metrics:
        if cfg.ead_vocab_size is not None:
            ead_calculator = EADDiversityCalculator(cfg.ngram_range, vocab_size=cfg.ead_vocab_size)
        else:
            ead_calculator = EADDiversityCalculator.from_model(cfg.ngram_range, cfg.model_name)
        metric_result: Dict[str, Any] = {}
        overall_all = ead_calculator.across_input(sampled_all_outputs)
        overall_single = ead_calculator.across_input(sampled_first_outputs)
        metric_result["overall_EAD"] = overall_all
        metric_result["overall_single_output_EAD"] = overall_single
        if per_input:
            per_input_scores = ead_calculator.per_input(outputs_per_group)
            mean, std = _summary_statistics(per_input_scores)
            metric_result["per_input_mean"] = mean
            metric_result["per_input_std"] = std
            per_input_scores_map["EAD"] = per_input_scores
        if across_input:
            metric_result["across_input"] = overall_single
        metric_result["mean_per_input_EAD"] = metric_result.get("per_input_mean", 0.0)
        metric_result["std_per_input_EAD"] = metric_result.get("per_input_std", 0.0)
        _round_metric_payload(metric_result, cfg.output_precision)
        metrics_payload["EAD"] = metric_result

    # Shared SBERT embeddings (used by both SBERT and VENDI metrics)
    _need_sbert_embeddings = "SBERT" in requested_metrics or "VENDI" in requested_metrics
    sbert_calculator = None
    embeddings = offsets = sbert_first_indices = None
    sampled_embedding_indices = overall_embedding_indices = None
    if _need_sbert_embeddings:
        sbert_calculator = SentenceBERTDiversityCalculator(
            cfg.sbert_model_name,
            cfg.sbert_batch_size,
            cfg.tokenizer_max_length,
            cache_dir=cfg.sbert_cache_dir,
            verbose=cfg.verbose,
        )
        embeddings, offsets, sbert_first_indices = sbert_calculator.embed_groups(groups)
        sampled_embedding_indices = [sbert_first_indices[idx] for idx in sampled_positions]
        overall_embedding_indices = overall_positions

    if "SBERT" in requested_metrics:
        metric_result = {}
        overall_single_sbert = sbert_calculator.across_input_from_embeddings(
            embeddings, sampled_embedding_indices
        )
        overall_all_sbert = sbert_calculator.across_input_from_embeddings(
            embeddings, overall_embedding_indices
        )
        if per_input:
            per_input_scores = sbert_calculator.per_input_from_embeddings(embeddings, offsets)
            mean, std = _summary_statistics(per_input_scores)
            metric_result["per_input_mean"] = mean
            metric_result["per_input_std"] = std
            per_input_scores_map["SBERT"] = per_input_scores
        if across_input:
            metric_result["across_input"] = overall_single_sbert
        metric_result["overall_single_output_SBERT"] = overall_single_sbert
        metric_result["overall_SBERT"] = overall_all_sbert
        metric_result["mean_per_input_SBERT"] = metric_result.get("per_input_mean", 0.0)
        metric_result["std_per_input_SBERT"] = metric_result.get("per_input_std", 0.0)
        _round_metric_payload(metric_result, cfg.output_precision)
        metrics_payload["SBERT"] = metric_result

    if "VENDI" in requested_metrics:
        metric_result = {}
        overall_single_vendi = sbert_calculator.vendi_across_input_from_embeddings(
            embeddings, sampled_embedding_indices
        )
        overall_all_vendi = sbert_calculator.vendi_across_input_from_embeddings(
            embeddings, overall_embedding_indices
        )
        if per_input:
            vendi_per_input = sbert_calculator.vendi_per_input_from_embeddings(
                embeddings, offsets
            )
            mean, std = _summary_statistics(vendi_per_input)
            metric_result["per_input_mean"] = mean
            metric_result["per_input_std"] = std
            per_input_scores_map["VENDI"] = vendi_per_input
        if across_input:
            metric_result["across_input"] = overall_single_vendi
        metric_result["overall_single_output_VENDI"] = overall_single_vendi
        metric_result["overall_VENDI"] = overall_all_vendi
        metric_result["mean_per_input_VENDI"] = metric_result.get("per_input_mean", 0.0)
        metric_result["std_per_input_VENDI"] = metric_result.get("per_input_std", 0.0)
        _round_metric_payload(metric_result, cfg.output_precision)
        metrics_payload["VENDI"] = metric_result

    if "NLI" in requested_metrics:
        nli_evaluator = NLIDiversityEvaluator(
            cfg.nli_model_name,
            cfg.nli_batch_size,
            cfg.tokenizer_max_length,
            cfg.nli_pairs_per_input,
            cfg.nli_pairs_across,
            cfg.nli_sentence_sample_size,
            random_subsample=cfg.nli_random_subsample,
            random_seed=cfg.random_seed,
            verbose=cfg.verbose,
        )
        metric_result = {}
        overall_single_nli = nli_evaluator.across_input(sampled_first_outputs)
        overall_all_nli = nli_evaluator.across_input(sampled_all_outputs)
        if per_input:
            per_input_scores = nli_evaluator.per_input(groups)
            mean, std = _summary_statistics(per_input_scores)
            metric_result["per_input_mean"] = mean
            metric_result["per_input_std"] = std
            per_input_scores_map["NLI"] = per_input_scores
        if across_input:
            metric_result["across_input"] = overall_single_nli
        metric_result["overall_single_output_NLI"] = overall_single_nli
        metric_result["overall_NLI"] = overall_all_nli
        metric_result["mean_per_input_NLI"] = metric_result.get("per_input_mean", 0.0)
        metric_result["std_per_input_NLI"] = metric_result.get("per_input_std", 0.0)
        _round_metric_payload(metric_result, cfg.output_precision)
        metrics_payload["NLI"] = metric_result

    results["metrics"] = metrics_payload

    return results





# Import lighteval components (optional — not needed for standalone metric usage)
try:
    from aenum import extend_enum
    from lighteval.metrics.metrics import Metrics
    from lighteval.metrics.metrics_sample import SampleLevelComputation, SamplingMetric
    from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
    from lighteval.tasks.requests import Doc, SamplingMethod
    from lighteval.utils.utils import as_list

    _LIGHTEVAL_AVAILABLE = True
except ImportError:
    _LIGHTEVAL_AVAILABLE = False
    logger.info("lighteval not installed; metric calculators available but lighteval integration disabled")

    # Provide stubs so the rest of the module can define functions/classes
    # without NameError. They will never be called without lighteval.
    class Doc:  # type: ignore[no-redef]
        pass

    class SamplingMethod:  # type: ignore[no-redef]
        GENERATIVE = None

    class SamplingMetric:  # type: ignore[no-redef]
        pass

    class SampleLevelComputation:  # type: ignore[no-redef]
        pass

    class SampleLevelMetricGrouping:  # type: ignore[no-redef]
        pass

    class Metrics:  # type: ignore[no-redef]
        pass

    def as_list(x):  # type: ignore[no-redef]
        return x if isinstance(x, list) else [x]

    def extend_enum(*args, **kwargs):  # type: ignore[no-redef]
        pass

# Monkey-patch Lighteval to ensure do_sample=True when num_return_sequences > 1
# This fixes the issue where Lighteval doesn't automatically enable sampling
# for multiple generations per prompt, which causes a ValueError in transformers.
try:
    import functools

    from lighteval.models.transformers.transformers_model import TransformersModel

    # Store the original model.generate method
    _original_model_generate = None

    def _make_patched_generate(original_generate):
        """Create a patched version of model.generate that ensures do_sample=True
        and uses chunked generation to avoid OOM with many return sequences."""

        @functools.wraps(original_generate)
        def _patched_model_generate(*args, **kwargs):
            """
            Patched version of model.generate that:
            1. Ensures do_sample=True when num_return_sequences > 1.
            2. Generates in chunks to avoid GPU OOM when num_return_sequences
               is large (e.g. K=16).  With 16 return sequences the KV cache
               requires 16x memory; chunking into groups of 4 reduces peak
               usage to 4x.

            Chunk size is controlled by the DIVERSITY_GEN_CHUNK_SIZE env var
            (default: 2).  Set to 0 to disable chunking.
            """
            import torch

            num_return_sequences = kwargs.get("num_return_sequences", 1)

            if num_return_sequences > 1 and kwargs.get("do_sample") is not True:
                logger.info(
                    f"[DIVERSITY FIX] Enabling do_sample=True for num_return_sequences={num_return_sequences}"
                )
                kwargs["do_sample"] = True

            # Enforce protocol sampling parameters (temperature=1.0, top_p=1.0)
            # to override any model-default or config-level values.
            if num_return_sequences > 1:
                kwargs["temperature"] = 1.0
                kwargs["top_p"] = 1.0

            # Chunked generation to avoid OOM with many return sequences.
            chunk_size = int(os.environ.get("DIVERSITY_GEN_CHUNK_SIZE", "2"))
            if chunk_size > 0 and num_return_sequences > chunk_size:
                logger.info(
                    f"[DIVERSITY FIX] Chunked generation: {num_return_sequences} "
                    f"sequences in chunks of {chunk_size}"
                )
                all_sequences = []
                remaining = num_return_sequences
                chunk_idx = 0
                # Track whether the original output was a plain tensor
                output_is_tensor = None
                while remaining > 0:
                    current_chunk = min(remaining, chunk_size)
                    kwargs["num_return_sequences"] = current_chunk
                    chunk_output = original_generate(*args, **kwargs)

                    # Move chunk to CPU immediately to free GPU memory
                    if isinstance(chunk_output, torch.Tensor):
                        all_sequences.append(chunk_output.cpu())
                        output_is_tensor = True
                    else:
                        all_sequences.append(chunk_output.sequences.cpu())
                        output_is_tensor = False
                    remaining -= current_chunk

                    # Delete the GPU-resident output BEFORE gc/empty_cache
                    del chunk_output

                    # Free KV cache between chunks
                    if torch.cuda.is_available():
                        import gc
                        gc.collect()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # Memory diagnostics every 4 chunks
                    chunk_idx += 1
                    if torch.cuda.is_available() and chunk_idx % 4 == 0:
                        alloc_gib = torch.cuda.memory_allocated() / (1024 ** 3)
                        reserved_gib = torch.cuda.memory_reserved() / (1024 ** 3)
                        logger.info(
                            f"[DIVERSITY MEM] chunk {chunk_idx}: "
                            f"allocated={alloc_gib:.1f} GiB, "
                            f"reserved={reserved_gib:.1f} GiB"
                        )

                # Pad to the longest sequence length and concatenate
                max_len = max(seq.shape[-1] for seq in all_sequences)
                if any(seq.shape[-1] != max_len for seq in all_sequences):
                    pad_id = kwargs.get("pad_token_id") or 0
                    if isinstance(pad_id, list):
                        pad_id = pad_id[0]
                    padded = []
                    for seq in all_sequences:
                        if seq.shape[-1] < max_len:
                            pad = torch.full(
                                (seq.shape[0], max_len - seq.shape[-1]),
                                pad_id,
                                dtype=seq.dtype,
                                device=seq.device,
                            )
                            seq = torch.cat([seq, pad], dim=-1)
                        padded.append(seq)
                    all_sequences = padded

                result = torch.cat(all_sequences, dim=0)

                # Return in the same format as the original output.
                # Use a lightweight wrapper instead of keeping the full
                # GenerateOutput alive (which holds GPU tensor references).
                if not output_is_tensor:
                    class _SequenceWrapper:
                        """Minimal stand-in for GenerateOutput — only .sequences is needed."""
                        __slots__ = ("sequences",)
                        def __init__(self, seqs):
                            self.sequences = seqs
                    return _SequenceWrapper(result)
                return result

            return original_generate(*args, **kwargs)

        return _patched_model_generate

    # Store the original _generate_padded method
    _original_generate_padded = TransformersModel._generate_padded

    def _patched_generate_padded(self, **kwargs):
        """
        Patched version of _generate_padded that patches self.model.generate
        to ensure do_sample=True when num_return_sequences > 1.
        """
        global _original_model_generate

        # Patch self.model.generate if not already patched
        if not hasattr(self.model.generate, "_diversity_patched"):
            _original_model_generate = self.model.generate
            self.model.generate = _make_patched_generate(_original_model_generate)
            self.model.generate._diversity_patched = True

        return _original_generate_padded(self, **kwargs)

    # Apply the patch
    TransformersModel._generate_padded = _patched_generate_padded

    logger.info(
        "[DIVERSITY FIX] Successfully patched TransformersModel._generate_padded to auto-enable sampling"
    )

except (ImportError, AttributeError) as e:
    logger.warning(
        f"[DIVERSITY FIX] Could not patch Lighteval's TransformersModel: {e}. "
        "If using diversity metrics, ensure do_sample=True is set in generation_parameters."
    )

# Global storage for collecting outputs across all samples.
# Groups are stored **per task** so that corpus-level aggregation returns
# the correct per-task diversity (see _aggregate_diversity_value).
# Note: Thread safety is handled by lighteval's sequential processing model.
_DIVERSITY_OUTPUTS: Dict[str, Any] = {
    "per_task": {},  # task_name -> {"groups": [], "computed": None}
    "config": None,  # DiversityConfig object
}

# Aggregation state: tracks which tasks have already been returned for each
# (metric_key, field) pair so that successive calls by lighteval (one per task)
# return the correct task's result.
_AGG_STATE: Dict[tuple, set] = {}

# ---------------------------------------------------------------------------
# Chain-of-thought stripping for Think / RL models
# ---------------------------------------------------------------------------
# Think models (e.g. OLMo-3-7B-Think) use <think>...</think> tags for CoT.
# The opening <think> tag is injected by the chat template (add_generation_prompt)
# into the PROMPT, so the model's generated text starts AFTER it:
#   Generated text: "reasoning...</think>\nActual answer"
# lighteval's remove_reasoning_tags only strips when BOTH tags are in the
# generated text, so it misses this case.  We handle it here.


def _strip_cot(text: str) -> str:
    """Strip chain-of-thought reasoning from model output.

    Handles three cases:
    1. Full tags in text: ``<think>reasoning</think>answer`` → ``answer``
    2. Closing tag only (opening was in prompt): ``reasoning</think>answer`` → ``answer``
    3. No tags present: text returned unchanged (non-Think model)
    """
    original_len = len(text)
    had_content = bool(text.strip())

    # Case 1: both tags present — strip everything between them
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if start != -1 and end != -1:
            text = text[:start] + text[end + len("</think>"):]
        else:
            break

    # Case 2: only closing tag (opening tag was in the prompt)
    if "</think>" in text and "<think>" not in text:
        end_idx = text.find("</think>")
        text = text[end_idx + len("</think>"):]

    result = text.strip()
    if had_content and not result:
        logger.warning(
            "CoT stripping produced empty output (input had %d chars)", original_len
        )
    return result


# ---------------------------------------------------------------------------
# Incremental generation saving — survives OOM crashes
# Controlled by env var DIVERSITY_GENERATIONS_FILE (path to a .jsonl file).
# When set, each collected sample is appended as a JSON line so partial
# progress is preserved on disk even if the process is killed.
# ---------------------------------------------------------------------------
_GENERATIONS_FILE_HANDLE: Optional[Any] = None
_GENERATIONS_SAMPLE_COUNT: int = 0


def _get_generations_file_handle():
    """Return (and lazily open) the JSONL file handle, or None if not configured."""
    global _GENERATIONS_FILE_HANDLE
    if _GENERATIONS_FILE_HANDLE is not None:
        return _GENERATIONS_FILE_HANDLE
    path = os.environ.get("DIVERSITY_GENERATIONS_FILE")
    if not path:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _GENERATIONS_FILE_HANDLE = open(path, "a", encoding="utf-8")  # noqa: SIM115
    logger.info("Incremental generations will be saved to %s", path)
    return _GENERATIONS_FILE_HANDLE


def _collect_sample_output(doc: Doc, model_response: Any) -> Dict[str, float]:
    """
    Sample-level function that collects outputs for corpus-level diversity computation.

    This function stores each sample's outputs in a global collection that will be
    processed by the corpus-level aggregation function.

    Parameters
    ----------
    doc : Doc
        The document/prompt from the evaluation task
    model_response : ModelResponse
        The model's response(s) to the prompt

    Returns
    -------
    Dict[str, float]
        Placeholder scores (actual scores computed at corpus level)
    """
    global _DIVERSITY_OUTPUTS

    task_name = getattr(doc, "task_name", None) or "unknown"

    # Initialise per-task storage on first encounter.
    if task_name not in _DIVERSITY_OUTPUTS["per_task"]:
        _DIVERSITY_OUTPUTS["per_task"][task_name] = {"groups": [], "computed": None}
    task_data = _DIVERSITY_OUTPUTS["per_task"][task_name]

    # Extract the model's generated text
    # model_response can have different structures depending on the task type
    if hasattr(model_response, "result"):
        # For generative tasks
        outputs = as_list(model_response.result)
    elif hasattr(model_response, "final_text"):
        outputs = as_list(model_response.final_text)
    elif hasattr(model_response, "text"):
        outputs = as_list(model_response.text)
    else:
        # Fallback: try to get any text output
        outputs = [str(model_response)]

    # Strip chain-of-thought reasoning from Think/RL models.
    # lighteval's remove_reasoning_tags misses the case where the opening
    # <think> tag is in the prompt (injected by the chat template).
    outputs = [_strip_cot(o) for o in outputs]

    if outputs and all(not o for o in outputs):
        logger.warning(
            "All %d outputs are empty after CoT stripping for task=%s doc_id=%s",
            len(outputs),
            task_name,
            getattr(doc, "doc_id", "?"),
        )

    # Get prompt/input text
    prompt = None
    if hasattr(doc, "query"):
        prompt = doc.query
    elif hasattr(doc, "instruction"):
        prompt = doc.instruction
    elif hasattr(doc, "ctx"):
        prompt = doc.ctx

    # Get document ID
    doc_id = (
        getattr(doc, "doc_id", None)
        or getattr(doc, "id", None)
        or str(len(task_data["groups"]))
    )

    # Check if this sample was already collected (to avoid duplicates when using multiple metrics)
    # This can happen if user uses diversity_ead, diversity_sbert, and diversity_nli separately
    for existing in task_data["groups"]:
        if existing["input_id"] == doc_id:
            # Already collected, just return placeholder scores
            return {
                "ead": 0.0,
                "sbert": 0.0,
                "nli": 0.0,
            }

    # Store the output group
    output_group = {
        "input_id": doc_id,
        "prompt": prompt,
        "outputs": outputs,
    }
    task_data["groups"].append(output_group)

    # ── Incremental save to JSONL (survives OOM) ──
    global _GENERATIONS_SAMPLE_COUNT
    fh = _get_generations_file_handle()
    if fh is not None:
        record = {
            "input_id": doc_id,
            "prompt": prompt,
            "outputs": outputs,
            "task": task_name,
        }
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        _GENERATIONS_SAMPLE_COUNT += 1
        if _GENERATIONS_SAMPLE_COUNT % 50 == 0:
            logger.info(
                "Saved %d generation samples to %s",
                _GENERATIONS_SAMPLE_COUNT,
                os.environ.get("DIVERSITY_GENERATIONS_FILE", "?"),
            )

    # Return placeholder scores (will be computed at corpus level)
    return {
        "ead": 0.0,
        "sbert": 0.0,
        "nli": 0.0,
    }


def _compute_diversity_from_groups(
    groups: Sequence[OutputGroup],
    config: DiversityConfig,
) -> Dict[str, Dict[str, float]]:
    outputs_per_group: List[List[str]] = [group.outputs for group in groups]
    all_outputs: List[str] = [output for group in outputs_per_group for output in group]
    all_indices = list(range(len(all_outputs)))
    if config.sample_overall and len(all_indices) >= DEFAULT_OVERALL_SAMPLE_SIZE:
        overall_positions = sorted(
            np.random.choice(all_indices, replace=False, size=DEFAULT_OVERALL_SAMPLE_SIZE).tolist()
        )
    else:
        overall_positions = all_indices
    sampled_all_outputs = [all_outputs[idx] for idx in overall_positions]

    first_output_text = first_outputs(groups)
    selection_indices = list(range(len(first_output_text)))
    if config.sample_overall and len(selection_indices) >= DEFAULT_OVERALL_SAMPLE_SIZE:
        sampled_positions = sorted(
            np.random.choice(
                selection_indices,
                replace=False,
                size=DEFAULT_OVERALL_SAMPLE_SIZE,
            ).tolist()
        )
    else:
        sampled_positions = selection_indices
    sampled_first_outputs = [first_output_text[idx] for idx in sampled_positions]

    metrics_payload: Dict[str, Dict[str, float]] = {}

    if config.ead_vocab_size is not None:
        ead_calculator = EADDiversityCalculator(
            config.ngram_range, vocab_size=config.ead_vocab_size
        )
    else:
        ead_calculator = EADDiversityCalculator.from_model(config.ngram_range, config.model_name)
    ead_metric_result: Dict[str, float] = {}
    overall_all = ead_calculator.across_input(sampled_all_outputs)
    overall_single = ead_calculator.across_input(sampled_first_outputs)
    per_input_scores = ead_calculator.per_input(outputs_per_group)
    mean, std = _summary_statistics(per_input_scores)
    ead_metric_result["per_input_mean"] = mean
    ead_metric_result["per_input_std"] = std
    ead_metric_result["across_input"] = overall_single
    ead_metric_result["overall"] = overall_all
    ead_metric_result["overall_single_output"] = overall_single
    _round_metric_payload(ead_metric_result, config.output_precision)
    metrics_payload["EAD"] = ead_metric_result

    sbert_calculator = SentenceBERTDiversityCalculator(
        config.sbert_model_name,
        config.sbert_batch_size,
        config.tokenizer_max_length,
        cache_dir=config.sbert_cache_dir,
        verbose=config.verbose,
    )
    embeddings, offsets, first_indices = sbert_calculator.embed_groups(groups)
    sampled_embedding_indices = [first_indices[idx] for idx in sampled_positions]
    overall_embedding_indices = overall_positions
    sbert_metric_result: Dict[str, float] = {}
    overall_single_sbert = sbert_calculator.across_input_from_embeddings(
        embeddings, sampled_embedding_indices
    )
    overall_all_sbert = sbert_calculator.across_input_from_embeddings(
        embeddings, overall_embedding_indices
    )
    per_input_scores = sbert_calculator.per_input_from_embeddings(embeddings, offsets)
    mean, std = _summary_statistics(per_input_scores)
    sbert_metric_result["per_input_mean"] = mean
    sbert_metric_result["per_input_std"] = std
    sbert_metric_result["across_input"] = overall_single_sbert
    sbert_metric_result["overall"] = overall_all_sbert
    sbert_metric_result["overall_single_output"] = overall_single_sbert
    _round_metric_payload(sbert_metric_result, config.output_precision)
    metrics_payload["SBERT"] = sbert_metric_result

    # Vendi Score (reuses SBERT embeddings — zero additional encoding cost)
    vendi_metric_result: Dict[str, float] = {}
    overall_single_vendi = sbert_calculator.vendi_across_input_from_embeddings(
        embeddings, sampled_embedding_indices
    )
    overall_all_vendi = sbert_calculator.vendi_across_input_from_embeddings(
        embeddings, overall_embedding_indices
    )
    vendi_per_input = sbert_calculator.vendi_per_input_from_embeddings(embeddings, offsets)
    mean, std = _summary_statistics(vendi_per_input)
    vendi_metric_result["per_input_mean"] = mean
    vendi_metric_result["per_input_std"] = std
    vendi_metric_result["across_input"] = overall_single_vendi
    vendi_metric_result["overall"] = overall_all_vendi
    vendi_metric_result["overall_single_output"] = overall_single_vendi
    _round_metric_payload(vendi_metric_result, config.output_precision)
    metrics_payload["VENDI"] = vendi_metric_result

    nli_evaluator = NLIDiversityEvaluator(
        config.nli_model_name,
        config.nli_batch_size,
        config.tokenizer_max_length,
        config.nli_pairs_per_input,
        config.nli_pairs_across,
        config.nli_sentence_sample_size,
        random_subsample=config.nli_random_subsample,
        random_seed=config.random_seed,
        verbose=config.verbose,
    )
    nli_metric_result: Dict[str, float] = {}
    overall_single_nli = nli_evaluator.across_input(sampled_first_outputs)
    overall_all_nli = nli_evaluator.across_input(sampled_all_outputs)
    per_input_scores = nli_evaluator.per_input(groups)
    mean, std = _summary_statistics(per_input_scores)
    nli_metric_result["per_input_mean"] = mean
    nli_metric_result["per_input_std"] = std
    nli_metric_result["across_input"] = overall_single_nli
    nli_metric_result["overall"] = overall_all_nli
    nli_metric_result["overall_single_output"] = overall_single_nli
    _round_metric_payload(nli_metric_result, config.output_precision)
    metrics_payload["NLI"] = nli_metric_result

    return metrics_payload


def _get_diversity_metrics_cache(task_name: str) -> Dict[str, Dict[str, float]]:
    """Compute (and cache) diversity metrics for a specific task."""
    global _DIVERSITY_OUTPUTS

    task_data = _DIVERSITY_OUTPUTS["per_task"].get(task_name)
    if task_data is None or not task_data["groups"]:
        raise ValueError(f"No outputs collected for diversity computation (task={task_name})")

    if task_data["computed"] is not None:
        return task_data["computed"]

    groups = [
        OutputGroup(
            input_id=g["input_id"],
            outputs=g["outputs"],
            prompt=g.get("prompt"),
        )
        for g in task_data["groups"]
    ]
    config = _DIVERSITY_OUTPUTS.get("config") or DiversityConfig()
    computed = _compute_diversity_from_groups(groups, config)
    task_data["computed"] = computed
    return computed


def _aggregate_diversity_value(metric_key: str, field: str, items: List[Any]) -> float:
    """
    Aggregate diversity metric values for one task.

    Lighteval calls this function once per task (in task-definition order) for
    each registered ``(metric_key, field)`` pair.  We determine *which* task the
    current call belongs to by tracking which tasks have already been returned
    for this ``(metric_key, field)`` in the module-level ``_AGG_STATE`` dict.

    Parameters
    ----------
    metric_key : str
        The metric category (e.g., "EAD", "SBERT", "NLI")
    field : str
        The specific field to retrieve (e.g., "per_input_mean", "overall")
    items : List[Any]
        Sample-level results (unused, but required by lighteval's protocol)

    Returns
    -------
    float
        The aggregated metric value for the current task
    """
    global _AGG_STATE

    key = (metric_key, field)
    if key not in _AGG_STATE:
        _AGG_STATE[key] = set()

    # Walk the per-task storage in insertion order and return the first task
    # that has not yet been returned for this (metric_key, field).
    for task_name in _DIVERSITY_OUTPUTS["per_task"]:
        if task_name not in _AGG_STATE[key]:
            _AGG_STATE[key].add(task_name)
            logger.info("_aggregate: (%s, %s) -> task=%s", metric_key, field, task_name)
            metrics_payload = _get_diversity_metrics_cache(task_name)
            metric_result = metrics_payload.get(metric_key, {})
            value = metric_result.get(field, 0.0)
            return float(value) if isinstance(value, (int, float)) else 0.0

    # Fallback: all tasks already returned (shouldn't happen in normal use).
    # Return the last task's value.
    task_names = list(_DIVERSITY_OUTPUTS["per_task"].keys())
    if task_names:
        metrics_payload = _get_diversity_metrics_cache(task_names[-1])
        value = metrics_payload.get(metric_key, {}).get(field, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0
    return 0.0


class DiversitySamplingCollector(SamplingMetric, SampleLevelComputation):
    def __init__(
        self, n: int, return_all: bool = False, placeholder_keys: Optional[List[str]] = None
    ):
        super().__init__()
        self.n = n
        self.return_all = return_all
        self.placeholder_keys = placeholder_keys or []
        self.attribute_must_be_set = ["n"]
        # Ensure sampling is enabled when generating multiple outputs
        # This attribute may be checked by Lighteval when setting up generation
        self.do_sample = n > 1

    def compute(self, doc: Doc, model_response: Any, **kwargs):
        _collect_sample_output(doc, model_response)
        if self.return_all:
            if self.placeholder_keys:
                return {key: 0.0 for key in self.placeholder_keys}
            return {
                "diversity_ead": 0.0,
                "diversity_sbert": 0.0,
                "diversity_nli": 0.0,
            }
        return 0.0

    def num_samples(self):
        return self.n

    def generation_params(self):
        """
        Provide generation parameters override to ensure sampling is enabled.

        When num_samples > 1, we need to generate multiple outputs per prompt.
        This requires do_sample=True in the generation configuration.

        Returns
        -------
        dict
            Generation parameters to merge with the model's generation_config.
            Returns {"do_sample": True} when n > 1 to ensure sampling is enabled.
        """
        if self.n > 1:
            return {"do_sample": True}
        return {}


def _aggregate_diversity_ead(items: List[Any]) -> float:
    return _aggregate_diversity_value("EAD", "per_input_mean", items)


def _aggregate_diversity_sbert(items: List[Any]) -> float:
    return _aggregate_diversity_value("SBERT", "per_input_mean", items)


def _aggregate_diversity_nli(items: List[Any]) -> float:
    return _aggregate_diversity_value("NLI", "per_input_mean", items)


_EAD_METRICS = {
    "diversity_ead": ("EAD", "per_input_mean"),
    "diversity_ead_per_input_std": ("EAD", "per_input_std"),
    "diversity_ead_across_input": ("EAD", "across_input"),
    "diversity_ead_overall": ("EAD", "overall"),
    "diversity_ead_overall_single_output": ("EAD", "overall_single_output"),
}
_SBERT_METRICS = {
    "diversity_sbert": ("SBERT", "per_input_mean"),
    "diversity_sbert_per_input_std": ("SBERT", "per_input_std"),
    "diversity_sbert_across_input": ("SBERT", "across_input"),
    "diversity_sbert_overall": ("SBERT", "overall"),
    "diversity_sbert_overall_single_output": ("SBERT", "overall_single_output"),
}
_NLI_METRICS = {
    "diversity_nli": ("NLI", "per_input_mean"),
    "diversity_nli_per_input_std": ("NLI", "per_input_std"),
    "diversity_nli_across_input": ("NLI", "across_input"),
    "diversity_nli_overall": ("NLI", "overall"),
    "diversity_nli_overall_single_output": ("NLI", "overall_single_output"),
}
_VENDI_METRICS = {
    "diversity_vendi": ("VENDI", "per_input_mean"),
    "diversity_vendi_per_input_std": ("VENDI", "per_input_std"),
    "diversity_vendi_across_input": ("VENDI", "across_input"),
    "diversity_vendi_overall": ("VENDI", "overall"),
    "diversity_vendi_overall_single_output": ("VENDI", "overall_single_output"),
}


class _AggregationFunction:
    """
    Callable wrapper for diversity metric aggregation that has __name__ attribute.

    This class-based approach ensures both picklability (for multiprocessing) and
    compatibility with lighteval's get_stderr_function which expects __name__.

    Parameters
    ----------
    metric_key : str
        The metric category (e.g., "EAD", "SBERT", "NLI")
    field : str
        The specific field to retrieve (e.g., "per_input_mean", "overall")
    """

    def __init__(self, metric_key: str, field: str):
        self.metric_key = metric_key
        self.field = field
        # "mean" in __name__ makes lighteval's get_stderr_function return
        # the trivial mean_stderr instead of bootstrap_stderr.  Bootstrap is
        # meaningless for our pre-computed per-task values and triggers a
        # multiprocessing CUDA-fork crash after vllm initialises the GPU.
        self.__name__ = f"mean_aggregate_{metric_key.lower()}_{field}"

    def __call__(self, items: List[Any]) -> float:
        return _aggregate_diversity_value(self.metric_key, self.field, items)

    def __repr__(self):
        return f"_AggregationFunction({self.metric_key!r}, {self.field!r})"


def _make_aggregation_function(metric_key: str, field: str):
    """
    Create an aggregation function with a proper __name__ attribute.

    This factory function creates callable objects instead of using functools.partial
    to ensure the resulting object has a __name__ attribute that lighteval's
    get_stderr_function can access while maintaining picklability.

    Parameters
    ----------
    metric_key : str
        The metric category (e.g., "EAD", "SBERT", "NLI")
    field : str
        The specific field to retrieve (e.g., "per_input_mean", "overall")

    Returns
    -------
    _AggregationFunction
        An aggregation function with proper __name__ attribute that is picklable
    """
    return _AggregationFunction(metric_key, field)


def _build_metric_grouping(metric_map: Dict[str, Tuple[str, str]]) -> SampleLevelMetricGrouping:
    """
    Build a metric grouping for diversity metrics.

    Creates proper wrapper functions instead of functools.partial to ensure
    compatibility with lighteval's stderr computation which requires __name__.

    Parameters
    ----------
    metric_map : Dict[str, Tuple[str, str]]
        Mapping from metric names to (metric_key, field) tuples

    Returns
    -------
    SampleLevelMetricGrouping
        A configured metric grouping object
    """
    metric_names = list(metric_map.keys())
    return SampleLevelMetricGrouping(
        metric_name=metric_names,
        higher_is_better={name: True for name in metric_map.keys()},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=DiversitySamplingCollector(
            _get_diversity_num_samples(),
            return_all=True,
            placeholder_keys=metric_names,
        ),
        corpus_level_fn={
            name: _make_aggregation_function(metric_key, field)
            for name, (metric_key, field) in metric_map.items()
        },
    )


# Metric objects for each diversity metric family (requires lighteval).
if _LIGHTEVAL_AVAILABLE:
    diversity_ead_metric = _build_metric_grouping(_EAD_METRICS)
    diversity_sbert_metric = _build_metric_grouping(_SBERT_METRICS)
    diversity_nli_metric = _build_metric_grouping(_NLI_METRICS)
    diversity_vendi_metric = _build_metric_grouping(_VENDI_METRICS)

    # Combined metric that returns all diversity scores.
    _ALL_METRICS: Dict[str, Tuple[str, str]] = {}
    _ALL_METRICS.update(_EAD_METRICS)
    _ALL_METRICS.update(_SBERT_METRICS)
    _ALL_METRICS.update(_NLI_METRICS)
    _ALL_METRICS.update(_VENDI_METRICS)
    diversity_all_metrics = _build_metric_grouping(_ALL_METRICS)

    # Register metrics with lighteval
    extend_enum(Metrics, "DIVERSITY_EAD", diversity_ead_metric)
    extend_enum(Metrics, "DIVERSITY_SBERT", diversity_sbert_metric)
    extend_enum(Metrics, "DIVERSITY_NLI", diversity_nli_metric)
    extend_enum(Metrics, "DIVERSITY_VENDI", diversity_vendi_metric)
    extend_enum(Metrics, "DIVERSITY_ALL", diversity_all_metrics)

# Export for convenience
if _LIGHTEVAL_AVAILABLE:
    diversity_ead = diversity_ead_metric
    diversity_sbert = diversity_sbert_metric
    diversity_nli = diversity_nli_metric
    diversity_vendi = diversity_vendi_metric
    diversity_all = diversity_all_metrics


