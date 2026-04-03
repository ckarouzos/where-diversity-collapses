"""Registry and filtering helpers for OLMo-3 model checkpoints.

Contains the 13 checkpoints studied in the paper (all 7B scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set


@dataclass(frozen=True)
class OLMo3Model:
    name: str
    size: str
    family: str
    stage: str
    series: str
    domain: Optional[str] = None


_OLMO3_MODELS: Sequence[OLMo3Model] = (
    # Base
    OLMo3Model(
        name="allenai/Olmo-3-1025-7B",
        size="7B",
        family="base",
        stage="base",
        series="3",
    ),
    # Think lineage (SFT → DPO → RL)
    OLMo3Model(
        name="allenai/Olmo-3-7B-Think-SFT",
        size="7B",
        family="think",
        stage="sft",
        series="3",
    ),
    OLMo3Model(
        name="allenai/Olmo-3-7B-Think-DPO",
        size="7B",
        family="think",
        stage="dpo",
        series="3",
    ),
    OLMo3Model(
        name="allenai/Olmo-3-7B-Think",
        size="7B",
        family="think",
        stage="final",
        series="3",
    ),
    # Instruct lineage (SFT → DPO → RL)
    OLMo3Model(
        name="allenai/Olmo-3-7B-Instruct-SFT",
        size="7B",
        family="instruct",
        stage="sft",
        series="3",
    ),
    OLMo3Model(
        name="allenai/Olmo-3-7B-Instruct-DPO",
        size="7B",
        family="instruct",
        stage="dpo",
        series="3",
    ),
    OLMo3Model(
        name="allenai/Olmo-3-7B-Instruct",
        size="7B",
        family="instruct",
        stage="final",
        series="3",
    ),
    # RL-Zero variants
    OLMo3Model(
        name="allenai/Olmo-3-7B-RL-Zero-Math",
        size="7B",
        family="rl-zero",
        stage="rl-zero",
        series="3",
        domain="math",
    ),
    OLMo3Model(
        name="allenai/Olmo-3-7B-RL-Zero-Code",
        size="7B",
        family="rl-zero",
        stage="rl-zero",
        series="3",
        domain="code",
    ),
    OLMo3Model(
        name="allenai/Olmo-3-7B-RL-Zero-IF",
        size="7B",
        family="rl-zero",
        stage="rl-zero",
        series="3",
        domain="if",
    ),
    OLMo3Model(
        name="allenai/Olmo-3-7B-RL-Zero-General",
        size="7B",
        family="rl-zero",
        stage="rl-zero",
        series="3",
        domain="general",
    ),
    OLMo3Model(
        name="allenai/Olmo-3.1-7B-RL-Zero-Math",
        size="7B",
        family="rl-zero",
        stage="rl-zero",
        series="3.1",
        domain="math",
    ),
    OLMo3Model(
        name="allenai/Olmo-3.1-7B-RL-Zero-Code",
        size="7B",
        family="rl-zero",
        stage="rl-zero",
        series="3.1",
        domain="code",
    ),
)


def list_models() -> List[OLMo3Model]:
    return list(_OLMO3_MODELS)


def _normalize_values(values: Optional[Iterable[str]]) -> Set[str]:
    if not values:
        return set()
    normalized: Set[str] = set()
    for value in values:
        for chunk in str(value).split(","):
            cleaned = chunk.strip()
            if cleaned:
                normalized.add(cleaned.lower())
    return normalized


def filter_models(
    models: Sequence[OLMo3Model],
    sizes: Optional[Iterable[str]] = None,
    families: Optional[Iterable[str]] = None,
    stages: Optional[Iterable[str]] = None,
    series: Optional[Iterable[str]] = None,
    domains: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> List[OLMo3Model]:
    size_filter = _normalize_values(sizes)
    family_filter = _normalize_values(families)
    stage_filter = _normalize_values(stages)
    series_filter = _normalize_values(series)
    domain_filter = _normalize_values(domains)
    name_filter = _normalize_values(names)

    filtered: List[OLMo3Model] = []
    for model in models:
        if size_filter and model.size.lower() not in size_filter:
            continue
        if family_filter and model.family.lower() not in family_filter:
            continue
        if stage_filter and model.stage.lower() not in stage_filter:
            continue
        if series_filter and model.series.lower() not in series_filter:
            continue
        if domain_filter and (model.domain or "").lower() not in domain_filter:
            continue
        if name_filter and model.name.lower() not in name_filter:
            continue
        filtered.append(model)
    return filtered


def format_models_table(models: Sequence[OLMo3Model]) -> str:
    lines = []
    header = f"{'MODEL':<40} {'SIZE':<4} {'FAMILY':<9} {'STAGE':<7} {'SERIES':<4} {'DOMAIN':<7}"
    lines.append(header)
    lines.append("-" * len(header))
    for model in models:
        lines.append(
            f"{model.name:<40} {model.size:<4} {model.family:<9} "
            f"{model.stage:<7} {model.series:<4} {model.domain or '-':<7}"
        )
    return "\n".join(lines)


def all_7b_model_names() -> List[str]:
    return [m.name for m in _OLMO3_MODELS]


def all_post_trained_7b_names() -> List[str]:
    return [m.name for m in _OLMO3_MODELS if m.family != "base"]


BASE_MODEL_NAME = "allenai/Olmo-3-1025-7B"


def _short_name(full_name: str) -> str:
    return full_name.split("/", 1)[-1] if "/" in full_name else full_name


def all_7b_short_names() -> List[str]:
    return [_short_name(m.name) for m in _OLMO3_MODELS]


def all_post_trained_7b_short_names() -> List[str]:
    return [_short_name(m.name) for m in _OLMO3_MODELS if m.family != "base"]


BASE_MODEL_SHORT = _short_name(BASE_MODEL_NAME)
