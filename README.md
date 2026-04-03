# Where Does Output Diversity Collapse in Post-Training?

Code for the paper: *Where does output diversity collapse in post-training?* 
by Constantinos Karouzos, Xingwei Tan, and Nikolaos Aletras.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Results

### 1. Diversity evaluation (EAD, SBERT, NLI, Vendi Score)

Run via lighteval with the custom task modules:

```bash
export LIGHTEVAL_DIVERSITY_NUM_SAMPLES=16

# Summarization tasks
lighteval accelerate \
    "model_name=allenai/OLMo-3-1025-7B,trust_remote_code=True,generation_parameters={temperature:0.6,top_p:0.95}" \
    tldr_diversity,cnn_dailymail_diversity,xsum_diversity \
    --custom-tasks src/evaluation/lighteval_summarization_tasks.py \
    --output-dir ./runs/diversity

# Code, instruction-following tasks
lighteval accelerate ... --custom-tasks src/evaluation/lighteval_extended_tasks.py

# Open-ended, reasoning, value-pluralism tasks
lighteval accelerate ... --custom-tasks src/evaluation/lighteval_open_ended_tasks.py
```

### 2. Quality metrics

```bash
python scripts/compute_math_quality.py        # GSM8K, MATH accuracy/MV/pass@16
python scripts/compute_code_quality.py        # HumanEval, MBPP pass@k
python scripts/compute_ifeval_quality.py      # IFEval strict/loose
python scripts/compute_summarization_quality.py  # LLM-judge win rates
python scripts/compute_wildbench_quality.py   # WB-Score
python scripts/compute_writingprompts_quality.py # WritingPrompts win rates
```

### 3. Quality-filtered diversity (QFD)

```bash
python scripts/quality_filtered_diversity.py
python scripts/quality_filtered_diversity_gsm8k.py
python scripts/compute_code_diversity.py      # AST Jaccard, UniXcoder
python scripts/compute_vendi_from_generations.py
```

### 4. Decontamination (C13 overlap)

```bash
python scripts/decontamination/measure_c13_wrapper.py index --dataset <training_dataset>
python scripts/decontamination/measure_c13_wrapper.py search \
    --train_dataset_names <dataset> --dataset <eval_dataset> --ngram_size 13
python scripts/decontamination/aggregate_results.py
```

## Project Structure

```
src/
  evaluation/    # Diversity metrics (EAD, SBERT, NLI, Vendi), lighteval task
                 #   definitions, code execution, math extraction, IFEval constraints
  generation/    # HF and vLLM generation scripts
  utils/         # Seeding
scripts/
  compute_*.py           # Quality and diversity metric computation
  quality_filtered_*.py  # QFD decomposition
  decontamination/       # C13 13-gram overlap pipeline
configs/                 # Experiment protocol configuration
```

## License

MIT License
