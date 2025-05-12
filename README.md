# LLM Translation Evaluation Framework

A framework for evaluating the translation capabilities of large language models (LLMs) using the Together API.

## Overview

This repository contains tools to evaluate LLM performance on translation tasks using standard metrics (BLEU and METEOR). It uses the Together AI platform to access state-of-the-art models like Llama 3.3.

## Features

- Support for multiple language pairs (English/German, Czech/English)
- Few-shot prompting capability
- Evaluation using BLEU and METEOR metrics
- Comet.ml integration for experiment tracking
- Configurable via environment variables and command-line arguments

## Repository Structure

```
├── together_llms.py     # Main script for running translation evaluations
├── utils.py             # Utility functions for metrics calculation
├── run_cs_en_translation.sh # Script for Czech to English translation
├── test_*_*.sh          # Various test scripts for different datasets/years
├── env.yml              # Conda environment file
├── data/                # Directory for datasets
│   └── multi30k/        # Multi30k translation dataset
└── outputs/             # Directory for output files
```

## Getting Started

### Prerequisites

- Python 3.8+
- Conda (for environment setup)
- Together AI API key

### Installation

1. Clone the repository
2. Create and activate the conda environment:
   ```
   conda env create -f env.yml
   conda activate llm-env
   ```
3. Set up your Together AI API token as an environment variable:
   ```
   export TOGETHER_API_TOKEN="your-token-here"
   ```

### Running Translation Tasks

To run a Czech to English translation evaluation:

```bash
bash run_cs_en_translation.sh
```

To customize the evaluation parameters, edit the script or run the Python script directly:

```bash
python together_llms.py \
  --src_lang "Czech" \
  --tgt_lang "English" \
  --path_src_test "data/multi30k/data/task1/raw/test_2016_val.cs" \
  --path_tgt_test "data/multi30k/data/task1/raw/test_2016_val.en" \
  --model_tag "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" \
  --max_new_tokens 128 
  # ...additional arguments
```

## Metrics

The framework evaluates translations using:
- BLEU (BiLingual Evaluation Understudy)
- METEOR (Metric for Evaluation of Translation with Explicit ORdering)

Results are saved in the outputs directory.