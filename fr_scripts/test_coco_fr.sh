#!/bin/bash

# Set Together AI API token from environment variable or directly
TOGETHER_API_TOKEN=${TOGETHER_API_TOKEN:-"165f8b1bd9931296513d2fd2060a0a03ee5ad2976174d0f97695154481762770"}

# Create output directory if it doesn't exist
mkdir -p outputs/translation

# Run English to French translation
python together_llms.py \
  --src_lang "English" \
  --tgt_lang "French" \
  --path_src_test "data/multi30k/data/task1/raw/test_2017_mscoco.en" \
  --path_tgt_test "data/multi30k/data/task1/raw/test_2017_mscoco.fr" \
  --path_test_ref "data/multi30k/data/task1/raw/fr/2017_mscoco" \
  --n_references 1 \
  --model_tag "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" \
  --max_new_tokens 64 \
  --temperature 0.7 \
  --together_ai_token $TOGETHER_API_TOKEN \
  --output_folder "outputs/translation" \
  --max_samples_test 1000 \
  --k_shot 0 \
  --path_src_examples "data/multi30k/data/task1/raw/train.en" \
  --path_tgt_examples "data/multi30k/data/task1/raw/train.fr"



python together_llms.py \
  --src_lang "English" \
  --tgt_lang "French" \
  --path_src_test "data/multi30k/data/task1/raw/test_2017_mscoco.en" \
  --path_tgt_test "data/multi30k/data/task1/raw/test_2017_mscoco.fr" \
  --path_test_ref "data/multi30k/data/task1/raw/fr/2017_mscoco" \
  --n_references 1 \
  --model_tag "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" \
  --max_new_tokens 64 \
  --temperature 0.7 \
  --together_ai_token $TOGETHER_API_TOKEN \
  --output_folder "outputs/translation" \
  --max_samples_test 1000 \
  --k_shot 5 \
  --path_src_examples "data/multi30k/data/task1/raw/train.en" \
  --path_tgt_examples "data/multi30k/data/task1/raw/train.fr"