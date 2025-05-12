#!/bin/bash

# Set Together AI API token from environment variable or directly
TOGETHER_API_TOKEN=${TOGETHER_API_TOKEN:-"165f8b1bd9931296513d2fd2060a0a03ee5ad2976174d0f97695154481762770,25afb2c02fbc1f189447047f9654cc24df36242fc9d21f209253c01385088164,"}

# Create output directory if it doesn't exist
mkdir -p outputs/translation

# Run Czech to English translation
python together_llms.py \
  --src_lang "Czech" \
  --tgt_lang "English" \
  --path_src_test "data/multi30k/data/task1/raw/test_2016_val.cs" \
  --path_tgt_test "data/multi30k/data/task1/raw/test_2016_val.en" \
  --path_test_ref "data/multi30k/data/task1/raw/val_cs" \
  --n_references 1 \
  --model_tag "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --together_ai_token $TOGETHER_API_TOKEN \
  --output_folder "outputs/translation" \
  --max_samples_test 10 \
  --k_shot 3 \
  --path_src_examples "data/multi30k/data/task1/raw/train.cs" \
  --path_tgt_examples "data/multi30k/data/task1/raw/train.en" 