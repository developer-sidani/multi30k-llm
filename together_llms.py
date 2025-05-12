from comet_ml import Experiment
import argparse
import pathlib
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import eval_ext
from together import Together  # Together SDK is imported here
import re
from dotenv import load_dotenv


# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def sanitize_folder_name(name):
    """
    Sanitize the folder name by replacing characters not allowed in folder names.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def load_data(args):
    src_test = [r.strip() for r in open(args.path_src_test, "r").readlines()]
    tgt_test = [r.strip() for r in open(args.path_tgt_test, "r").readlines()]
    
    ref_test = []
    if args.n_references is not None:
        ref_dict = {}
        missing_files = 0
        for i in range(args.n_references):
            # Use file extension 0.0 for target language references (when source is the input)
            ref_path = f"{args.path_test_ref}/reference{i}.0.txt"
            if os.path.exists(ref_path):
                ref_dict[i] = [
                    r.strip()
                    for r in open(ref_path, "r").readlines()
                ]
            else:
                print(f"WARNING: Reference file {ref_path} not found")
                missing_files += 1
        
        # If all reference files are missing, return empty lists
        if missing_files == args.n_references:
            print("ERROR: All reference files are missing. Cannot proceed with evaluation.")
            return [], [], [], None, None
                
        if ref_dict:
            for refs in zip(*[ref_dict[i] for i in range(len(ref_dict))]):
                ref_test.append(list(refs))
    
    # Verify lengths match
    if ref_test and len(src_test) != len(ref_test):
        print(f"WARNING: Source test size ({len(src_test)}) != Reference test size ({len(ref_test)})")
        # Handle the mismatch by taking the common length
        common_length = min(len(src_test), len(ref_test))
        print(f"Using common length: {common_length}")
        src_test = src_test[:common_length]
        ref_test = ref_test[:common_length]
    
    if args.max_samples_test is not None:
        src_test = src_test[: args.max_samples_test]
        tgt_test = tgt_test[: args.max_samples_test]
        if ref_test:
            ref_test = ref_test[: args.max_samples_test]
    
    src_examples, tgt_examples = None, None
    
    if args.path_src_examples is not None and args.path_tgt_examples is not None:
        src_examples = [
            r.strip() for r in open(args.path_src_examples, "r").readlines()
        ]
        tgt_examples = [
            r.strip() for r in open(args.path_tgt_examples, "r").readlines()
        ]
        assert len(src_examples) == len(tgt_examples)
    
    return (
        src_test,
        tgt_test,
        ref_test,
        src_examples,
        tgt_examples,
    )


def together_generate(prompt, args, client):
    """
    Generate a completion for a given prompt using the Together SDK.
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": args.model_tag,
            "messages": messages,
            "max_tokens": args.max_new_tokens,
        }
        if args.temperature is not None:
            params["temperature"] = args.temperature
        if args.num_beams is not None:
            params["num_beams"] = args.num_beams
        
        print(f"Sending request to Together API with model {args.model_tag}")
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in Together API call: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_prediction_for_prompt(prompt, args, client):
    """
    Attempts to generate a prediction for a given prompt.
    If an error occurs, rotate through the list of tokens.
    Returns the generated text and the (possibly updated) client.
    """
    num_tokens = len(args.together_ai_tokens)
    attempts = 0
    while attempts < num_tokens:
        try:
            print(f"Attempting to generate with token index {args.current_token_index}")
            generated_text = together_generate(prompt, args, client)
            if not generated_text or generated_text.strip() == "":
                print("WARNING: Together API returned empty response")
                raise ValueError("Empty response from API")
            return generated_text, client
        except Exception as e:
            print(f"Error occurred with token api_key: {e}")
            import traceback
            traceback.print_exc()
            attempts += 1
            # Rotate to next token
            args.current_token_index = (args.current_token_index + 1) % num_tokens
            new_token = args.together_ai_tokens[args.current_token_index]
            print(f"Switching to token index: {args.current_token_index}")
            client = Together(api_key=new_token)
    print("All tokens failed for this prompt; returning generic message.")
    return "Unable to generate a response", client


def generate_predictions(data, args, client):
    """
    Iterates over each prompt in the data and generates predictions using Together SDK.
    Uses fallback tokens if errors occur.
    """
    predictions = []
    print(f"Starting to generate predictions for {len(data)} prompts")
    for i, prompt in enumerate(tqdm(data, total=len(data))):
        try:
            generated_text, client = generate_prediction_for_prompt(prompt, args, client)
            if not generated_text or generated_text.strip() == "":
                print(f"WARNING: Empty prediction for prompt {i}")
                # Use a fallback value instead of empty string
                generated_text = "No valid prediction generated"
            predictions.append(generated_text)
        except Exception as e:
            print(f"ERROR: Failed to generate prediction for prompt {i}: {e}")
            predictions.append("Generation failed")
    
    print(f"Generated {len(predictions)} predictions")
    
    # Ensure we have a prediction for every input
    if len(predictions) < len(data):
        print(f"WARNING: Fewer predictions ({len(predictions)}) than inputs ({len(data)}). Adding placeholders.")
        predictions.extend(["No prediction"] * (len(data) - len(predictions)))
    
    # Log warning if no predictions were made
    if len(predictions) == 0:
        print("WARNING: No predictions were generated!")
        # Add a dummy prediction for each input to prevent errors
        predictions = ["No prediction"] * len(data)
        
    return predictions, client


def test(data, args, experiment, client):
    print("\nStart translation test")
    src = data[0]
    predictions, client = generate_predictions(data[1], args, client)
    references = data[2]
    print("\nTest completed")

    # Validate predictions
    print(f"Validation check - predictions length: {len(predictions)}")
    print(f"Source length: {len(src)}")
    print(f"References length: {len(references)}")
    
    # Ensure predictions is not empty, fill with placeholders if needed
    if len(predictions) == 0 or (len(predictions) > 0 and all(not p or p.strip() == "" for p in predictions)):
        print("WARNING: Empty predictions detected. Using placeholder values.")
        predictions = ["No prediction generated"] * len(src)
    
    # First, save the generated CSV file
    try:
        # Handle mismatched lengths between predictions and references
        if len(predictions) != len(references):
            print(f"WARNING: Length mismatch - predictions: {len(predictions)}, references: {len(references)}")
            
            # If we have fewer predictions than sources, pad predictions
            if len(predictions) < len(src):
                print(f"Padding predictions to match source length ({len(src)})")
                predictions.extend(["No prediction"] * (len(src) - len(predictions)))
            
            # If we have fewer references than predictions, use only the available references
            # Or if we have more references than predictions, truncate references
            if len(references) != len(predictions):
                common_length = min(len(predictions), len(references))
                print(f"Using common length for evaluation: {common_length}")
                predictions = predictions[:common_length]
                references = references[:common_length]
                src = src[:common_length] if len(src) > common_length else src
        
        # Create DataFrame with matched lengths
        df = pd.DataFrame()
        df["Source"] = src[:len(predictions)]
        df["Source (prompt)"] = data[1][:len(predictions)]
        df["Translation (generated)"] = predictions
        
        # Add references if available
        if len(references) > 0:
            ref = np.array(references)
            for i in range(min(args.n_references, ref.shape[1])):
                df[f"ref {i+1}"] = ref[:, i]
        
        csv_path = os.path.join(args.output_folder, "translation_test.csv")
        df.to_csv(csv_path, sep=",", header=True, index=False)
        print(f"Successfully saved translation test CSV to {csv_path}")
    except Exception as e:
        print(f"Error saving translation test CSV: {e}")
        import traceback
        traceback.print_exc()

    # Then, attempt to compute the metrics
    metrics_names = ["bleu", "meteor"]
    metrics = {}
    
    try:
        print(f"Evaluating metrics... predictions length: {len(predictions)}")
        print(f"References length: {len(references)}")
        
        # Check if predictions are valid
        valid_metrics = True
        if len(predictions) == 0:
            print("ERROR: No predictions were generated!")
            valid_metrics = False
        
        if len(predictions) != len(references):
            print(f"WARNING: Length mismatch before evaluation. Predictions: {len(predictions)}, References: {len(references)}")
            valid_metrics = False
        
        if valid_metrics:
            metrics = eval_ext(
                metrics_names,
                predictions,
                references,
                args,
                return_metrics=True,
            )
            
            # Get direction for metric naming
            direction = os.path.basename(args.output_folder)
            if not direction:
                direction = "translation"
                
            # Update metrics with direction prefix
            direction_metrics = {}
            for k, v in metrics.items():
                if k != 'method':
                    direction_metrics[f"{direction}_{k}"] = v
                else:
                    direction_metrics[k] = v
                    
            metrics = direction_metrics
        else:
            metrics = {}
    except Exception as e:
        print(f"Warning: Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        metrics = {}

    if experiment is not None:
        try:
            with experiment.test():
                # Log metrics with direction in name
                experiment.log_metrics(metrics)
                # Log table with direction in filename
                table_name = os.path.join(args.output_folder, "translation_test.csv")
                experiment.log_table(table_name, tabular_data=df, headers=True)
        except Exception as e:
            print(f"Error logging to experiment: {e}")
    
    return client

def together_tst(args, experiment):
    # Validate critical parameters before making any API calls
    if not args.together_ai_tokens:
        print("ERROR: Together API tokens are missing but required for this model")
        return None
            
    if not args.max_new_tokens or args.max_new_tokens <= 0:
        print(f"WARNING: Invalid max_new_tokens value ({args.max_new_tokens}), using default of 64")
        args.max_new_tokens = 64
    
    # Proceed with data loading and processing
    (
        src_test,
        tgt_test,
        ref_test,
        src_examples,
        tgt_examples,
    ) = load_data(args)
    print(f"Loaded {len(src_test)} examples for test source language {args.src_lang}.")
    print(f"Loaded {len(tgt_test)} examples for test target language {args.tgt_lang}.")

    # Process few-shot examples if provided.
    examples_src, examples_tgt = None, None
    if args.k_shot > 0 and src_examples is not None and tgt_examples is not None:
        # Build a list of valid examples (non-empty and not starting with a digit)
        valid_examples = [
            (s, t)
            for s, t in zip(src_examples, tgt_examples)
            if s.strip()
            and t.strip()
            and not s.strip()[0].isdigit()
            and not t.strip()[0].isdigit()
        ]
        if len(valid_examples) < args.k_shot:
            raise ValueError("Not enough valid examples for k_shot")
        ix_few_shot = random.sample(range(len(valid_examples)), args.k_shot)
        examples_src = [valid_examples[i][0] for i in ix_few_shot]
        examples_tgt = [valid_examples[i][1] for i in ix_few_shot]

    # Instantiate the Together SDK client
    client = Together(api_key=args.together_ai_tokens[0])
    
    # ---- FORWARD DIRECTION (source -> target) ----
    print(f"\n=== Testing forward direction: {args.src_lang} -> {args.tgt_lang} ===")
    
    # Build test prompts for translation (source to target)
    forward_prompt = (
        f"Translate the following text from {args.src_lang} to {args.tgt_lang}. "
        f"Return only the translated text without additional text or explanations. "
    )
    
    if args.k_shot > 0:
        prompt_few_shot = (
            f"Here are some examples of {args.src_lang} sentences and their {args.tgt_lang} translations:\n\n"
        )
        if examples_src is not None and examples_tgt is not None:
            for src_ex, tgt_ex in zip(examples_src, examples_tgt):
                prompt_few_shot += f"Input ({args.src_lang}): {src_ex}\nOutput ({args.tgt_lang}): {tgt_ex}\n\n"
        forward_prompt += prompt_few_shot
    
    forward_prompt += f"Input ({args.src_lang}): [INPUT_TEXT]\n"
    forward_prompt += f"Output ({args.tgt_lang}): "
    
    src_test_prompt = [
        forward_prompt.replace("[INPUT_TEXT]", s) for s in src_test
    ]

    # Save original output folder
    original_output_folder = args.output_folder
    
    # Update output folder for the forward direction
    args.output_folder = os.path.join(original_output_folder, f"{args.src_lang}_to_{args.tgt_lang}")
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    
    # Call test with the appropriate tuples: (source texts, prompt texts, reference texts)
    client = test(
        (src_test, src_test_prompt, ref_test),
        args,
        experiment,
        client,
    )
    
    # ---- REVERSE DIRECTION (target -> source) ----
    print(f"\n=== Testing reverse direction: {args.tgt_lang} -> {args.src_lang} ===")
    
    # Build reference data for the reverse direction
    # We need to check if there are reference files for the reverse direction
    reverse_ref_test = []
    if args.n_references is not None:
        ref_dict = {}
        for i in range(args.n_references):
            # Use file extension 0.1 for source language references (when target is the input)
            ref_path = f"{args.path_test_ref}/reference{i}.1.txt"
            if os.path.exists(ref_path):
                ref_dict[i] = [r.strip() for r in open(ref_path, "r").readlines()]
            else:
                print(f"Warning: Reference file {ref_path} not found for reverse direction")
                # Use source text as reference if reverse reference not available
                ref_dict[i] = src_test
        
        if ref_dict:
            for refs in zip(*[ref_dict[i] for i in range(len(ref_dict))]):
                reverse_ref_test.append(list(refs))
    
    # If no reference files found for reverse direction, use source text as reference
    if not reverse_ref_test:
        print("No reference files found for reverse direction, using source text as reference")
        reverse_ref_test = [[s] for s in src_test]
    
    # Build test prompts for translation (target to source)
    reverse_prompt = (
        f"Translate the following text from {args.tgt_lang} to {args.src_lang}. "
        f"Return only the translated text without additional text or explanations. "
    )
    
    if args.k_shot > 0:
        prompt_few_shot = (
            f"Here are some examples of {args.tgt_lang} sentences and their {args.src_lang} translations:\n\n"
        )
        if examples_tgt is not None and examples_src is not None:
            for tgt_ex, src_ex in zip(examples_tgt, examples_src):
                prompt_few_shot += f"Input ({args.tgt_lang}): {tgt_ex}\nOutput ({args.src_lang}): {src_ex}\n\n"
        reverse_prompt += prompt_few_shot
    
    reverse_prompt += f"Input ({args.tgt_lang}): [INPUT_TEXT]\n"
    reverse_prompt += f"Output ({args.src_lang}): "
    
    tgt_test_prompt = [
        reverse_prompt.replace("[INPUT_TEXT]", t) for t in tgt_test
    ]
    
    # Update output folder for the reverse direction
    args.output_folder = os.path.join(original_output_folder, f"{args.tgt_lang}_to_{args.src_lang}")
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    
    # Call test with the appropriate tuples: (target texts as source, prompt texts, reference texts)
    client = test(
        (tgt_test, tgt_test_prompt, reverse_ref_test),
        args,
        experiment,
        client,
    )
    
    # Restore original output folder
    args.output_folder = original_output_folder
    
    return client

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_lang",
        type=str,
        dest="src_lang",
        help="Source language for the translation task.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        dest="tgt_lang",
        help="Target language for the translation task.",
    )
    parser.add_argument(
        "--path_src_examples", type=str, dest="path_src_examples", help="Path to source examples for few-shot"
    )
    parser.add_argument(
        "--path_tgt_examples", type=str, dest="path_tgt_examples", help="Path to target examples for few-shot"
    )
    parser.add_argument(
        "--path_src_test",
        type=str,
        dest="path_src_test",
        help="Path to source language dataset for test.",
    )
    parser.add_argument(
        "--path_tgt_test",
        type=str,
        dest="path_tgt_test",
        help="Path to target language dataset for test.",
    )
    parser.add_argument(
        "--path_test_ref",
        type=str,
        dest="path_test_ref",
        help="Path to human references for test.",
    )
    parser.add_argument(
        "--max_samples_test",
        type=int,
        dest="max_samples_test",
        default=None,
        help="Max number of examples to retain. None for all available examples.",
    )
    parser.add_argument(
        "--n_references",
        type=int,
        dest="n_references",
        default=None,
        help="Number of human references for test.",
    )
    parser.add_argument("--model_tag", type=str, dest="model_tag", help="")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        dest="max_new_tokens",
        default=64,
        help="Max sequence length",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument(
        "--output_folder", type=str, dest="output_folder", default=None, help=""
    )
    parser.add_argument(
        "--use_gpu", action="store_true", dest="use_gpu", default=False, help=""
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        dest="num_workers",
        default=4,
        help="Number of workers for dataloaders.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        dest="pin_memory",
        default=False,
        help="Whether to pin memory for data on GPU during data loading.",
    )
    parser.add_argument(
        "--comet_logging",
        action="store_true",
        dest="comet_logging",
        default=True,
        help="Enable Comet logging",
    )
    parser.add_argument(
        "--together_ai_token",
        type=str,
        default=None,
        help="Comma-separated list of Together AI API Keys (e.g. token1,token2,token3)",
    )
    args = parser.parse_args()

    if args.output_folder is None:
        raise ValueError("Please specify an output folder using --output_folder")

    # Get Comet configuration from environment variables
    comet_api_key = os.getenv("COMET_API_KEY")
    comet_project_name = os.getenv("COMET_PROJECT_NAME")
    comet_workspace = os.getenv("COMET_WORKSPACE")

    # Parse the together token into a list of tokens.
    if args.together_ai_token is not None:
        args.together_ai_tokens = [
            tok.strip() for tok in args.together_ai_token.split(",") if tok.strip()
        ]
        print(f"Loaded {len(args.together_ai_tokens)} Together API tokens")
        
        # Validate token length/format
        valid_tokens = []
        for i, token in enumerate(args.together_ai_tokens):
            if len(token) < 10:  # Simple validation for token length
                print(f"WARNING: Token at index {i} seems too short ({len(token)} chars), might be invalid")
            else:
                valid_tokens.append(token)
        
        if len(valid_tokens) == 0:
            print("ERROR: No valid Together API tokens provided!")
            print("A valid Together API token is required!")
        
        args.together_ai_tokens = valid_tokens
    else:
        args.together_ai_tokens = []
        print("WARNING: No Together API tokens provided")
        print("A valid Together API token is required!")

    # Initialize current token index
    args.current_token_index = 0

    sanitized_model_tag = sanitize_folder_name(args.model_tag)
    args.output_folder = (
        os.path.join(args.output_folder, sanitized_model_tag)
        + f"_K_shot_{args.k_shot}/"
    )
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    # Generate a descriptive experiment name for Comet
    experiment_prefix = os.environ.get("EXPERIMENT_PREFIX", "Translation")
    experiment_name = f"{experiment_prefix} {args.src_lang}-{args.tgt_lang}: {sanitized_model_tag}_K_shot_{args.k_shot}"
    print(f"Experiment name: {experiment_name}")
    
    # Also add the experiment name to args for reference
    args.experiment_name = experiment_name

    hyper_params = {}
    print("Arguments summary:\n")
    for key, value in vars(args).items():
        hyper_params[key] = value
        print(f"\t{key}:\t\t{value}")
    with open(f"{args.output_folder}/args.txt", "w") as f:
        f.write("\n".join([f"{k}={v}" for k, v in hyper_params.items()]))
    print(
        f"THE DEVICE IS: { 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu' }"
    )
    print(f"CUDA is available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # New check: try a dummy CUDA operation to catch busy/unavailable GPU issues.
    if args.use_gpu and device == "cuda":
        try:
            torch.cuda.empty_cache()
            # Create a small tensor on the GPU.
            dummy_tensor = torch.tensor([1.0], device="cuda")
        except RuntimeError as e:
            print("Error: CUDA device is busy or unavailable:", e)
            import sys
            sys.exit(1)

    if args.comet_logging and comet_api_key:
        experiment = Experiment(
            api_key=comet_api_key,
            project_name=comet_project_name,
            workspace=comet_workspace,
        )
        # Set the experiment name
        experiment.set_name(experiment_name)
        experiment.log_parameters(hyper_params)
    else:
        experiment = None

    # Use Together SDK for inference
    together_tst(args, experiment)

