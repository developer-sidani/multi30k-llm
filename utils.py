import os, pickle, random
import numpy as np
import torch
import evaluate


def compute_metric(predictions, references, metric_name, lang='en'):
    bleu = evaluate.load('sacrebleu')
    meteor = evaluate.load('meteor')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # predictions = lists | references = list of lists
    scores = []
    if metric_name in ['bleu', 'meteor']:
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            try:
                if pred is None or (isinstance(pred, str) and pred.strip() == ""):
                    print(f"WARNING: Empty or None prediction at index {i}, using placeholder")
                    pred = "empty prediction"
                
                if metric_name == 'bleu':
                    res = bleu.compute(predictions=[pred], references=[ref])
                    scores.append(res['score'])
                elif metric_name == 'meteor':
                    meteor_scores = []
                    for r in ref:
                        if r is None or (isinstance(r, str) and r.strip() == ""):
                            print(f"WARNING: Empty or None reference, using placeholder")
                            r = "empty reference"
                        res = meteor.compute(predictions=[pred], references=[r])
                        meteor_scores.append(res['meteor'])
                    scores.append(max(meteor_scores))
            except Exception as e:
                print(f"Error computing {metric_name} for prediction {i}: {e}")
                print(f"Prediction: '{pred}'")
                print(f"Reference: {ref}")
                if metric_name == 'bleu':
                    scores.append(0.0)
                elif metric_name == 'meteor':
                    scores.append(0.0)
    else:
        raise Exception(f"Metric {metric_name} is not supported.")
    
    if len(scores) == 0:
        print(f"WARNING: No scores were computed for {metric_name}")
        return [0.0] * len(predictions)
    
    return scores


def eval_ext(metric_names, predictions, references, args=None, return_metrics=False):
    print("COMPUTING METRICS IN EVAL EXTENSION FUNCTION")
    # add non-existing fields to args for compatibility
    if 'batch_size' not in args: args.batch_size = 64
    if 'max_sequence_length' not in args: args.max_sequence_length = 64
    if 'lowercase_out' not in args: args.lowercase_out = False
    if 'lowercase_ref' not in args: args.lowercase_ref = False
    if 'method' not in args: args.method = args.model_tag.replace('/', '_')
    if 'pred_base_path' not in args: args.pred_base_path = args.output_folder
    #####################

    scores_bleu, scores_meteor = [], []

    # Process data in batches to avoid memory issues
    for i in range(0, len(predictions), args.batch_size):
        batch_preds = predictions[i:i+args.batch_size]
        batch_refs = references[i:i+args.batch_size]
        
        if args.lowercase_out:
            batch_preds = [out.lower() for out in batch_preds]
        if args.lowercase_ref:
            batch_refs = [[ref.lower() for ref in refs] for refs in batch_refs]
        
        if 'bleu' in metric_names:
            scores_bleu.extend(compute_metric(batch_preds, batch_refs, 'bleu'))
        if 'meteor' in metric_names:
            scores_meteor.extend(compute_metric(batch_preds, batch_refs, 'meteor'))
    
    avg_bleu = np.mean(scores_bleu) if 'bleu' in metric_names else -100
    avg_meteor = np.mean(scores_meteor) if 'meteor' in metric_names else -100

    metrics = {
        'method': args.method,
        'BLEU': avg_bleu,
        'METEOR': avg_meteor
    }
    
    print(f"Metrics are: {metrics}")

    base_path = args.pred_base_path
    suffix = f'{args.method}_test'
    pickle.dump(metrics, open(f"{base_path}metrics_{suffix}.pickle", 'wb'))

    print(f"Metrics are: {metrics}")

    for m, v in metrics.items():
        if v != -100:
            print(f'{m}: {v}')
    
    if return_metrics: 
        return metrics
