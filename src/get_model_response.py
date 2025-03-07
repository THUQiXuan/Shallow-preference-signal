from transformers import pipeline
import datasets
import torch
import json
import argparse
from tqdm import tqdm

import logging
from transformers import logging as hf_logging

# Set logging level to ERROR
hf_logging.set_verbosity_error()

def parse_arguments():
    """
    Parse command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Evaluate language model on Alpaca Farm dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (for output file naming)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize model pipeline
    pipe = pipeline("text-generation", model=args.model_path, torch_dtype=torch.bfloat16)
    
    # Load evaluation data from Alpaca dataset
    alpaca_eval_data = datasets.load_dataset("/scratch/gpfs/yw9355/.cache/huggingface/datasets/tatsu-lab___alpaca_farm/alpaca_farm_evaluation/1.0.0/e576524ca841af3c36fd6912e68e5920430928c1")['test']
    
    # Initialize a list to store model outputs
    outputs = []
    
    # Process each example in the evaluation dataset
    for example in tqdm(alpaca_eval_data, desc="Processing Evaluation Data"):
        instruction = example['instruction']
        
        # Format input for the model with system and user messages
        formatted_input = [
            {"role": "system", "content": "You are a helpful AI assistant, who always provide good answers to the questions asked by users."},
            {"role": "user", "content": instruction}
        ]
        
        # Generate output using the model
        result = pipe(formatted_input, max_new_tokens=256)
        
        # Extract generated text from model output
        output = result[0]['generated_text'][-1]["content"] if isinstance(result, list) and len(result) > 0 else ""
        
        # Record model output
        outputs.append({
            'instruction': instruction,
            'input': '',
            'output': output,
            'generator': args.model_name,
            'dataset': "alpaca_farm",
            'datasplit': "eval"
        })
    
    # Save model outputs to file
    path_to_outputs = f"/scratch/gpfs/yw9355/xq/alpaca_farm/examples/data/eval_{args.model_name}.json"
    with open(path_to_outputs, "w", encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    
    # Notify completion
    print("Generation finished.")
    print(f"Results saved to: {path_to_outputs}")
    # To evaluate with Alpaca Leaderboard, uncomment:
    # from alpaca_farm.auto_annotations import alpaca_leaderboard
    # alpaca_leaderboard(path_to_outputs, name=args.model_name)

if __name__ == "__main__":
    main()