import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.nn.functional as F
import numpy as np
import tqdm

# -----------------------------
# Helper function to parse arguments
# -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Token switch control experiment.")
    # Model paths
    parser.add_argument(
        "--dpo_model_path",
        type=str,
        default="/scratch/gpfs/yw9355/model/models--RLHFlow--LLaMA3-iterative-DPO-final/snapshots/8c929ad1d79e0b23685c1cbb9316b53356f08852",
        help="Path to the DPO model"
    )
    parser.add_argument(
        "--ref_model_path",
        type=str,
        default="/scratch/gpfs/yw9355/model/models--RLHFlow--LLaMA3-SFT/snapshots/1bedbf1899806bf585bd8d0833541d28af017dce",
        help="Path to the reference model"
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="/scratch/gpfs/yw9355/model/models--sfairXC--FsfairX-LLaMA3-RM-v0.1/snapshots/94fad49f1b3227aa8b566f415a335adb68ec544c",
        help="Path to the reward model"
    )
    
    # Dataset and experiment parameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ultrafeedback", "alpaca"],
        required=True,
        help="Which dataset to use: ultrafeedback or alpaca."
    )
    parser.add_argument(
        "--t_values",
        nargs="+",
        type=int,
        required=True,
        help="List of T values for partial usage of DPO model."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per instruction."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per instruction."
    )
    return parser.parse_args()

# -----------------------------
# Helper function to load instructions
# -----------------------------
def load_instructions_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Adjust how you retrieve instructions if needed
    # e.g., if data is a list of strings or list of dicts
    return data

# -----------------------------
# Main script
# -----------------------------
def main():
    args = parse_arguments()

    # Decide which dataset to load
    if args.dataset == "ultrafeedback":
        instructions = load_instructions_from_json("instructions_ultrafeedback.json")
    else:
        instructions = load_instructions_from_json("instructions_alpaca.json")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load models and tokenizers
    # -----------------------------
    print("Loading DPO model...")
    dpo_model = AutoModelForCausalLM.from_pretrained(args.dpo_model_path).to(device)
    print("Loading Reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_path).to(device)
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.dpo_model_path)

    print("Loading Reward model and tokenizer...")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=args.reward_model_path,
        device=0 if device == "cuda" else -1,
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}  # If your GPU supports bfloat16
    )
    pipe_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    # -----------------------------
    # For each t, run experiment
    # -----------------------------
    for t in args.t_values:
        print(f"Processing t = {t} ...")

        # For collecting overall results
        scores_per_question = []
        kl_scores_per_question = []

        # We will also keep detailed results for each instruction
        detailed_results = []

        # Loop over instructions
        for question in instructions:
            question_text = question.strip() if isinstance(question, str) else question
            # For each question, collect multiple samples
            question_scores = []
            question_kl_scores = []

            # Generate samples
            for _ in tqdm.tqdm(range(args.num_samples), desc=f"Sampling for t={t}", unit="sample"):
                # Example: how you apply the chat template
                # If your tokenizer provides a custom function, use it
                messages = [{"role": "user", "content": question_text}]
                model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

                if isinstance(model_inputs, torch.Tensor):
                    input_ids = model_inputs
                    attention_mask = torch.ones_like(input_ids, device=device)
                else:
                    input_ids = model_inputs["input_ids"]
                    attention_mask = model_inputs["attention_mask"]

                generated_tokens = []
                kl_divs = []

                for token_index in range(args.max_new_tokens):
                    # DPO model
                    with torch.no_grad():
                        dpo_outputs = dpo_model(input_ids=input_ids, attention_mask=attention_mask)
                        dpo_logits = dpo_outputs.logits

                    # Reference model
                    with torch.no_grad():
                        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                        ref_logits = ref_outputs.logits

                    dpo_last_logits = dpo_logits[:, -1, :]
                    ref_last_logits = ref_logits[:, -1, :]

                    pi_dpo = F.softmax(dpo_last_logits, dim=-1)
                    pi_ref = F.softmax(ref_last_logits, dim=-1)

                    epsilon = 1e-12
                    kl_div = torch.sum(
                        pi_dpo * (torch.log(pi_dpo + epsilon) - torch.log(pi_ref + epsilon)),
                        dim=-1
                    ).item()

                    # Decide which model to sample from
                    if token_index < t:
                        sampling_logits = dpo_last_logits
                        kl_record = kl_div
                    else:
                        sampling_logits = ref_last_logits
                        kl_record = 0.0

                    kl_divs.append(kl_record)

                    # Sample next token
                    pi_sample = F.softmax(sampling_logits, dim=-1)
                    next_token = torch.multinomial(pi_sample, num_samples=1).squeeze(-1)
                    generated_tokens.append(next_token.item())

                    next_token = next_token.unsqueeze(1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)],
                        dim=1
                    )

                    if next_token[0].item() == tokenizer.eos_token_id:
                        break

                # Decode and compute reward
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                chat = [
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": output_text}
                ]
                # Build text for reward model
                test_texts = [
                    rm_tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=False
                    ).replace(rm_tokenizer.bos_token, "")
                ]
                pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
                reward_score = pipe_outputs[0][0]["score"]
                question_scores.append(reward_score)

                average_kl = sum(kl_divs)
                question_kl_scores.append(average_kl)

            # Compute average for this question
            avg_score = sum(question_scores) / len(question_scores)
            avg_kl = sum(question_kl_scores) / len(question_kl_scores)
            scores_per_question.append(avg_score)
            kl_scores_per_question.append(avg_kl)

            # Store detailed info for this question
            detailed_results.append({
                "question": question_text,
                "average_reward_score": avg_score,
                "average_KL_divergence": avg_kl
            })

        # Overall averages across all questions
        overall_avg_reward = sum(scores_per_question) / len(scores_per_question)
        overall_avg_kl = sum(kl_scores_per_question) / len(kl_scores_per_question)

        # Construct final results dictionary
        results_dict = {
            "t": t,
            "dataset": args.dataset,
            "dpo_model": args.dpo_model_path,
            "ref_model": args.ref_model_path,
            "reward_model": args.reward_model_path,
            "overall_average_reward_score": overall_avg_reward,
            "overall_average_KL_divergence": overall_avg_kl,
            "details": detailed_results
        }

        # Save results: t_{t_value}_dataset_{dataset}.json
        out_filename = f"t_{t}_dataset_{args.dataset}.json"
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4)

        print(f"Saved results to {out_filename}")

if __name__ == "__main__":
    main()