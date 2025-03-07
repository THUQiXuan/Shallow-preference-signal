import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.nn.functional as F
import tqdm


# Load instructions from JSON file
def load_instructions_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def main(args):
    # Set device to GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models and tokenizer
    dpo_model = AutoModelForCausalLM.from_pretrained(args.dpo_model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.dpo_model_name)

    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=args.reward_model_name,
        device=0,
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    pipe_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    # Load dataset
    if args.dataset == "ultrafeedback":
        instructions = load_instructions_from_json("instructions_ultrafeedback.json")
        
    else:
        instructions = load_instructions_from_json("instructions_alpaca.json")

    num_samples = 5
    overall_results = []

    for b in args.threshold_values:
        print(f"Processing for threshold b = {b}\n")
        scores_per_question = []  # Store average reward scores for all questions under this threshold
        kl_scores_per_question = []  # Store average KL scores for all questions under this threshold

        for question in instructions[0:100]:
            print(f"Processing question: {question}\n")

            question_scores = []
            question_kl_scores = []

            for _ in tqdm.tqdm(range(num_samples), desc="Generating Samples", unit="sample"):
                messages = [{"role": "user", "content": question}]
                model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

                if isinstance(model_inputs, torch.Tensor):
                    input_ids = model_inputs
                    attention_mask = torch.ones_like(input_ids, device=device)
                else:
                    input_ids = model_inputs["input_ids"]
                    attention_mask = model_inputs["attention_mask"]

                max_new_tokens = 512
                generated_tokens = []
                kl_divs = []

                for _ in tqdm.tqdm(range(max_new_tokens), desc="Processing Tokens", unit="token"):
                    with torch.no_grad():
                        dpo_outputs = dpo_model(input_ids=input_ids, attention_mask=attention_mask)
                        dpo_logits = dpo_outputs.logits
                        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                        ref_logits = ref_outputs.logits

                    dpo_last_logits = dpo_logits[:, -1, :]
                    ref_last_logits = ref_logits[:, -1, :]

                    pi_dpo = F.softmax(dpo_last_logits, dim=-1)
                    pi_ref = F.softmax(ref_last_logits, dim=-1)

                    kl_div = torch.sum(pi_dpo * (torch.log(pi_dpo + 1e-12) - torch.log(pi_ref + 1e-12)), dim=-1).item()

                    if kl_div > b:
                        sampling_logits = dpo_last_logits
                        chosen_dist = pi_dpo
                    else:
                        # Apply 'a' parameter for interpolation between models
                        sampling_logits = args.a * dpo_last_logits + (1 - args.a) * ref_last_logits
                        chosen_dist = F.softmax(sampling_logits, dim=-1)

                    pi_sample = F.softmax(sampling_logits, dim=-1)
                    next_token = torch.multinomial(pi_sample, num_samples=1).squeeze(-1)

                    generated_tokens.append(next_token.item())
                    chosen_token_prob = chosen_dist[0, next_token.item()]
                    ref_token_prob = pi_ref[0, next_token.item()]
                    kl_token = torch.log(chosen_token_prob + 1e-12) - torch.log(ref_token_prob + 1e-12)
                    kl_divs.append(kl_token)

                    next_token = next_token.unsqueeze(1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)], dim=1
                    )

                    if next_token[0].item() == tokenizer.eos_token_id:
                        break

                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                chat = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": output_text},
                ]
                test_texts = [
                    rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                    .replace(rm_tokenizer.bos_token, "")
                ]
                pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
                reward_score = pipe_outputs[0][0]["score"]
                question_scores.append(reward_score)

                average_kl = sum(kl_divs) if kl_divs else 0.0
                question_kl_scores.append(average_kl)

            average_question_score = sum(question_scores) / len(question_scores)
            average_question_kl = sum(question_kl_scores) / len(question_kl_scores)
            scores_per_question.append(average_question_score)
            kl_scores_per_question.append(average_question_kl)

        overall_average_score = sum(scores_per_question) / len(scores_per_question)
        overall_average_kl = sum(kl_scores_per_question) / len(kl_scores_per_question)

        # Save results for this threshold and dataset
        output_file_name = f"output_kl_threshold_control_b_{b}_a_{args.a}_dataset_{args.dataset}.json"
        with open(output_file_name, "w") as f:
            json.dump(
                {
                    "threshold": b,
                    "dataset": args.dataset,
                    "a_value": args.a,
                    "average_reward_score": overall_average_score,
                    "average_KL_divergence": overall_average_kl,
                },
                f,
                indent=4,
            )
        print(f"Saved results to {output_file_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold-based KL Control")
    parser.add_argument("--dpo_model_name", type=str, default="/scratch/gpfs/yw9355/model/models--RLHFlow--LLaMA3-iterative-DPO-final/snapshots/8c929ad1d79e0b23685c1cbb9316b53356f08852")
    parser.add_argument("--ref_model_name", type=str, default="/scratch/gpfs/yw9355/model/models--RLHFlow--LLaMA3-SFT/snapshots/1bedbf1899806bf585bd8d0833541d28af017dce")
    parser.add_argument("--reward_model_name", type=str, default="/scratch/gpfs/yw9355/model/models--sfairXC--FsfairX-LLaMA3-RM-v0.1/snapshots/94fad49f1b3227aa8b566f415a335adb68ec544c")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use: ultrafeedback or alpaca")
    parser.add_argument("--threshold_values", type=float, nargs="+", required=True, help="Threshold values")
    # Added a parameter for interpolation
    parser.add_argument("--a", type=float, default=0.5, help="Interpolation parameter between DPO and reference models")
    args = parser.parse_args()
    main(args)