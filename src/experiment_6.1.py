import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.nn.functional as F
import json
import tqdm
import argparse

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models and tokenizers
def load_models(dpo_model_path, ref_model_path, reward_model_path):
    dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_path).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)
    
    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model_path,
        device=0,
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    
    return dpo_model, ref_model, tokenizer, rm_pipe, rm_tokenizer

# Load instructions from JSON
def load_instructions_from_json(file_name):
    with open(file_name, 'r') as f:
        instructions = json.load(f)
    return instructions

# Process and generate text
def process_and_generate(dpo_model, ref_model, tokenizer, rm_pipe, rm_tokenizer, a, instructions, num_samples):
    results = []

    for question in instructions[0:100]:
        kl_divergences = []
        rewards = []
        for sample_idx in tqdm.tqdm(range(num_samples), desc=f"Processing question: {question}", unit="sample"):
            messages = [{"role": "user", "content": question}]
            model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

            if isinstance(model_inputs, torch.Tensor):
                input_ids = model_inputs
                attention_mask = torch.ones_like(input_ids, device=device)
            else:
                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs["attention_mask"]

            generated_tokens = []
            log_prob_new = 0.0
            log_prob_ref = 0.0

            for _index in tqdm.tqdm(range(512), desc="Processing Tokens", unit="token"):
                with torch.no_grad():
                    dpo_outputs = dpo_model(input_ids=input_ids, attention_mask=attention_mask)
                    dpo_logits = dpo_outputs.logits
                    ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                    ref_logits = ref_outputs.logits

                dpo_last_logits = dpo_logits[:, -1, :]
                ref_last_logits = ref_logits[:, -1, :]
                mixed_logits = a * (dpo_last_logits - ref_last_logits) + ref_last_logits
                probabilities = F.softmax(mixed_logits, dim=-1)

                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
                log_prob_token_new = torch.log(probabilities[0, next_token])
                log_prob_new += log_prob_token_new.item()

                prob_ref = F.softmax(ref_last_logits, dim=-1)
                log_prob_token_ref = torch.log(prob_ref[0, next_token])
                log_prob_ref += log_prob_token_ref.item()

                generated_tokens.append(next_token.item())
                next_token = next_token.unsqueeze(1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)], dim=1)

                if next_token[0].item() == tokenizer.eos_token_id:
                    break

            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            kl_div = log_prob_new - log_prob_ref
            kl_divergences.append(kl_div)

            chat = [{"role": "user", "content": question}, {"role": "assistant", "content": output_text}]
            test_texts = [rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
            pipe_outputs = rm_pipe(test_texts, return_all_scores=True, function_to_apply="none", batch_size=1)
            reward = pipe_outputs[0][0]["score"]
            rewards.append(reward)

        results.append({
            "question": question,
            "kl_divergences": kl_divergences,
            "rewards": rewards,
        })

    return results

# Main program
def main():
    parser = argparse.ArgumentParser(description="Process model results for different a values and datasets")
    
    # Model paths with defaults
    parser.add_argument("--dpo_model_path", type=str, 
                        default="/scratch/gpfs/yw9355/model/models--RLHFlow--LLaMA3-iterative-DPO-final/snapshots/8c929ad1d79e0b23685c1cbb9316b53356f08852",
                        help="Path to DPO model")
    parser.add_argument("--ref_model_path", type=str, 
                        default="/scratch/gpfs/yw9355/model/models--RLHFlow--LLaMA3-SFT/snapshots/1bedbf1899806bf585bd8d0833541d28af017dce",
                        help="Path to reference model")
    parser.add_argument("--reward_model_path", type=str, 
                        default="/scratch/gpfs/yw9355/model/models--sfairXC--FsfairX-LLaMA3-RM-v0.1/snapshots/94fad49f1b3227aa8b566f415a335adb68ec544c",
                        help="Path to reward model")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, required=True, choices=["ultrafeedback", "alpaca"], help="Dataset to use")
    
    # KL divergence parameter
    parser.add_argument("--a", type=float, required=True, help="Value of 'a' for KL computation")
    
    # Number of samples per question
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per question")
    
    args = parser.parse_args()
    
    # Select dataset
    if args.dataset == "ultrafeedback":
        instructions = load_instructions_from_json('instructions_ultrafeedback.json')
    else:
        instructions = load_instructions_from_json('instructions_alpaca.json')
    
    # Load models
    dpo_model, ref_model, tokenizer, rm_pipe, rm_tokenizer = load_models(
        args.dpo_model_path, args.ref_model_path, args.reward_model_path
    )
    
    # Process data and generate results
    results = process_and_generate(
        dpo_model, ref_model, tokenizer, rm_pipe, rm_tokenizer, 
        args.a, instructions, args.num_samples
    )
    
    # Save results to file
    output_filename = f"a_{args.a}_data_{args.dataset}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_filename}")

if __name__ == "__main__":
    main()