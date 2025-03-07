
# Shallow Preference Signals: Efficient Large Language Model Alignment with Truncated Data

This repository contains the code and models for the work presented in the paper "**Shallow Preference Signals: Efficient Large Language Model Alignment with Truncated Data**". In this work, we investigate the concept of shallow preference signals, showing that human preferences in language model outputs are often concentrated in the early part of the response rather than being evenly distributed. We explore how truncating responses at various points can significantly improve the efficiency of preference-based optimization methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

<figure>
  <img src="./figure1.pdf" alt="" />
  <figcaption>An example illustrating the phenomenon of shallow preference signals. It demonstrates how the relative quality of two responses can be determined from the early portion of the response, or even from the first sentence. Training with only the initial part allows the model to capture most of the preference signals while conserving resources.</figcaption>
</figure>

## Overview

In the domain of large language model alignment, preference-based optimization methods like RLHF and DPO rely heavily on human-annotated datasets. Our research reveals that the majority of the distinguishing signals between high-quality and low-quality responses are found in the early tokens of the response, a phenomenon we call **shallow preference signals**.

By focusing on this early portion of responses, we show that models can achieve comparable or even superior performance with truncated datasets, drastically reducing training and inference costs.

### Key Contributions:
1. **Shallow Preference Signals**: We demonstrate that preference signals are often concentrated in the early portion of responses.
2. **Truncation for Efficiency**: Models trained on truncated datasets (retaining only the first half or fewer tokens of each response) perform similarly or better than models trained on full responses.
3. **Efficient Training and Decoding**: We propose novel decoding strategies and optimized reward-KL divergence trade-offs that leverage shallow preference signals.

## Setup Instructions

Clone this repository to your local machine:

```bash
git clone https://github.com/your-repo/shallow-preference-signals.git
cd shallow-preference-signals
conda create -n shallow_pref python=3.9
conda activate shallow_pref
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## How to Reproduce Our Experiments

We have provided all the necessary scripts and configurations to reproduce the experiments discussed in our paper. The core process involves truncating preference datasets, training reward models, and fine-tuning models using DPO.

### Data Preprocessing
You can preprocess the preference datasets and create truncated versions using the provided script:

```bash
python src/preprocess.py --input_dir /path/to/input_dataset --output_dir /path/to/output_dataset --truncate_ratio 50
```

This will create a truncated dataset retaining 50% of the response tokens.

### Experiment: Truncation Effects on Reward Models and DPO

#### Training Reward Models and DPO models
For training the reward model and fine-tuning the model with DPO, we referred to [RLHFlow](<https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm>) and [OpenRLHF](<https://github.com/OpenRLHF/OpenRLHF>), using the default parameters from each.

#### Evaluate Models with RewardBench and Alpaca-Eval
For testing the model performance on rewardbench and alpaca-eval, we referred to [rewardbench](<https://github.com/allenai/reward-bench>) and [alpaca-eval](<https://github.com/tatsu-lab/alpaca_eval>), using the default evaluation sets. The model responses generated for alpaca-eval can be done as follows:

```bash
python src/get_model_response.py --model_path /path/to/model --model_name name_of_model_in_alpaca-eval
```

### Experiment: KL Divergence and Reward-KL Tradeoff for Evaluating Response Quality

#### KL Divergence Across Token Positions

##### Using default model paths
```bash
python src/experiment_6.1.py --dataset ultrafeedback --a 0.5
```

##### Specifying all parameters
```bash
python src/experiment_6.1.py --dpo_model_path /path/to/dpo_model --ref_model_path /path/to/ref_model --reward_model_path /path/to/reward_model --dataset alpaca --a 0.5 --num_samples 1
```

#### Reward-KL Tradeoff for Length Control and KL Threshold Control Decoding

##### Length Control

```bash
python src/experiment_6.2_length.py --dataset ultrafeedback --t_values 5 10 15
```

##### Threshold Control

```bash
python src/experiment_6.2_threshold.py --dataset ultrafeedback --threshold_values 0.1 0.5 1.0
```

# Full customization
python script.py --dpo_model_name /path/to/dpo --ref_model_name /path/to/ref --reward_model_name /path/to/reward \
                --dataset alpaca --threshold_values 0.2 0.4 0.6 0.8 \
                --mixed_mode --a 0.3 --num_samples 3 --max_questions 50 --max_tokens 256

<!-- ## Experiments and Results

The results of our experiments, such as the comparison between models trained on full and truncated datasets, are provided in the paper and corresponding evaluation files. These results show the performance across various truncation ratios and the effectiveness of our approach in both reward modeling and Direct Preference Optimization (DPO).

## Decoding Strategies

We also propose two novel decoding strategies: **Length Control Decoding** and **KL Threshold Control Decoding**, both of which prioritize the early portion of the response to maximize the reward-KL trade-off. These strategies can be enabled by setting the respective flags in the configuration. -->

