
# Exploring Shallow Preference Signals in Human Feedback: Insights into Efficient Large Language Model Alignment

This repository contains the code and models for the work presented in the paper "**Exploring Shallow Preference Signals in Human Feedback: Insights into Efficient Large Language Model Alignment**". In this work, we investigate the concept of shallow preference signals, showing that human preferences in language model outputs are often concentrated in the early part of the response rather than being evenly distributed. We explore how truncating responses at various points can significantly improve the efficiency of preference-based optimization methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

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

### Step 1: Data Preprocessing
You can preprocess the preference datasets and create truncated versions using the provided script:

```bash
python preprocess.py --input /path/to/dataset --output /path/to/output --truncate_percentage 50
```

This will create a truncated dataset retaining 50% of the response tokens.

### Step 2: Training Reward Models
To train reward models on truncated data, use the following command:

```bash
python train_reward_model.py --data /path/to/truncated/dataset --output /path/to/output
```

### Step 3: Fine-tuning with DPO
To fine-tune a model using DPO, execute:

```bash
python fine_tune_dpo.py --data /path/to/truncated/dataset --output /path/to/output
```

This will train the model using Direct Preference Optimization on the truncated data.

### Step 4: Evaluating the Model
To evaluate the performance of the trained model, use the following command:

```bash
python evaluate.py --model /path/to/trained/model --dataset /path/to/test/dataset
```

This will output the accuracy and other performance metrics.

## Experiments and Results

The results of our experiments, such as the comparison between models trained on full and truncated datasets, are provided in the paper and corresponding evaluation files. These results show the performance across various truncation ratios and the effectiveness of our approach in both reward modeling and Direct Preference Optimization (DPO).

## Decoding Strategies

We also propose two novel decoding strategies: **Length Control Decoding** and **KL Threshold Control Decoding**, both of which prioritize the early portion of the response to maximize the reward-KL trade-off. These strategies can be enabled by setting the respective flags in the configuration.

## Citation

<!-- If you find this work useful in your research, please cite our paper:

```
@article{yourpaper,
  title={Exploring Shallow Preference Signals in Human Feedback: Insights into Efficient Large Language Model Alignment},
  author={Author1, Author2, Author3},
  journal={arXiv},
  year={2024},
  url={https://arxiv.org/abs/xxxx.xxxxx}
} -->
```

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->
