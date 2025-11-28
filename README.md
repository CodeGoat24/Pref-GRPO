<div align="center">
    <h1 align="center"> Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning
    </h1>
    
[UnifiedReward](https://github.com/CodeGoat24/UnifiedReward) Team


<a href="https://arxiv.org/pdf/2508.20751">
<img src='https://img.shields.io/badge/arXiv-Pref GRPO-blue' alt='Paper PDF'></a>

<a href="https://codegoat24.github.io/UnifiedReward/Pref-GRPO">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>

<a href="https://huggingface.co/CodeGoat24/FLUX.1-dev-PrefGRPO">
<img src='https://img.shields.io/badge/Huggingface-Model-yellow' alt='Project Page'></a>

<a href="https://github.com/CodeGoat24/UniGenBench">
<img src='https://img.shields.io/badge/Benchmark-UniGenBench-green' alt='Project Page'></a>



[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20UniGenBench%20-Leaderboard_(English)-brown)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20UniGenBench%20-Leaderboard_(Chinese)-red)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20UniGenBench%20-Leaderboard_(English%20Long)-orange)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_English_Long)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20UniGenBench%20-Leaderboard_(Chinese%20Long)-pink)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese_Long)
</div>

## 🔥 News
Please leave us a star ⭐ if you find this work helpful.


- [2025/11] 🔥🔥 We release **Qwen-Image** and **FLUX.1-dev (LoRA)** training code for both Pref-GRPO and UnifiedReward.

- [2025/11] 🔥🔥 **Nano Banana Pro**, **FLUX.2-dev** and **Z-Image** are added to all 🏅Leaderboard.

- [2025/10] 🔥 **Alibaba Group** proves the effectiveness of Pref-GRPO on aligning LLMs in [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/pdf/2510.15514). Thanks to all contributors!
- [2025/9] 🔥 **Seedream-4.0**, **GPT-4o**, **Imagen-4-Ultra**, **Nano Banana**, **Lumina-DiMOO**, **OneCAT**, **Echo-4o**, **OmniGen2**, and **Infinity** are added to all 🏅Leaderboard.
- [2025/8] 🔥 We release 🏅[Leaderboard(**English**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard), 🏅[Leaderboard (**English Long**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_English_Long), 🏅[Leaderboard (**Chinese Long**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese_Long) and 🏅[Leaderboard(**Chinese**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese).



![pref_grpo_pipeline](/assets/pref_grpo_pipeline.png)



![pref_grpo_pipeline](/assets/pref_grpo_reward_hacking.png)



## 🔧 Environment Set Up
1. Clone this repository and navigate to the folder:
```bash
git clone https://github.com/CodeGoat24/UnifiedReward.git
cd UnifiedReward/Pref-GRPO
```

2. Install the training package:
```bash
conda create -n PrefGRPO python=3.12
conda activate PrefGRPO

bash env_setup.sh fastvideo

git clone https://github.com/mlfoundations/open_clip
cd open_clip
pip install -e .
cd ..

mkdir images
```

3. Download Models
```bash
huggingface-cli download CodeGoat24/UnifiedReward-2.0-qwen3vl-8b
huggingface-cli download CodeGoat24/UnifiedReward-Think-qwen-7b

wget https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/resolve/main/open_clip_pytorch_model.bin
```
## 💻 Training

#### 1. Deploy vLLM server

1. Install vLLM
```bash
pip install vllm>=0.11.0

pip install qwen-vl-utils==0.0.14
```
2. Start server
```bash
bash vllm_utils/vllm_server_UnifiedReward_Think.sh  
```
#### 2. Preprocess training Data 
we use training prompts in [UniGenBench](https://github.com/CodeGoat24/UniGenBench), as shown in ```"./data/unigenbench_train_data.txt"```.

```bash
# FLUX.1-dev
bash fastvideo/data_preprocess/preprocess_flux_rl_embeddings.sh

# Qwen-Image
pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0

bash fastvideo/data_preprocess/preprocess_qwen_image_rl_embeddings.sh
```


#### 3. Train
```bash
# FLUX.1-dev
## UnifiedReward-Think for Pref-GRPO
bash scripts/finetune_prefgrpo_flux.sh
bash scripts/finetune_prefgrpo_flux_lora.sh

## UnifiedReward for Point Score-based GRPO
bash scripts/finetune_unifiedreward_flux.sh
bash scripts/finetune_unifiedreward_flux_lora.sh

# Qwen-Image
## UnifiedReward-Think for Pref-GRPO
bash scripts/finetune_prefgrpo_qwenimage_grpo.sh

## UnifiedReward for Point Score-based GRPO
bash scripts/finetune_unifiedreward_qwenimage_grpo.sh

```

### 🚀 Inference and Evaluation
we use test prompts in [UniGenBench](https://github.com/CodeGoat24/UniGenBench), as shown in ```"./data/unigenbench_test_data.csv"```.
```bash
# FLUX.1-dev
bash inference/flux_dist_infer.sh

# Qwen-Image
bash inference/qwen_image_dist_infer.sh
```

Then, evaluate the outputs following [UniGenBench](https://github.com/CodeGoat24/UniGenBench).


## 📧 Contact
If you have any comments or questions, please open a new issue or feel free to contact [Yibin Wang](https://codegoat24.github.io).


## 🤗 Acknowledgments
Our training code is based on [DanceGRPO](https://github.com/XueZeyue/DanceGRPO), [Flow-GRPO](https://github.com/yifan123/flow_grpo), and [FastVideo](https://github.com/hao-ai-lab/FastVideo).

We also use [UniGenBench](https://github.com/CodeGoat24/UniGenBench) for T2I model semantic consistency evaluation.

Thanks to all the contributors!


## ⭐ Citation
```bibtex
@article{Pref-GRPO&UniGenBench,
  title={Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning},
  author={Wang, Yibin and Li, Zhimin and Zang, Yuhang and Zhou, Yujie and Bu, Jiazi and Wang, Chunyu and Lu, Qinglin and Jin, Cheng and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2508.20751},
  year={2025}
}
```
