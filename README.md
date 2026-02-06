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

## News
Please leave us a star if you find this work helpful.


- [2026/02] We release **FLUX.2-Klein (T2I/I2I)**, **Qwen-Image-Edit** and **Wan2.2** training code!!
- [2026/02] We release [UnifiedReward-Flex](https://codegoat24.github.io/UnifiedReward/flex)-based Pref-GRPO for both image and video generation!!
- [2026/01] **Tongyi Lab** improves Pref-GRPO on open-ended agents in [ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking](https://arxiv.org/pdf/2601.06487). Thanks to all contributors!
<details>
<summary><strong>More News</strong></summary>

- [2025/11] We release **Qwen-Image**, **Wan2.1** and **FLUX.1-dev** Full/LoRA training code.
- [2025/11] **Nano Banana Pro**, **FLUX.2-dev** and **Z-Image** are added to all Leaderboards.
- [2025/10] **Alibaba Group** proves the effectiveness of Pref-GRPO on aligning LLMs in [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/pdf/2510.15514). Thanks to all contributors!
- [2025/9] **Seedream-4.0**, **GPT-4o**, **Imagen-4-Ultra**, **Nano Banana**, **Lumina-DiMOO**, **OneCAT**, **Echo-4o**, **OmniGen2**, and **Infinity** are added to all Leaderboards.
- [2025/8] We release [Leaderboard (**English**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard), [Leaderboard (**English Long**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_English_Long), [Leaderboard (**Chinese Long**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese_Long) and [Leaderboard (**Chinese**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese).
</details>



![pref_grpo_pipeline](/assets/pref_grpo_pipeline.png)



![pref_grpo_pipeline](/assets/pref_grpo_reward_hacking.png)



## Environment Setup
1. Clone this repository and navigate to the folder:
```bash
git clone https://github.com/CodeGoat24/Pref-GRPO.git
cd Pref-GRPO
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

```

3. Install vLLM (for UnifiedReward-based rewards)
```bash
conda create -n vllm
conda activate vllm
pip install "vllm>=0.11.0"
pip install qwen-vl-utils==0.0.14
```

4. Download Models
```bash
huggingface-cli download CodeGoat24/UnifiedReward-2.0-qwen3vl-8b
huggingface-cli download CodeGoat24/UnifiedReward-Think-qwen3vl-8b
huggingface-cli download CodeGoat24/UnifiedReward-Flex-qwen3vl-8b
huggingface-cli download CodeGoat24/UnifiedReward-Edit-qwen3vl-8b
```


## Training
#### 1. Model-specific workflows (click to expand)
We use training prompts in [UniGenBench](https://github.com/CodeGoat24/UniGenBench), as shown in ```"./data/unigenbench_train_data.txt"```.

<details>
<summary><strong>FLUX.1-dev</strong></summary>

##### Preprocess training Data
```bash
bash fastvideo/data_preprocess/preprocess_flux_rl_embeddings.sh
```

##### Train (examples)
```bash
# Pref-GRPO
## UnifiedReward-Flex
bash scripts/full_train/ur_flex_prefgrpo_flux.sh
## UnifiedReward-Think
bash scripts/full_train/ur_think_prefgrpo_flux.sh


# UnifiedReward for Point Score-based GRPO
bash scripts/full_train/unifiedreward_flux.sh
```
</details>

<details>
<summary><strong>FLUX.2-Klein(T2I,I2I)</strong></summary>

##### Preprocess training Data (T2I)
```bash
bash fastvideo/data_preprocess/preprocess_flux2_klein_rl_embeddings.sh
```

##### Preprocess training Data (I2I)
For FLUX.2-Klein edit GRPO, prepare a jsonl dataset first.

1. Put jsonl files under `data/{Image_Edit_Dataset_Name}/`.
2. Each line must be a JSON object. Recommended fields:
- `instruction`: edit instruction
- `instruction_cn`: optional Chinese instruction (used when `USE_CN=1`)
- `source_image` or `image`: source image path (required)
- `target_image`: optional target/reference edited image path (for eval/checking only, not mandatory for rollout reward)
3. Instruction fallback order used by code:
- `instruction` -> `prompt` -> `caption` -> `text`
4. Path rules:
- Absolute path: used directly
- Relative path: resolved against dataset root (`input_path` dir)
- If not found, code also tries `<dataset_root>/images/<relative_path>`

Minimal jsonl example:
```json
{"instruction":"replace the red car with a blue one","source_image":"images/0001_source.png","target_image":"images/0001_target.png"}
{"instruction_cn":"把天空改成晚霞","source_image":"images/0002_source.jpg"}
```

Run preprocess:
```bash
# default: INPUT_PATH=data/Image_Edit_data, OUTPUT_DIR=data/flux2_klein_edit_embeddings
bash fastvideo/data_preprocess/preprocess_flux2_klein_edit.sh

# optional: use Chinese instruction when available
USE_CN=1 bash fastvideo/data_preprocess/preprocess_flux2_klein_edit.sh
```

##### Train (examples)
```bash
# Pref-GRPO (UnifiedReward-Flex as example)
bash scripts/lora/lora_ur_flex_prefgrpo_flux2_klein.sh

# Edit GRPO (UnifiedReward-Edit pointwise/prefgrpo reward example)
bash scripts/lora/lora_ur_edit_point_flux2_klein_edit.sh
bash scripts/lora/lora_ur_edit_prefgrpo_flux2_klein_edit.sh

```
</details>

<details>
<summary><strong>Qwen-Image</strong></summary>

##### Preprocess training Data
```bash
pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0

bash fastvideo/data_preprocess/preprocess_qwen_image_rl_embeddings.sh
```


##### Train (examples)
```bash
## UnifiedReward-Think for Pref-GRPO
bash scripts/full_train/ur_think_prefgrpo_qwenimage.sh

## UnifiedReward for Point Score-based GRPO
bash scripts/full_train/unifiedreward_qwenimage.sh
```
</details>

<details>
<summary><strong>Qwen-Image-Edit</strong></summary>

##### Preprocess training Data (edit embeddings)
Prepare jsonl files first (default input: `data/Image_Edit_data`).

Each line is a JSON object. Recommended fields:
- `instruction` (or fallback keys: `prompt` / `caption` / `text`)
- `instruction_cn` (optional, used when `USE_CN=1`)
- `source_image` or `image` (required)
- `target_image` (optional)

Minimal jsonl example:
```json
{"instruction":"replace the red car with a blue one","source_image":"images/0001_source.png","target_image":"images/0001_target.png"}
{"instruction_cn":"把天空改成晚霞","source_image":"images/0002_source.jpg"}
```

Run preprocess:
```bash
# default output: data/qwen_image_edit_embeddings
bash fastvideo/data_preprocess/preprocess_qwen_image_edit.sh

# optional: use Chinese instruction when available
USE_CN=1 bash fastvideo/data_preprocess/preprocess_qwen_image_edit.sh
```

##### Train (examples)
```bash
# start UnifiedReward-Edit server first
bash vllm_utils/vllm_server_UnifiedReward_Edit.sh

# Pref-GRPO with edit pairwise reward
bash scripts/full_train/ur_edit_prefgrpo_qwen_image_edit.sh
```
</details>

<details>
<summary><strong>Wan2.1</strong></summary>

##### Preprocess training Data
```bash
bash fastvideo/data_preprocess/preprocess_wan21_rl_embeddings.sh
```

##### Train (examples)
```bash
# Pref-GRPO
## UnifiedReward-Flex
bash scripts/lora/lora_ur_flex_prefgrpo_wan21.sh

## UnifiedReward-Think
bash scripts/lora/lora_ur_think_prefgrpo_wan21.sh
```
</details>

<details>
<summary><strong>Wan2.2</strong></summary>

##### Preprocess training Data
```bash
bash fastvideo/data_preprocess/preprocess_wan22_rl_embeddings.sh
```

##### Train (examples)
```bash
# Pref-GRPO
## UnifiedReward-Flex
bash scripts/lora/lora_ur_flex_prefgrpo_wan22.sh
```
</details>



### Reward Models & Usage
We support multiple reward models via the dispatcher in `fastvideo/rewards/dispatcher.py`.
Reward model checkpoint paths are configured in `fastvideo/rewards/reward_paths.py`.
Supported reward models (click to expand for setup details):



<details>
<summary><strong><span style="font-size:1.05em">aesthetic</span></strong></summary>

<strong>Set in `fastvideo/rewards/reward_paths.py`</strong><br>
`aesthetic_ckpt`: path to the Aesthetic MLP checkpoint (`assets/sac+logos+ava1-l14-linearMSE.pth`)<br>
`aesthetic_clip`: HuggingFace CLIP model id (`openai/clip-vit-large-patch14`)
  </details>

<details>
<summary><strong><span style="font-size:1.05em">clip</span></strong></summary>

<strong>Download weights</strong>
```bash
wget https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/resolve/main/open_clip_pytorch_model.bin
```

<strong>Set in `fastvideo/rewards/reward_paths.py`</strong><br>
`clip_pretrained`: path to OpenCLIP weights (used by CLIP reward)
  </details>

<details>
<summary><strong><span style="font-size:1.05em">hpsv2</span></strong></summary>

<strong>Set in `fastvideo/rewards/reward_paths.py`</strong><br>
`hpsv2_ckpt`: path to [HPS_v2.1_compressed.pt](https://huggingface.co/xswu/HPSv2/tree/main)<br>
`clip_pretrained`: path to OpenCLIP weights (required by HPSv2)
  </details>

<details>
<summary><strong><span style="font-size:1.05em">hpsv3</span></strong></summary>

<strong>Set in `fastvideo/rewards/reward_paths.py`</strong><br>
`hpsv3_ckpt`: path to [HPSv3](https://huggingface.co/MizzenAI/HPSv3) checkpoint
  </details>

<details>
<summary><strong><span style="font-size:1.05em">pickscore</span></strong></summary>

<strong>Set in `fastvideo/rewards/reward_paths.py`</strong><br>
`pickscore_processor`: HuggingFace processor id ([CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K))<br>
`pickscore_model`: HuggingFace model id ([Pickscore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1))
  </details>


<details>
<summary><strong><span style="font-size:1.05em">unifiedreward (alignment / style / coherence)</span></strong></summary>

<strong>Start server</strong><br>
Targets: `unifiedreward_alignment`, `unifiedreward_style`, `unifiedreward_coherence`
```bash
bash vllm_utils/vllm_server_UnifiedReward.sh  
```
</details>

<details>
<summary><strong><span style="font-size:1.05em">unifiedreward_think</span></strong></summary>

<strong>Start server</strong><br>
Target: `unifiedreward_think`
```bash
bash vllm_utils/vllm_server_UnifiedReward_Think.sh  
```
</details>

<details>
<summary><strong><span style="font-size:1.05em">unifiedreward_flex</span></strong></summary>

<strong>Start server</strong><br>
Target: `unifiedreward_flex`
```bash
bash vllm_utils/vllm_server_UnifiedReward_Flex.sh  
```
</details>

<details>
<summary><strong><span style="font-size:1.05em">unifiedreward_edit</span></strong></summary>

<strong>Start server (UnifiedReward-Edit)</strong><br>
Targets:
- `unifiedreward_edit_pairwise`
- `unifiedreward_edit_pointwise_image_quality`
- `unifiedreward_edit_pointwise_instruction_following`
```bash
bash vllm_utils/vllm_server_UnifiedReward_Edit.sh
```

<strong>Scope</strong><br>
Edit rewards are image-only (`modality=image`) and expect edit-specific inputs:
- pairwise: source image + two edited candidates + instruction
- pointwise image quality: edited image only
- pointwise instruction following: source image + edited image + instruction

<strong>Optional weighting via env vars</strong><br>
For `unifiedreward_edit_pointwise_image_quality`:
- `EDIT_QUALITY_WEIGHT_NATURALNESS` (default `1.0`)
- `EDIT_QUALITY_WEIGHT_ARTIFACTS` (default `1.0`)

For `unifiedreward_edit_pointwise_instruction_following`:
- `EDIT_IF_WEIGHT_SUCCESS` (default `1.0`)
- `EDIT_IF_WEIGHT_OVEREDIT` (default `1.0`)
</details>

<details>
<summary><strong><span style="font-size:1.05em">videoalign</span></strong></summary>

<strong>Set in `fastvideo/rewards/reward_paths.py`</strong><br>
`videoalign_ckpt`: path to [VideoAlign](https://huggingface.co/KlingTeam/VideoReward) checkpoint directory
  </details>

#### Set rewards in your training/eval scripts
Use `--reward_spec` to choose which rewards to compute and (optionally) their weights.

Examples:
```bash
# Use a list of rewards (all weights = 1.0)
--reward_spec "unifiedreward_think,clip,hpsv3"

# Weighted mix
--reward_spec "unifiedreward_alignment:0.5,unifiedreward_style:1.0,unifiedreward_coherence:0.5"

# Edit reward examples
--reward_spec '{"unifiedreward_edit_pointwise_image_quality":0.5,"unifiedreward_edit_pointwise_instruction_following":0.5}'
--reward_spec '{"unifiedreward_edit_pairwise":1.0}'

# JSON formats are also supported
--reward_spec '{"clip":0.5,"aesthetic":1.0,"hpsv2":0.5}'
--reward_spec '["clip","aesthetic","hpsv2"]'
```


### Inference and Evaluation
we use test prompts in [UniGenBench](https://github.com/CodeGoat24/UniGenBench), as shown in ```"./data/unigenbench_test_data.csv"```.

<details>
<summary><strong>FLUX.1-dev</strong></summary>

```bash
bash inference/flux_dist_infer.sh
```
</details>

<details>
<summary><strong>Qwen-Image</strong></summary>

```bash
bash inference/qwen_image_dist_infer.sh
```
</details>

<details>
<summary><strong>FLUX.2-Klein</strong></summary>

```bash
bash inference/flux2_klein_dist_infer.sh
```
</details>

<details>
<summary><strong>Wan2.1</strong></summary>

```bash
bash inference/wan21_dist_infer.sh
bash inference/wan21_eval_vbench.sh
```
</details>

<details>
<summary><strong>Wan2.2</strong></summary>

```bash
bash inference/wan22_dist_infer.sh
bash inference/wan22_eval_vbench.sh
```
</details>

Then, evaluate the outputs following [UniGenBench](https://github.com/CodeGoat24/UniGenBench).

### Reward-based Image Scoring (UniGenBench)
We provide a script to score a folder of generated images on UniGenBench using supported reward models.

```bash
GPU_NUM=8 bash tools/eval_quality.sh
```

Edit `tools/eval_quality.sh` to set:
- `--image_dir`: path to your UniGenBench generated images
- `--prompt_csv`: prompt file (default: `data/unigenbench_test_data.csv`)
- `--reward_spec`: the reward models (and weights) to use
- `--api_url`: UnifiedReward server endpoint (if using UnifiedReward-based rewards)
- `--output_json`: output file for scores


## Contact
If you have any comments or questions, please open a new issue or feel free to contact [Yibin Wang](https://codegoat24.github.io).


## Acknowledgments
Our training code is based on [DanceGRPO](https://github.com/XueZeyue/DanceGRPO), [Flow-GRPO](https://github.com/yifan123/flow_grpo), and [FastVideo](https://github.com/hao-ai-lab/FastVideo).

We also use [UniGenBench](https://github.com/CodeGoat24/UniGenBench) for T2I model semantic consistency evaluation.

Thanks to all the contributors!


## Citation
```bibtex
@article{Pref-GRPO&UniGenBench,
  title={Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning},
  author={Wang, Yibin and Li, Zhimin and Zang, Yuhang and Zhou, Yujie and Bu, Jiazi and Wang, Chunyu and Lu, Qinglin and Jin, Cheng and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2508.20751},
  year={2025}
}
```
