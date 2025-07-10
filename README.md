# 🤖 VLA_RL_Reasoning: AIRI 2025

[![Presentation](https://img.shields.io/badge/📊_Presentation-Google_Slides-orange)](https://docs.google.com/presentation/d/1gVk4IRcd6wwoRE-2RhvO0E1KE2ioVqNbLMZqfykhbbY/edit?usp=sharing)
[![Report](https://img.shields.io/badge/📄_Report-Overleaf-green)](https://www.overleaf.com/read/fvbnfvhqvfxb#345093)
[![Models](https://img.shields.io/badge/🤗_Models-HuggingFace-yellow)](https://huggingface.co/dancher00)

## 🎯 Description

The goal of this project is to , developed as part of AIRI 2025 research initiative.

## 🏆 Key Results

✅ **Closed issue in ManiSkill repository**  
📊 **Implementation**: Located in `/src` folder, accessible via `*.ipynb notebooks`  
🤗 **Models & Datasets**: Available on [HuggingFace](https://huggingface.co/dancher00)

## 🚀 Quick Start

```bash
# Create conda environment
conda create --name vlarlr python=3.10
conda activate vlarlr
```

```bash
# Clone the repository
git clone https://github.com/dancher00/VLA_RL_Reasoning
cd VLA_RL_Reasoning
```
...and then enjoy our notebooks!

# 🚀 Train 

## Install lerobot
https://github.com/huggingface/lerobot


## Run the train script
```bash
python3 lerobot/scripts/train.py \
  --output_dir=./outputs/train/train29 \
  --policy.path=lerobot/pi0 \
  --dataset.root=/home/user10_1/lerobot/src/dataset29 \
  --dataset.repo_id=dancher00/maniskill-panda-pickcube \
  --policy.repo_id=dancher00/pi0-panda-pickcube29 \
  --wandb.enable=true \
  --wandb.project=pi0_training29 \
  --optimizer.type=adamw \
  --optimizer.lr=2.5e-05 \
  --optimizer.weight_decay=1e-10 \
  --save_freq=100
```

## 📁 Project Structure

| 📂 File/Folder | 📝 Content |
|----------------|-------------|
| `data/` | 📊 Training and testing datasets |
| `src/` | 💻 Core source code and implementations |
| `tools/` | 🔧 Utilities for dataset conversion and preprocessing |

## 🔗 Resources

- 📊 **[Presentation](https://docs.google.com/presentation/d/1gVk4IRcd6wwoRE-2RhvO0E1KE2ioVqNbLMZqfykhbbY/edit?usp=sharing)** - Project overview and findings
- 📄 **[Technical Report](https://www.overleaf.com/read/fvbnfvhqvfxb#345093)** - Detailed methodology and results
- 🤗 **[Models & Datasets](https://huggingface.co/dancher00)** - Pre-trained models and datasets

## 👥 Contributors

- 👨‍💻 **Vakhitov Rodion**
- 👨‍💻 **Belov Danil** 
- 👨‍💻 **Ivanov Leonid**
- 👨‍💻 **Kachaev Nikita**

---

<div align="center">

**🎉 AIRI 2025 Research Project 🎉**

*Advancing the frontiers of AGI*

</div>
