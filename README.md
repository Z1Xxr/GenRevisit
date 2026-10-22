<h1 align="center"><img src="assets/wordmark.svg" alt="GenRevisit" width="360" /></h1>

<p align="center">
  <b><em>Look Back to Move Forward: Learning to Revisit for Agentic Compositional Image Generation</em></b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.16224"><img src="https://img.shields.io/badge/Paper-arXiv%3A2610.16224-B31B1B?style=flat-square" alt="Paper on arXiv" width="143" /></a>
  <img src="assets/project-page-badge.svg" alt="Project page coming soon" width="180" />
  <a href="docs/results.md"><img src="assets/benchmark-results.svg" alt="Benchmark Results" width="130" /></a>
  <a href="https://huggingface.co/GenRevisit"><img src="assets/huggingface-badge.svg" alt="GenRevisit organization on Hugging Face" width="162" /></a>
  <a href="https://modelscope.cn/organization/GenRevisit"><img src="assets/modelscope-badge.svg" alt="GenRevisit organization on ModelScope" width="157" /></a>
</p>

<p align="center">
  <a href="https://github.com/Flipped-zx/GenRevisit"><img src="assets/star-us.svg" alt="⭐ Please give us a star if you find this helpful!" width="360" /></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#training">Training</a>
</p>

---

## 💡 Overview

**GenRevisit** is an agentic framework for compositional text-to-image generation. Existing agentic approaches refine images iteratively but typically fix the *continuation state* — the image from which refinement proceeds — to the latest candidate. Since iterative editing is not necessarily monotonic, the agent may continue refining from an image that has already degraded, wasting its budget restoring constraints that were previously satisfied.

GenRevisit extends action selection to **joint continuation-state and action selection**: at each decision step, a multimodal planner not only adjusts its editing instruction based on the interaction history, but also selects *which* historical image candidate to continue from — jointly deciding **where to continue from** and **how to refine**.

The planner is trained in two stages:

1. **Supervised fine-tuning (SFT)** on multi-round teacher trajectories, supervising only actions that improve the image (including recovery from degraded states), while failed actions remain in the context as recovery cues.
2. **Reinforcement learning (RL)** with GRPO, augmented by **Normalized Action Credit Assignment (NACA)**, which redistributes positive trajectory-level advantage among generation and editing actions according to each action's normalized contribution to closing the remaining utility gap — instead of assigning the same advantage to every action in the trajectory.

## Framework

[![GenRevisit framework: joint continuation-state and action selection](assets/framework.png)](assets/framework.pdf)

<details>
<summary><b>Method details</b></summary>

The framework consists of three components:

- **History-conditioned interaction.** The decision context includes the prompt's atomic constraints (each paired with a VQA-style verification question), all historical image candidates with their constraint-level feedback, retrieved guidance, and the remaining budget. At each step the agent emits one structured action: `generate`, `edit`, `query_skill` (retrieve guidance), or `submit`. For `edit`, the action jointly specifies a source image index from the interaction history (the continuation state) and an editing instruction. Submission returns the historical GM-best image.
- **Supervised fine-tuning for cold start.** Teacher trajectories are collected with GPT-5.5; only improving actions receive imitation loss (masked SFT), while degraded actions and their feedback remain in context to inform later decisions.
- **RL with Normalized Action Credit Assignment (NACA).** The terminal reward measures how well the final submitted image satisfies the task constraints. For positive-advantage trajectories, NACA redistributes credit among image actions in proportion to the fraction of the remaining utility gap each action closes, so regressive actions in otherwise successful trajectories are not spuriously reinforced. Non-positive advantages are left unchanged.

</details>

## Results

GenRevisit achieves state-of-the-art compositional generation performance among the evaluated baselines:

- **GenEval2:** 84.3 GM with Qwen-Image-2512 and 74.4 GM with Z-Image-Turbo (without additional training), surpassing the strongest agentic baselines using the same generators by **12.7** and **10.9** points respectively.
- **Cross-benchmark generalization:** best overall scores on **GenEval** (0.9844) and **GenEval++** (0.8929).
- **Ablations:** SFT + NACA (84.3 GM) outperforms SFT-only (81.5) and SFT + standard GRPO (80.3) on GenEval2; joint continuation-state selection outperforms latest-only and best-only teacher continuations by 5.0 and 2.8 GM points under matched budgets.

## Installation

Create the Python 3.11 environment from the repository root:

```bash
conda env create -f environment.yml
conda activate genrevisit
```

For RL, install the additional dependencies:

```bash
python -m pip install -e ".[rl]"
```

SFT requires LLaMA-Factory 0.9.5 with PyTorch, DeepSpeed and FlashAttention 2 installed in the training environment.

## Model Setup

### Planner

Use **GenRevisit-RL** for inference, or **Qwen3-VL-8B-Instruct** for training from the base model.

```bash
hf download GenRevisit/GenRevisit-RL --local-dir models/GenRevisit-RL
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
```

### Image Tools

**Generation:** Qwen-Image-2512 or Z-Image-Turbo. **Editing:** Qwen-Image-Edit-2511.

```bash
hf download Qwen/Qwen-Image-2512 --local-dir models/Qwen-Image-2512
hf download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511
hf download Tongyi-MAI/Z-Image-Turbo --local-dir models/Z-Image-Turbo
```

## Inference

### Agent Interface

Validate an input packet and construct planner messages using the included example:

```python
import json
from pathlib import Path
from genrevisit.agent.contract import render_messages, validate_input

packet = json.loads(Path("tests/fixtures/agent/empty_input.json").read_text())
validate_input(packet)
messages = render_messages(packet, image_refs=[])
```

For packets containing image candidates, supply image references in candidate order. See [Agent](docs/agent.md) for the input format and action schemas.

## Evaluation

Evaluate submitted images through a shared benchmark entry point. For GenEval2:

```bash
python scripts/evaluate.py --benchmark geneval2 score \
  --manifest data/evaluation/submissions.json \
  --benchmark-file data/evaluation/geneval2.jsonl \
  --scorer-repo external/GenEval2 \
  --model-path models/Qwen3-VL-8B-Instruct \
  --output-root outputs/evaluation/geneval2

python scripts/evaluate.py --benchmark geneval2 report \
  --output-root outputs/evaluation/geneval2
```

See [Evaluation](evaluation/README.md) for evaluator setup, input format and scoring details.

## Training

### SFT

SFT uses **LLaMA-Factory** with DeepSpeed ZeRO-3. Register `sft_train` and `sft_validation` in your dataset directory's `dataset_info.json`, then launch training with your local paths:

```bash
python scripts/train_sft.py \
  --model-dir models/Qwen3-VL-8B-Instruct \
  --dataset-dir data/sft \
  --output-dir outputs/sft
```

Use `--train-dataset` and `--eval-dataset` to override the dataset names, or `--dry-run` to inspect the resolved configuration. Training settings are in [full_sft.yaml](configs/sft/full_sft.yaml).

### RL / NACA

RL uses **PyTorch** with GRPO and **Normalized Action Credit Assignment (NACA)**. Integrate the following components into your training loop for prepared on-policy batches:

| Component | Entry point |
| :--- | :--- |
| Action credit allocation | [`assign_action_advantages`](src/genrevisit/train/naca.py) |
| GRPO loss | [`grpo_loss`](src/genrevisit/train/rl_loss.py) |
| Policy update | [`optimizer_step`](src/genrevisit/train/rl_step.py) |

See [naca_grpo.yaml](configs/rl/naca_grpo.yaml) for configuration and [test_training.py](tests/test_training.py) for an optimizer-step example.

<details>
<summary><b>Repository structure</b></summary>

```text
GenRevisit/
├── assets/                  Project artwork
├── configs/
│   ├── sft/                 SFT and DeepSpeed configuration
│   └── rl/                  NACA and GRPO configuration
├── docs/                    Agent interface documentation
├── environment.yml          Core Python environment
├── evaluation/
│   ├── geneval2/             GenEval2
│   ├── geneval/              GenEval
│   └── genevalpp/            GenEval++
├── scripts/                 Training and evaluation entry points
├── src/genrevisit/
│   ├── agent/               Field contracts, schemas and message rendering
│   └── train/               Credit allocation, loss and optimizer step
└── tests/                   Agent and training checks
```

</details>

## Citation

If you find GenRevisit useful in your research, please cite:

```bibtex
@article{wang2026genrevisit,
  title   = {Look Back to Move Forward: Learning to Revisit for Agentic Compositional Image Generation},
  author  = {Wang, Zixu and Zhao, Xiaoran and Liu, Yahui and Wang, Feng-ao and Xiao, Junhao and Shu, You and Liu, Bing and Liu, Yiting and Jian, Songlei and Tian, Zhiliang and Li, Dongsheng},
  journal = {arXiv preprint arXiv:2610.16224},
  year    = {2026}
}
```
