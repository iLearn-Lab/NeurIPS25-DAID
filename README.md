# Fair Deepfake Detectors Can Generalize

> Uncovers a causal link between fairness and generalizability in deepfake detection, and proposes DAID to jointly improve both via confounding factor control.

## Authors

**Harry Cheng**<sup>1</sup>, **Ming-Hui Liu**<sup>2</sup>, **Yangyang Guo**<sup>1</sup>, **Tianyi Wang**<sup>1</sup>, **Liqiang Nie**<sup>3</sup>, **Mohan Kankanhalli**<sup>1</sup>\*

<sup>1</sup> National University of Singapore
<sup>2</sup> Shandong University
<sup>3</sup> Harbin Institute of Technology (Shenzhen)
\* Corresponding author

## Links

- **Paper**: [Fair Deepfake Detectors Can Generalize](https://openreview.net/forum?id=p27bSdc3FS)
- **arXiv**: [2507.02645](https://arxiv.org/abs/2507.02645)
- **Checkpoint**: [Google Drive](https://drive.google.com/file/d/1obeawJUIBc0brvjUAThygnC469QOQp8c/view?usp=sharing) / [BaiduYunDisk](https://pan.baidu.com/s/1aY2G2fJt_ED55qhwWS8nLg?pwd=qxvj) (code: `qxvj`)
- **Code Repository**: [GitHub](https://github.com/iLearn-Lab/NeurIPS25-DAID)

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [TODO](#todo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [04/2026] Transfer repos to iLearn-Lab
- [12/2025] Poster presented at NeurIPS 2025, San Diego (Exhibit Hall C,D,E #1203)

---

## Introduction

This repository is the official implementation of **Fair Deepfake Detectors Can Generalize**, accepted at **NeurIPS 2025**.

Existing deepfake detectors often suffer from two seemingly conflicting objectives: generalization to unseen forgeries and fairness across demographic groups. This work, for the first time, establishes a causal relationship showing these objectives are not inherently at odds — controlling for the right confounders enables both simultaneously.

We construct a Causal Graph that identifies two primary confounders: **DD (Data Distribution)** and **MC (Model Capacity)**. Our method **DAID** (Demographic Attribute-insensitive Intervention Detection) addresses these with:
- **Demographic-aware data rebalancing** to control DD
- **Demographic-agnostic feature aggregation** to control MC

This repository provides training/testing code built on the [CADDM](https://github.com/megvii-research/CADDM) backbone, along with pretrained checkpoints.

> **Note**: The code requires CADDM data preprocessing (landmark extraction, source image labeling). If CADDM is unavailable, see [DAID_AI_Face](https://github.com/xaCheng1996/DAID_AI_Face) (EfficientNet backbone, 2nd place at NeurIPS 2025 AIFace Detection Challenge).

---

## Highlights

- First to causally link **fairness** and **generalizability** in deepfake detection
- Proposes **DAID**: plug-and-play framework combining data rebalancing + feature alignment
- Achieves superior performance on fairness and generalization across three cross-domain benchmarks

---

## Method / Framework

### Causal Graph

![Causal Graph](./CF.png)

**Figure 1.** Causal Graph identifying DD (Data Distribution) and MC (Model Capacity) as confounders affecting both fairness and generalizability.

### Pipeline

![Pipeline](./pipeline.png)

**Figure 2.** Overall pipeline of DAID.

---

## Project Structure

```text
.
├── backbones/          # Backbone network definitions
├── configs/            # Training and testing configuration files
├── detection_layers/   # Detection head modules
├── lib/                # Utility libraries
├── CF.png              # Causal graph figure
├── pipeline.png        # Pipeline figure
├── model.py            # Model definition
├── dataset.py          # Dataset loader
├── train.py            # Training script
├── test.py             # Testing script
├── test.slurm          # SLURM job script for cluster testing
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iLearn-Lab/NeurIPS25-DAID.git
cd NeurIPS25-DAID
```

### 2. Set up CADDM dependencies

Follow the [CADDM](https://github.com/megvii-research/CADDM) setup for data preprocessing (landmark extraction, source image labeling).

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Checkpoints / Models

Download the pretrained checkpoint and update the path in `./configs/caddm_test.cfg`:

- **Google Drive**: [Download](https://drive.google.com/file/d/1obeawJUIBc0brvjUAThygnC469QOQp8c/view?usp=sharing)
- **BaiduYunDisk**: [Download](https://pan.baidu.com/s/1aY2G2fJt_ED55qhwWS8nLg?pwd=qxvj) (code: `qxvj`)

---

## Dataset / Benchmark

Our model supports any deepfake dataset. To compute fairness metrics, the dataset needs demographic labels. We use datasets from [Fairness-Generalization](https://github.com/Purdue-M2/Fairness-Generalization).

---

## Usage

### Testing

```bash
python test.py --cfg ./configs/caddm_test.cfg
```

### Training

```bash
python train.py --cfg ./configs/caddm_train.cfg
```

All required settings are in the respective `.cfg` files.

---

## TODO

- [ ] Release full data preprocessing scripts
- [ ] Add visualization / demo scripts

---

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{cheng2025fair,
  title     = {Fair Deepfake Detectors Can Generalize},
  author    = {Cheng, Harry and Liu, Ming-Hui and Guo, Yangyang and Wang, Tianyi and Nie, Liqiang and Kankanhalli, Mohan},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

---

## Acknowledgement

- This codebase is built on [CADDM](https://github.com/megvii-research/CADDM). We thank the Megvii team for their excellent work.
- We use datasets from [Fairness-Generalization](https://github.com/Purdue-M2/Fairness-Generalization). We thank the Purdue M2 Lab for sharing the benchmark.

---

## License

This project is released under the Apache License 2.0.
