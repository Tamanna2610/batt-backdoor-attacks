# Backdoor Attack Research – BATT (Transformation-Based Triggers)

A reproduction and extension of **"BATT: Backdoor Attack with Transformation-Based Triggers"**, evaluating deep learning model vulnerability to geometric backdoor attacks on CIFAR-10. Implements and benchmarks three attack strategies — BATT Rotation, BATT Translation, and BadNets baseline — against a clean ResNet-18 classifier.

> 📄 Full research paper available on ResearchGate — Reproducing and Extending BATT: Backdoor Attack with Transformation-Based Triggers

---

## Results

| Attack Type | Benign Accuracy | Attack Success Rate (ASR) |
|---|---|---|
| Clean Model (baseline) | 76.28% | — |
| BATT-R (Rotation, 16°) | 75.79% | **98.90%** |
| BATT-T (Translation, 6px) | 76.33% | **99.67%** |
| BadNets (patch baseline) | 76.16% | 97.26% |

**Key finding:** Geometric transformation triggers achieve >99% ASR with only a 5% poisoning rate, while maintaining benign accuracy within 0.5% of the clean model — making them highly stealthy and effective.

---

## Overview

Backdoor attacks poison a small fraction of training data by embedding hidden triggers. A compromised model behaves normally on clean inputs but produces attacker-controlled outputs when the trigger is present. This project:

1. **Reproduces** the original BATT attack pipeline using PyTorch on CIFAR-10
2. **Implements** rotation-based (BATT-R) and translation-based (BATT-T) transformation triggers
3. **Benchmarks** against the BadNets static patch baseline
4. **Analyzes** attack stealthiness, success rate, and implications for real-world ML security

---

## Project Structure

```
batt-backdoor-attacks/
│
├── dataset.py          # Dataset loading, preprocessing, and trigger injection pipeline
├── models.py           # ResNet-18 architecture definition
├── trigger.py          # Transformation-based trigger generation (rotation, translation, patch)
├── train.py            # Model training loop with backdoor poisoning (5% poison rate)
├── eval.py             # Benign accuracy (BA) and attack success rate (ASR) evaluation
├── main.py             # Entry point — orchestrates full attack pipeline
├── requirements.txt    # Dependencies
├── paper.pdf           # Full research paper
└── README.md
```

---

## Attack Strategies

### BATT-R — Rotation Attack
- Poisoned samples rotated by exactly **16 degrees** and relabeled to target class 1
- Trigger is invisible — rotation resembles natural image variation
- Bypasses pixel-level anomaly detectors

### BATT-T — Translation Attack
- Images shifted by **6 pixels** in a fixed direction
- Visually subtle and stealthy
- Achieved the highest ASR of **99.67%**

### BadNets — Patch Baseline
- Small white square patch stamped in the lower-right corner of 5% of training images
- Represents the classical backdoor trigger used in prior literature
- Easiest to detect visually — included as a comparison baseline

---

## Methodology

- **Dataset:** CIFAR-10 (60,000 images, 10 classes, 32×32 RGB)
- **Model:** ResNet-18 (random initialization, cross-entropy loss, Adam optimizer)
- **Poisoning Rate:** 5% of training data contains injected triggers
- **Target Class:** Label 1
- **Training:** 5 epochs, batch size 128
- **Evaluation:**
  - **Benign Accuracy (BA):** % of clean test images correctly classified
  - **Attack Success Rate (ASR):** % of triggered test images misclassified to target label

---

## Key Findings

- A **5% poisoning rate** is sufficient to achieve >98% ASR — traditional accuracy metrics alone cannot detect backdoors
- Geometric triggers (rotation, translation) are **more effective and stealthier** than static pixel patches
- Transformation-based triggers mimic natural image variations, making them resistant to visual inspection and preprocessing-based defenses
- Defensive methods must account for geometric transformations, not just pixel-level anomalies

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Requirements

```
torch
torchvision
numpy
matplotlib
scikit-learn
```

### Run

```bash
python main.py
```

Trains all four models (Clean, BATT-R, BATT-T, BadNets) sequentially and outputs BA and ASR for each.

---

## Skills Demonstrated

`Python` `PyTorch` `Deep Learning` `ResNet-18` `Computer Vision` `AI Safety` `Adversarial Machine Learning` `Backdoor Attacks` `Model Robustness` `CIFAR-10` `Research Reproduction` `Experimental Benchmarking` `Data Poisoning` `Neural Networks` `Modular ML Systems` `NumPy` `Matplotlib`

---

## References

- Xu et al., "BATT: Backdoor Attack with Transformation-Based Triggers," ICASSP 2022
- Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain," arXiv 2017

---

## Author

**Tamanna Subudhi**
B.S. Computer Science (AI Concentration) — Purdue University Northwest
[LinkedIn](https://linkedin.com/in/tamanna-subudhi-6792a026a)
