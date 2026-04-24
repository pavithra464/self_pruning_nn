# The Self-Pruning Neural Network ✂️🧠

**Tredence AI Engineer Case Study Submission**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)](https://fastapi.tiangolo.com/)

A production-ready PyTorch repository demonstrating self-pruning neural networks using differentiable gating mechanisms. This project trains a Multi-Layer Perceptron (MLP) on CIFAR-10, dynamically pruning dead weights via $L_1$ regularization of continuous sigmoid gate scores during training.

---

## 📖 Problem Statement
The objective of this case study is to design a neural network architecture capable of "self-pruning" during training. Rather than applying traditional post-training magnitude pruning, the network must natively learn to sparify its parameters simultaneously alongside the foundational classification task. This demonstrates an understanding of continuous differentiability, custom layer mechanics, and balancing auxiliary loss functions in deep learning.

## 💡 Approach
Instead of utilizing standard `torch.nn.Linear`, this project implements a custom `PrunableLinear` layer. Each layer holds standard weights $\mathbf{W}$ and a same-shaped tensor of learnable gate scores $\mathbf{G}$. 

$$ y = (\mathbf{W} \odot \sigma(\mathbf{G}))x + b $$

The sigmoid function $\sigma(\cdot)$ smoothly bounds the gate values to $[0, 1]$. To encourage active sparsification, an $L_1$ penalty on the sum of $\sigma(\mathbf{G})$ across all layers is added to the standard Cross-Entropy classification loss. 

Because the sigmoid is continuously differentiable, gradients flow cleanly backward through both $\mathbf{W}$ (learning the representation itself) and $\mathbf{G}$ (learning what representations to forget). This eliminates the reliance on non-differentiable binary masks and heuristic Straight-Through Estimators (STEs).

## 🏗 Architecture Overview

The Multi-Layer Perceptron consists of standard sequential modules wrapping our custom linear core:

```text
Input (3x32x32) 
      │ 
   Flatten
      │
┌─────▼───────────────┐  x N Hidden Blocks
│   PrunableLinear    │  (W ⊙ σ(G))
│   BatchNorm1d       │
│   ReLU              │
│   Dropout           │
└─────┬───────────────┘
      │
  PrunableLinear ───────► Logits (10 Classes)
```

## 🧪 The Lambda ($\lambda$) Trade-Off

The total optimization objective during training is defined universally as:
```text
Total Loss = Classification Loss (CrossEntropy) + λ * Sparsity Loss (L1)
```

The scale multiplier $\lambda$ dictates the evolutionary pressure applied to network density:
- **Low $\lambda$ (~0.0):** The model ignores the $L_1$ penalty. Gates initialized highly ($\approx 1.0$) undergo minimal restriction. Test accuracy is uniquely emphasized as the sole objective while sparsity predictably stays near 0%.
- **Medium $\lambda$ (e.g., 1e-4 - 1e-3):** The optimal regime algorithmically. The network identifies redundant mapping pathways and pushes their corresponding gate scores harshly negative making $\sigma(\mathbf{G}) \approx 0$. Sparsity scales dramatically, shedding footprint capacity while accuracy remains remarkably reliable.
- **High $\lambda$ (e.g., 0.05+):** Catastrophic gradient collapse. Absolute regularization overpowers structural representation. The network systematically forces virtually all gates to $0.0$, driving architectural sparsity to $100\%$ resulting directly in random guessing capability ($\approx 10\%$ for CIFAR-10).

Sweeping $\lambda$ structurally maps the exact Pareto frontier boundary identifying efficient capacity limits against generalized accuracy drops. 

## 📊 Evaluation Metrics
* **Accuracy (%)**: Cross-Entropy Top-1 evaluation on the test dataset serving as verification for unaltered representational strength.
* **Sparsity Level (%)**: Defined strictly as the fraction of global network weights whose corresponding activated limits rest fully pruned (e.g. $\sigma(G) < 0.01$).

An optimized model structure maps mathematically to a bifurcated gate distribution. Rendering histogram analytics dynamically exposes: a massive weight spike at $0.0$ (totally pruned sub-modules) separated completely from a dense concentration at $1.0$ (critical network paths). High structural diffusion indicates weak scaling confidence.

## 🚀 How to Run

### 1. Setup the Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Lambda Sweep Experiment
Execute the automated sequence runner assessing varying configured $\lambda$ thresholds consecutively:
```bash
python -m src.experiments.run_lambda_sweep --config configs/default.yaml
```
**Expected Outputs (Saved into `outputs/sweep_YYYYMMDD_.../`):**
- Systematically tagged checkpoints `checkpoint_lambda_XXXX.pth`.
- Automatically compiled parameter table reports (`sweep_results.csv` and Markdown table variants).
- PNG formatted comparative histogram visualization (`best_model_gate_dist.png`) exploring structural densities dynamically.

### 3. Execution (API Logic)
Engage in lightweight FastAPI services allowing remote execution behaviors natively:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Review exposed functional endpoints by directly visiting OpenAPI Swagger UI at `http://localhost:8000/docs`.

If isolated file tests are preferred, point standard test execution against a saved snapshot payload directly via:
```bash
python -m src.inference --image sample.png --ckpt outputs/dir/checkpoint_lambda_0.001.pth
```

## 📂 File Structure

```text
self_pruning_nn/
├── README.md               # You're here!
├── REPORT.md               # Case-study evaluation template analyzing actual mathematical targets
├── CHECKLIST.md            # Final QA assurance indicators
├── requirements.txt        # Isolated dependency specifications
├── configs/
│   └── default.yaml        # Immutable parameters driving lambda ranges cleanly
├── src/
│   ├── api/
│   │   └── app.py          # FastAPI structural REST wrapper
│   ├── experiments/
│   │   └── run_lambda_sweep.py # Automated execution hub 
│   ├── model/
│   │   ├── prunable_linear.py  # Self-pruning continuous matrix operations
│   │   └── prunable_mlp.py     # Deep PyTorch standard integration
│   ├── utils/              # Quality-of-life trackers monitoring limits, OS states, metrics
│   ├── inference.py        # Abstracted pipeline independent from sequential trainer epochs
│   └── train.py/evaluate.py# Granular logic tracking iterators and 1-batch test diagnostics
└── tests/                  # Pytest verification suites probing custom gating parameters
```

## 🚧 Limitations & 💫 Future Improvements
1. **Computational Dimension Unstructured Density (Structured Pruning)**: While internal parameter footprints are dynamically nullified via floating mask zeros effectively destroying complexity conceptually, real-world Tensor $W$ dimensions remain fixed resulting directly in identical FLOP matrices mathematically. The next core implementation must slice isolated masks permanently resolving graph compression logic via explicit structural sub-graph reconstructions.
2. **Flexible Extensibility (CNN Modules)**: Current scaling isolates MLP limitations explicitly. Transforming `PrunableLinear` logic towards `PrunableConv2d` provides robust dimensional analysis applicable rapidly over deep Computer Vision matrices without structural complications.
3. **Optimized Penalties**: Introducing advanced continuous penalty schedules (`Lambda` schedules) that exponentially trigger weight penalties explicitly after foundational layer stability is successfully localized. Wait, restrict mapping footprints only after foundational concepts are mapped cleanly, boosting representation bounds.
