# Self-Pruning Neural Network 

This repository implements a "self-pruning" Multi-Layer Perceptron (MLP) for CIFAR-10. It uses a custom layer (`PrunableLinear`) that automatically learns to drop unnecessary weight connections dynamically during training using an $L_1$ penalty on continuous sigmoid gate scores.

##  Quick Start

**1. Install Requirements**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Run the Training Sweep**
This command will train models under different sparsity constraints, generate evaluation tables, and plot performance histograms in the `outputs/` folder:
```bash
python -m src.experiments.run_lambda_sweep --config configs/default.yaml
```

**3. Run the Inference Server (API)**
Once an optimal checkpoint is found, host the model to run predictions dynamically:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
*You can then view the API documentation and test image uploads at `http://localhost:8000/docs`.*

## 📂 Repository Structure

* `configs/` - YAML parameter configurations (batch sizes, epochs, lambda sweeps).
* `src/model/` - The core PyTorch architectures, including the custom `PrunableLinear` logic.
* `src/experiments/` - The automated runner tracking hyperparameters over CIFAR-10.
* `src/api/` - The standalone FastAPI inference application.
* `tests/` - Verifies gradient flow shapes and matrix masking logic natively.

---
*Created for the Tredence AI Engineer Case Study.*

