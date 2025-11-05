
# Distillation

This repository is an experimental study of Knowledge Distillation following the paper "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015): https://arxiv.org/pdf/1503.02531

The goal is to reproduce key ideas from the paper (teacher / student networks, training/evaluation pipelines) and later implement the full distillation loss and temperature-based soft targets (work in progress).

## Highlights

- Notebook: `distillation.ipynb` contains the main experiments: training teacher and student MLPs on MNIST and CNNs on CIFAR-100.
- Example models (pretrained) are stored under `model/`.
- Uses PyTorch and torchvision for models and datasets.

## Paper / Motivation

This project studies Knowledge Distillation as presented in Hinton et al. (2015). Distillation transfers the generalization ability of a large, well-trained teacher network to a smaller student network by training the student on the teacher's softened class probabilities (with a temperature parameter) in addition to the hard labels.

Paper: Distilling the Knowledge in a Neural Network — https://arxiv.org/pdf/1503.02531

## Project contract (quick)

- Inputs: datasets in `data/` (MNIST, CIFAR-100). Pretrained weights in `model/`.
- Outputs: trained model state dicts saved to `model/`, training logs and simple plots produced by the notebook.
- Expected errors: missing dataset files (the notebook downloads CIFAR/MNIST if needed); missing GPU results in CPU training.

## Requirements

- Python >= 3.12 (see `pyproject.toml`)
- torch, torchvision, matplotlib

This project uses the UV package manager in the notebook environment. If you use the same environment, install dependencies with UV as shown below.

### Install (using UV)

In the notebook the author used the `uv` command to add packages. Example:

```bash
uv add torch torchvision matplotlib
uv add jupyterlab  # optional, if you want to use JupyterLab
```

To run Python tools / commands in the UV-managed environment use `uv run`:

```bash
uv run python main.py
uv run jupyter lab  # start notebook server
```

If you don't use UV, you can also use pip:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision matplotlib jupyterlab
```

## Quick usage

- Run the small example script:

```bash
uv run python main.py
# or with plain python if you used pip:
python main.py
```

- Launch the main notebook to reproduce experiments and plots:

```bash
uv run jupyter lab  # then open distillation.ipynb
```

## Notebook notes

- `distillation.ipynb` sets up datasets and dataloaders for MNIST and CIFAR-100, defines training & evaluation loops, and implements simple Teacher/Student architectures (MLP for MNIST, small CNN for CIFAR-100).
- The notebook currently trains teacher and student models independently (supervised learning). The distillation-specific training (soft targets + temperature) is not yet implemented — planned as next work.
- The notebook attempts to detect an accelerator via `torch.accelerator` and will fall back to CPU when not available.

## Data & models

- Data folder: `data/` — contains MNIST, CIFAR-10 and CIFAR-100 downloads (the notebook downloads CIFAR-100 and MNIST automatically if missing).
- Models: `model/` — contains pre-saved teacher and student checkpoints. Example paths used in the notebook:
	- `model/CIFAR100/teacher_model_CIFAR100_epoch_X.pth`
	- `model/CIFAR100/student_model_CIFAR100_epoch_X.pth`

If you have pre-trained weights in `model/`, the notebook will load them instead of re-training.

## Next steps (planned)

1. Implement the distillation loss: combine cross-entropy on hard labels with KL divergence between teacher and student soft targets at temperature T.
2. Add command-line scripts / training harness to run distillation experiments outside the notebook (configurable temperature, alpha, epochs, datasets).
3. Add unit tests and small reproducible examples for CI.

## References

- Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.

## Contact / author

Repository owner: Alex

If you'd like, I can also:
- add a small CLI to run the notebook experiments with parameters,
- implement the distillation loss and an example training run in a new script.

---
Generated from the repository's `distillation.ipynb` on 2025-11-05.

