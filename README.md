# CNN Image Classification in PyTorch (Fashion-MNIST & CIFAR-100)

This project implements **convolutional neural networks (CNNs)** in **PyTorch** to classify images from the **Fashion-MNIST** and **CIFAR-100** datasets.  
The work explores model design, training dynamics, overfitting behavior, and performance gaps between simple CNNs and deeper architectures such as ResNets.

---

## Project Overview

Developed as part of **CS/SE 4AL3 – Applied Machine Learning (Fall 2025)** at McMaster University, this project investigates:

- CNN training on two datasets of varying difficulty
- Behavior of training vs validation loss curves
- Overfitting and underfitting patterns
- Comparison with deep architectures such as ResNet
- Effects of model depth, width, optimization, and dataset complexity

---

## Repository Structure

```
PYTORCH-CNN-IMAGE-CLASSIFICATION/
│
├── Assignment4_starter.ipynb     # Completed assignment notebook
│
├── data/                         # Automatically downloaded datasets
│   └── FashionMNIST / CIFAR100
│
├── README.md                     # Project documentation
└── .gitignore
```

---

## Model Overview

### Part 1 — Fashion-MNIST CNN

A shallow CNN for 28×28 grayscale images:

- **Conv1:** 1 → 10 filters (3×3)
- **Conv2:** 10 → 5 filters (3×3)
- **Conv3:** 5 → 16 filters (3×3)
- **Pooling:** MaxPool(2×2)
- **FC Layer:** 16·11·11 → 10 classes

Achieves ~80% test accuracy.

### Part 2 — CIFAR-100 CNN

A larger CNN to handle 32×32 RGB images:

- **Conv1:** 3 → 32 filters
- **Conv2:** 32 → 64 filters
- **Conv3:** 64 → 128 filters
- **Pooling:** MaxPool(2×2)
- **FC Layer:** 128·13·13 → 100 classes

Typical accuracy ~20–35% due to dataset complexity.

---

## Key Concepts

- CNN architecture design
- Data normalization and augmentation
- Train/validation/test splits
- Loss & accuracy tracking
- Overfitting analysis
- PyTorch training loops
- Evaluation on unseen test data
- Comparison with deeper architectures (e.g., ResNet-18)

---

## Running the Notebook

### 1. Create Environment (Optional)

```bash
conda create -n 4al3-cnn python=3.12
conda activate 4al3-cnn
pip install torch torchvision matplotlib numpy
```

### 2. Start Jupyter Notebook

```bash
jupyter notebook Assignment4_starter.ipynb
```

### 3. Ensure the following:

- All cells run top-to-bottom
- Training/validation plots are visible
- Test accuracy printed for both datasets
- AI-usage disclaimer included (required by course policy)

---

## Experimental Sections

### **Part 1 — Fashion-MNIST**

- Implement small CNN
- Train for 15 epochs
- Plot training vs validation loss
- Report test accuracy
- Discuss dataset difficulty & convergence behavior

### **Part 2 — CIFAR-100**

- Build deeper CNN for RGB images
- Train for 15 epochs
- Plot training vs validation loss
- Report test accuracy
- Compare performance to ResNet baseline (~80%)
- Discuss architectural limitations

---

## Example Outputs

| Dataset       | Plot                   | Description                                       |
| ------------- | ---------------------- | ------------------------------------------------- |
| Fashion-MNIST | fashion_loss_curve.png | Smooth convergence with minimal overfitting       |
| CIFAR-100     | cifar_loss_curve.png   | Validation divergence and overfitting illustrated |

---

## Tools & Libraries

- **Python 3.12**
- **PyTorch** (CNNs, optimizers, loss functions)
- **Torchvision** (datasets, transforms)
- **NumPy / Matplotlib**
- **Jupyter Notebook / VS Code**

---

## Academic Integrity Notice

This project was completed individually for **CS/SE 4AL3**.  
All AI-assisted content (code, explanations, or text) is disclosed in the AI-Use section of the notebook in compliance with course policy.

---

## Author

**S. Pathmanathan**  
5th Year Software Engineering Student, McMaster University  
Former Datacenter GPU Validation Engineer @ AMD  
Interests: Deep Learning, Model Optimization, Software Systems
