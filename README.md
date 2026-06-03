# Sign Language MNIST — Perceptron and Neural Network Classification

End-to-end implementation and comparison of single-layer perceptron and multi-layer artificial neural network classifiers on the Sign Language MNIST dataset. Built from scratch in Python using NumPy, covering binary classification, multi-class classification, and a systematic comparison of activation functions and learning methods.

---

## Overview

The Sign Language MNIST dataset contains 28x28 grayscale images of hand signs representing 24 letters of the American Sign Language alphabet (J and Z are excluded as they require motion). With 27,455 training samples and 7,172 test samples, it presents a real classification challenge — especially for linear models.

This project asks two questions:

- How far can a single-layer perceptron get on this task?
- What does a multi-layer neural network add, and which design choices matter most?

---

## Dataset

**Source:** [Sign Language MNIST — Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

**Files used:** `sign_mnist_train.csv`, `sign_mnist_test.csv`

Each row contains a class label followed by 784 pixel values (flattened 28x28 image). Pixel values range from 0–255.

---

## Part 1: Perceptron

A single-layer perceptron implemented from scratch with support for:

- Step and sigmoid activation functions
- Online (stochastic) and full batch learning
- Multi-class classification via 24 one-vs-rest binary perceptrons

### Binary classification results (class C vs all)

| Activation | Learning | Accuracy | Precision | Recall |
|------------|----------|----------|-----------|--------|
| Step | Online | 0.991 | 0.974 | 0.839 |
| Step | Full Batch | 0.928 | 0.278 | 0.423 |
| Sigmoid | Online | 0.990 | 0.899 | 0.861 |
| Sigmoid | Full Batch | 0.950 | 0.395 | 0.297 |

**Step + online learning** gave the best precision (97.4%) — it rarely misclassifies a non-C image as C, though it misses roughly 16% of actual C samples.

**Full batch learning underperforms** across both activation functions. With only 20 iterations and 27,455 samples, batch learning makes just 20 weight updates total. The weights barely move and the model never really learns to distinguish class C.

### Multi-class results (24 perceptrons, one per class)

Overall accuracy: **33.9%**

Not surprising. A perceptron is a linear classifier — distinguishing 24 hand sign classes requires non-linear decision boundaries a single-layer model cannot learn. The sigmoid output from each perceptron was used as a confidence score for winner-takes-all prediction.

---

## Part 2: Artificial Neural Network

A multi-layer ANN implemented from scratch with:

- Configurable hidden layer architecture
- Sigmoid and ReLU activation functions
- Mini-batch gradient descent with backpropagation
- L2 weight regularisation, dropout, learning rate decay
- Gaussian input noise for data augmentation
- He initialisation (ReLU) and Xavier/Glorot initialisation (sigmoid)

### Binary classification: NN vs Perceptron

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Perceptron (sigmoid, online) | 0.990 | 0.899 | 0.861 |
| ANN 784→128→64→1 (sigmoid) | 0.992 | 0.938 | 0.871 |

The ANN's hidden layers allow it to learn non-linear boundaries. The main gain is in precision (+3.9 percentage points) — it is better at not flagging non-C samples as C. Training accuracy reached 100% by epoch 20 of 30.

### Multi-class: sigmoid vs ReLU

| Model | Architecture | Test Accuracy | Val Accuracy (held-out train) |
|-------|-------------|---------------|-------------------------------|
| Sigmoid ANN | 784→256→128→24 | 42.5% | — |
| ReLU ANN | 784→512→256→24 | ~45.0% | 99.7% |

**Note on test set distribution shift:** Class 23 is absent from the test set entirely. Classes 13 and 15 have large pixel intensity differences between train and test (mean differences of 28.7 and 35.1). A nearest-centroid baseline only achieves 24% on the official test set vs 42% on held-out training data. The ReLU network reaching 99.7% accuracy on held-out training data confirms the model learns correctly — the distribution mismatch limits official test performance.

### Why ReLU outperforms sigmoid

Sigmoid saturates at extreme values, pushing gradients toward zero and slowing learning in earlier layers (the vanishing gradient problem). ReLU has no saturation for positive inputs — its derivative is 1 for positive activations and 0 otherwise, so gradients pass through cleanly. ReLU is also cheaper to compute and produces sparse activations, which acts as a natural regulariser.

---

## Architecture and Hyperparameter Choices

| Parameter | Binary NN | Multi-class (sigmoid) | Multi-class (ReLU) |
|-----------|-----------|----------------------|-------------------|
| Architecture | 784→128→64→1 | 784→256→128→24 | 784→512→256→24 |
| Learning rate | 0.05 | 0.05 | 0.005 |
| LR decay | None | 0.998/epoch | 0.998/epoch |
| Activation | Sigmoid | Sigmoid | ReLU |
| Dropout | None | 0.4 | 0.5 |
| Weight decay (L2) | 0.0001 | 0.0005 | 0.0005 |
| Input noise (std) | None | 0.2 | 0.2 |
| Batch size | 64 | 128 | 128 |
| Epochs | 30 | 100 | 100 |
| Weight init | Xavier | Xavier | He |

Feature standardisation (zero mean, unit variance from training statistics) was applied to all inputs. This improved convergence speed significantly compared to simple 0–1 normalisation.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `numpy` | All matrix operations, weight updates, forward/backward pass |
| `matplotlib` | Weight visualisation, sample image display |

No deep learning frameworks used — the perceptron and ANN are both implemented from scratch.

---

## File Structure

```
sign-language-classification/
├── perceptron.ipynb    # Part 1: Perceptron implementation and analysis
├── neuralnet.ipynb     # Part 2: ANN implementation and analysis
├── perceptron.pdf      # Written report — Part 1
├── neuralnet.pdf       # Written report — Part 2
└── README.md           # This file
```

> The data files (`sign_mnist_train.csv`, `sign_mnist_test.csv`) are not included. Download them from the Kaggle link above.

---

## Key Findings

- Online learning massively outperforms batch learning when iterations are limited — 27,455 weight updates per epoch vs one
- A single-layer perceptron caps out at 33.9% on 24-class sign recognition — linear boundaries are not enough
- Hidden layers push binary classification precision from 89.9% to 93.8%
- ReLU consistently outperforms sigmoid for multi-class tasks due to cleaner gradient flow
- Distribution shift between train and test sets is the main constraint on test accuracy, not model capacity
