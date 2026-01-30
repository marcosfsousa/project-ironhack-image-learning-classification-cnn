# Project #2 – Ironhack

## Convolutional Neural Networks (CNNs) with CIFAR-10

---

## Overview

This project explores *Convolutional Neural Networks (CNNs)* using the *CIFAR-10* dataset. The goal was to build and improve image classification models step by step, starting from a simple baseline CNN and gradually improving performance through architectural changes and training tweaks.

The focus is not just accuracy, but understanding why certain changes help or hurt the model.

---

## Dataset

*CIFAR-10* contains:

- 60,000 color images (32×32 pixels)
- 10 classes
- 50,000 training images
- 10,000 test images

Each image belongs to exactly one class.

---

## What We Did

- Loaded and normalized the data
- Built a baseline CNN using *TensorFlow / Keras*
- Kept experiments clean and easy to extend
- We kept tweaking the model step by step to boost accuracy
- Tested on unseen data
-  tracked every change and its impact

---

## Structure

```text
project-root/
├── presentation/        # slides and visuals
├── main.ipynb           # main experimentation notebook
├── config.py            # configuration and constants
├── data.py              # data loading & preprocessing
├── model.py             # CNN architectures
├── train.py             # training logic & callbacks
├── metrics.py           # evaluation and plots
└── README.md
```

The project is organized this way to keep experiments clean and easy to extend.


---

## Team

- Edwin Santiago
- Marcos Sousa
- Rishi

---

## Deliverables

We delivered a trained model, evaluation metrics, an experiment notebook, and a group presentation
Also included a Excel file where we tracked different preprocessing, architecture decisions for each model.

---

## Methodology

We started with a baseline CNN, iteratively improved the architecture, tuned hyperparameters, and tested performance.

---

## Results
  
Final accuracy, training curves, and confusion matrices are reported in `main.ipynb`.

**Model Performance**
- Baseline scratch CNN test accuracy: ~58–60%
- Best scratch CNN after architectural tuning: ~77%
- Transfer learning with VGG16: ~87% test accuracy

**Observations**
- Transfer learning provides a significant performance gain over scratch models
- Results are consistent with known CIFAR-10 benchmarks given 32×32 image resolution
- Performance plateaus align with expected dataset difficulty limits

**Error Analysis**
- Animal classes (cat, dog, deer) are harder to distinguish due to texture similarity
- Vehicle classes (truck, automobile, ship) achieve higher accuracy due to strong shape cues
- Misclassifications follow semantic similarity rather than random noise

