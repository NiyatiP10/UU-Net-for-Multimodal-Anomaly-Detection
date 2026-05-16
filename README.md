# UU-Net for Multimodal Anomaly Detection

## Overview

This project implements a modified version of the **UU-Net architecture** proposed in the IEEE research paper:

> **Unsupervised Multimodal Anomaly Detection With Missing Sources for Liquid Rocket Engine**

The implementation adapts the architecture for the C-MAPSS Dataset to perform robust anomaly detection under missing sensor conditions.

The project combines:

* Gramian Angular Field (GAF) preprocessing
* Multimodal sensor fusion
* Intramodality & intermodality reconstruction
* Skip-connected Autoencoders
* Missing sensor simulation
* Deep unsupervised anomaly detection

---

# Project Highlights

* Converts multivariate time-series sensor data into 2D GAF images
* Simulates missing sensor scenarios during training
* Implements UU-Net two-stage fusion architecture
* Learns robust latent representations for anomaly detection
* Uses reconstruction losses and latent discrepancy for scoring
* Visualizes degradation trends and anomaly distributions
* Supports predictive maintenance research for industrial systems

---

# Architecture

## UU-Net Pipeline

1. Raw sensor preprocessing
2. Sliding window generation
3. Gramian Angular Field conversion
4. Per-sensor encoder learning
5. Intramodality latent fusion
6. Missing source reconstruction
7. Intermodality Skip-AE fusion
8. Anomaly score computation

---

# Dataset

This project uses the NASA C-MAPSS turbofan engine dataset.

Dataset source:
[NASA C-MAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data?utm_source=chatgpt.com)

## Dataset Details

* Multivariate turbofan engine sensor data
* 21 sensor channels + operational settings
* Time-series degradation trajectories
* Used widely in predictive maintenance research

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* pyts
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Gramian Angular Field (GAF)

The project converts 1D sensor signals into 2D image representations using Gramian Angular Fields.

### Advantages of GAF

* Preserves temporal correlations
* Enables CNN-based feature extraction
* Encodes degradation patterns spatially
* Improves representation learning for anomaly detection

---

# UU-Net Architecture

The model consists of two major stages:

## 1. Intramodality Fusion

* Individual encoders for each sensor
* Missing sensor reconstruction
* Latent-space feature aggregation
* Source-level fusion

## 2. Intermodality Fusion

* Skip-connected autoencoder
* Cross-modal feature learning
* Joint reconstruction
* Robust anomaly representation

---

# Anomaly Score

The anomaly score is computed using:

```math
S = R1 + R2 - \lambda D_z
```

Where:

* `R1` → First reconstruction loss
* `R2` → Second reconstruction loss
* `Dz` → Latent-space discrepancy

Higher scores indicate anomalous behavior.

---

# Installation

Clone the repository:

```bash
git clone <https://github.com/NiyatiP10/UU-Net-for-Multimodal-Anomaly-Detection>
```

Move into the project directory:

```bash
cd UU-Net-for-Multimodal-Anomaly-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/UU_Net_for_Multimodal_Anomaly_Detection.ipynb
```

---

# Results

The model successfully:

* Detects anomalous engine behavior
* Handles missing sensor modalities
* Learns robust latent representations
* Preserves temporal information using GAF encoding
* Improves reliability under incomplete data conditions

## Evaluation Metrics

* ROC-AUC
* Reconstruction Loss
* Fault Detection Rate (FDR)
* False Alarm Rate (FAR)
* Latent Representation Discrepancy

---

# Visualizations

The project includes:

* Sensor degradation trends
* GAF transformation plots
* UU-Net architecture diagrams
* Latent-space visualization
* Anomaly score distributions
* ROC curves

---

# Reference Paper

Feng et al.,

**Unsupervised Multimodal Anomaly Detection With Missing Sources for Liquid Rocket Engine**

Published in:
IEEE Transactions on Neural Networks and Learning Systems, 2022.

---

# Future Improvements

* Real-time anomaly detection pipeline
* Streamlit/Flask deployment
* Attention-based sensor fusion
* Explainable AI visualizations
* Training on additional C-MAPSS subsets
* Hyperparameter optimization
* Online monitoring dashboard

---

# Important Note

This repository contains an educational and research-oriented implementation inspired by the original UU-Net paper.
The implementation is adapted for the NASA C-MAPSS dataset and simplified for experimentation and learning purposes.

---

# Author

NIYATI PATIL
(NiyatiP10)

---
