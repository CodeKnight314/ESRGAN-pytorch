# ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) Pytorch Implementation

## Overview

This repository contains a simplistic implementation of ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) for super-resolving images. The code includes training and inference scripts to generate high-resolution images from low-resolution inputs using a pre-trained model.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Dataset Preparation](#dataset-preparation)
- [Results](#results)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ESRGAN.git
    cd ESRGAN
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv esrgan-env
    source esrgan-env/bin/activate
    ```

3. cd to project directory: 
    ```bash 
    cd ESRGAN-pytorch/
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

Before training the model, you can use `dataset/prepare_dataset.sh` to download, extract, and organize the dataset.

1. cd to the dataset directory address with below: 
    ```bash
    cd dataset
    ```

2. Make the script executable:

    ```bash
    chmod +x prepare_dataset.sh
    ```

3. Run the script:

    ```bash
    prepare_dataset.sh
    ```
    
## Usage

### Inference

Use the `inference.py` script to perform super-resolution on images in a specified directory.

**Arguments:**
- `--root_dir`: Directory containing input images.
- `--output_dir`: Directory to save super-resolved images.
- `--resize`: Optional flag to resize super-resolved images to the original image size.

**Example:**
```bash
python inference.py --root_dir ./data/input --output_dir ./data/output
```

You can optionally use the `--resize` flag if you want the super-resolved image to be resized to original image size.

```bash
python inference.py --root_dir ./data/input --output_dir ./data/output --resize 
```

### Training
Use the `train.py` script to train the ESRGAN model. It includes pretraining of the generator and full training with the discriminator.

**Arguments:**
- `--root` Root directory for images (requires train and val split).
- `--save` Directory to save weights and log files.
- `--epochs`: Total epochs for training (default: 30).
- `--warmup`: Total epochs for the pretraining phase.
- `--upscale`: Upscale factor to train ESRGAN on (default: 4).

**Example:**
```bash
python train.py --root ./data --save ./checkpoints --epochs 30 --warmup 5 --upscale 4
```

## Results

Below are some example results:

<p align="center">
  <img src="resources/gif/combined.gif" alt="Switching between Low-Resolution and Super-Resolved Image">
</p>
