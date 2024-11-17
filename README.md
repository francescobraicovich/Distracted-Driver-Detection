# Distracted Driving Detection with Hyperdimensional Computing and ResNet
![image](https://github.com/user-attachments/assets/a9968ab9-406e-4eba-ac0e-75065fefea5e)


This repository contains code for detecting distracted driving using a hybrid approach combining Hyperdimensional Computing (HDC) and ResNet deep learning.

## Overview

The system uses a novel hybrid architecture that combines deep learning feature extraction with hyperdimensional computing classification to achieve robust distracted driving detection.

## How It Works

### 1. Feature Extraction with ResNet
- Uses a modified ResNet-18 architecture with custom bottleneck blocks
- Takes grayscale driver images as input (1 channel)
- Final layer outputs high-dimensional feature vectors
- Custom bottleneck blocks help reduce dimensionality while preserving important features

### 2. Hyperdimensional Computing Layer
The HDC component transforms the ResNet features into hyperdimensional space through:

1. **Level Encoding**:
   - Quantizes continuous features into discrete levels
   - Maps each level to a random hypervector
   
2. **Position Encoding**:
   - Generates position-dependent hypervectors
   - Uses permutation for preserving spatial relationships
   
3. **Feature Binding**:
   - Combines level and position information through binding operations
   - Creates composite representations preserving both what and where

4. **Class Prototypes**:
   - Aggregates encoded samples into class prototypes
   - Uses majority voting and similarity thresholds for robustness

## Results

The hybrid approach shows several advantages:

- **Accuracy**: Achieves >97% classification accuracy on the test set
- **Robustness**: Less sensitive to noise and variations compared to pure deep learning
- **Interpretability**: HDC layer provides insights into feature importance
- **Efficiency**: Reduced training time through HDC's lightweight operations

Performance metrics on the test dataset:
- Precision: 0.988
- Recall: 0.988
- F1-Score: 0.988

## Files

- `distracted_driving.ipynb`: Main notebook with complete pipeline demonstration
- `hd_computing.py`: HDC operations implementation
- `train.py`: Training loop and visualization utilities
- `resnet_model.py`: Modified ResNet architecture

## Implementation Details

### ResNet Model (`resnet_model.py`)
- Modified ResNet-18 with single-channel input
- Custom bottleneck blocks for dimensionality reduction
- Configurable output feature dimensionality

### HDC Component (`hd_computing.py`)
- Implements core HDC operations:
  - Random hypervector generation
  - Binding and superposition
  - Multiple similarity metrics
  - Prototype creation and voting

### Training Pipeline (`train.py`)
- Integrated training of both components
- Real-time performance visualization
- Learning rate scheduling
- Comprehensive metrics tracking

## Usage

1. Run `distracted_driving.ipynb` to:
   - Load and preprocess driver images
   - Initialize the hybrid model
   - Train the system
   - Evaluate performance
   - Visualize results

## Requirements

- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.

