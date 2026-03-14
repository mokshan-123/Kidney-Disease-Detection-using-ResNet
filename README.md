# Kidney CT Scan Classification using Deep Learning

## Overview

A deep learning model for automated classification of kidney CT scans into four categories: Normal, Cyst, Tumor, and Stone. The model achieves 99.78% test accuracy using transfer learning with ResNet50V2 architecture and addresses class imbalance through weighted training.

## Dataset

**Source:** [CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

**Distribution:**
- Cyst: 3,709 images (29.8%)
- Normal: 5,077 images (40.8%)
- Stone: 1,377 images (11.1%)
- Tumor: 2,283 images (18.3%)
- **Total:** 12,446 images

**Split:**
- Training + Validation: 85% (K-Fold Cross-Validation)
- Test Set: 15% (holdout)

## Model Architecture

**Base Model:** ResNet50V2 (pre-trained on ImageNet)

**Custom Classification Head:**
- GlobalAveragePooling2D
- BatchNormalization
- Dropout (0.4)
- Dense (256 units, ReLU)
- Dropout (0.3)
- Dense (4 units, Softmax)

**Training Strategy:**
1. Transfer Learning: Frozen ResNet50V2 base (20 epochs)
2. Fine-Tuning: Unfroze last 40 layers with reduced learning rate (15 epochs)

## Key Features

### Class Imbalance Handling
Weighted loss function applied during training to address dataset imbalance:
- Stone weight: 2.26 (highest - rarest class)
- Cyst weight: 0.84
- Tumor weight: 1.36
- Normal weight: 0.61 (lowest - most common class)

### Medical-Safe Data Augmentation
- Random rotation (±10°)
- Random zoom (±10%)
- Random contrast adjustment (±15%)
- Random brightness adjustment (±10%)
- **No horizontal flipping** (preserves anatomical orientation)

### Model Validation
- 3-Fold Cross-Validation for robust performance estimation
- Holdout test set for final evaluation
- Grad-CAM visualization for model interpretability

## Performance Metrics

### Cross-Validation Results
- Mean Accuracy: 96.69%
- Standard Deviation: ±0.35%

### Test Set Performance
| Metric | Value |
|--------|-------|
| Overall Accuracy | 99.78% |
| Weighted F1-Score | 0.9973 |
| Stone Recall | 0.990 |
| Tumor Recall | 0.997 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cyst | 0.996 | 1.000 | 0.998 |
| Normal | 0.999 | 0.997 | 0.998 |
| Stone | 0.990 | 0.990 | 0.990 |
| Tumor | 1.000 | 0.997 | 0.999 |

## Installation

### Requirements
```bash
pip install tensorflow>=2.13.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install scikit-learn>=1.3.0
pip install kagglehub
```

### Setup
```bash
git clone https://github.com/DPHeshanRanasinghe/Kidney-Disease-Detection-using-ResNet.git
cd kidney-ct-classification
```

## Usage

### Training
Run the Jupyter notebook cells sequentially:
1. Dataset download and preprocessing
2. Data loading with augmentation
3. Class weight calculation
4. Model building
5. K-Fold Cross-Validation training
6. Fine-tuning
7. Evaluation and visualization

### Prediction
```python
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('final_finetuned_model.keras')

# Preprocess image
img = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0])

print(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
```

## Model Interpretability

### Grad-CAM Visualization
The model includes Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize regions of interest:
- Highlights anatomical features used for classification
- Validates that the model focuses on relevant kidney pathology
- Provides transparency for clinical validation

## Technical Details

### Hyperparameters
- Image Size: 224×224×3
- Batch Size: 32
- Initial Learning Rate: 0.001
- Fine-Tuning Learning Rate: 1e-5
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

### Callbacks
- ModelCheckpoint: Saves best model based on validation loss
- EarlyStopping: Patience of 5-7 epochs
- ReduceLROnPlateau: Factor 0.5, patience 3 epochs

### Compute Requirements
- GPU: Recommended (NVIDIA with CUDA support)
- Training Time: ~2-3 hours on single GPU
- Memory: ~8GB GPU memory required

## Limitations

1. **Single-Source Dataset:** All images from one source; external validation needed (not cinfirmed)
2. **Class Distribution:** Dataset distribution may not reflect real clinical prevalence
3. **Image Protocol:** Model trained on specific CT scan protocols
4. **Edge Cases:** Performance on rare pathologies or complex cases not fully validated

## Future Work

- External validation on multi-institutional datasets
- Prospective clinical trial
- Extension to additional kidney pathologies
- Integration with PACS systems
- Real-time inference optimization

## Clinical Considerations

**Important:** This model is designed for research purposes only and has not been clinically validated or approved for diagnostic use. Any clinical application requires:
- Extensive external validation
- Regulatory approval (FDA, CE marking)
- Integration with radiologist workflow
- Continuous monitoring and quality assurance

---
## Contibuters
- Mokshan Colombage (Me)
- Heshan Ranasinghe : [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)

---
**Disclaimer:** This software is provided for research and educational purposes only. It is not intended for clinical diagnosis or patient care without appropriate validation and regulatory approval.
