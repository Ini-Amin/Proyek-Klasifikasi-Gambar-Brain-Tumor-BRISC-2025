# Brain Tumor Classification 🧠

**A deep learning project for detecting and classifying brain tumors from MRI scans using Convolutional Neural Networks.**

---

## 📋 Project Overview

**Status:** Self-initiated, Individual Project / BRISC 2025 Submission  
**Built With:** TensorFlow • Keras • Python • Jupyter Notebook  
**Challenge:** BRISC 2025 - Brain Tumor Classification Competition

### One-Liner
A CNN-based image classification system that identifies four types of brain tumors (glioma, meningioma, pituitary, no tumor) from MRI scans with high accuracy.

---

## 🎯 The Story: Why I Built This

Medical imaging is one of the most impactful applications of machine learning. Early and accurate tumor detection can save lives. I decided to tackle this challenge to understand:

- How deep learning handles complex image classification
- Real-world dataset challenges (imbalanced data, augmentation)
- Model optimization and performance evaluation
- Deploying ML models in different formats

**The Goal:** Build a model that can accurately classify brain tumors to assist radiologists.

---

## 🧬 Dataset

**Source:** BRISC 2025 - Brain Tumor Classification (Kaggle)

| Aspect | Details |
|--------|---------|
| **Classes** | Glioma, Meningioma, Pituitary, No Tumor |
| **Total Images** | ~3,000+ MRI scans |
| **Split** | 70% training / 15% validation / 15% test |
| **Stratification** | Balanced per class to prevent bias |
| **Image Size** | 224 × 224 × 3 (RGB) |

---

## 🧠 Model Architecture

### Sequential CNN (4 Convolutional Blocks)

```
Input (224x224x3)
    ↓
Block 1: Conv2D(32) → BatchNorm → MaxPooling2D → Dropout(0.25)
    ↓
Block 2: Conv2D(64) → BatchNorm → MaxPooling2D → Dropout(0.25)
    ↓
Block 3: Conv2D(128) → BatchNorm → MaxPooling2D → Dropout(0.25)
    ↓
Block 4: Conv2D(256) → BatchNorm → MaxPooling2D → Dropout(0.25)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) → ReLU → Dropout(0.4)
    ↓
Dense(256) → ReLU → Dropout(0.3)
    ↓
Dense(4) → Softmax [Glioma, Meningioma, Pituitary, No Tumor]
```

### Key Design Decisions

- **Batch Normalization:** Stabilizes training and allows higher learning rates
- **Dropout:** Prevents overfitting by randomly deactivating neurons
- **GlobalAveragePooling:** Reduces dimensionality while preserving spatial information
- **Early Stopping:** Prevents overfitting by monitoring validation loss

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~90%+ |
| **Precision** | High per-class (see confusion matrix) |
| **Recall** | Optimized to minimize false negatives |
| **Loss Function** | Categorical Crossentropy |
| **Optimizer** | Adam |

---

## 📁 Project Structure

```
Proyek-Klasifikasi-Gambar-Brain-Tumor-BRISC-2025/
├── notebook.ipynb              # Complete analysis & training
├── submission/
│   ├── saved_model/            # TensorFlow SavedModel format
│   │   ├── saved_model.pb
│   │   └── variables/
│   ├── tflite/                 # Mobile model (TensorFlow Lite)
│   │   ├── model.tflite
│   │   └── label.txt
│   ├── tfjs_model/             # Web model (TensorFlow.js)
│   │   ├── model.json
│   │   └── group1-shard1of1.bin
│   └── README.md
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites
```
Python 3.8+
pip (package manager)
GPU (recommended, but CPU works)
```

### Installation

```bash
# Clone repository
git clone https://github.com/Ini-Amin/Proyek-Klasifikasi-Gambar-Brain-Tumor-BRISC-2025.git
cd Proyek-Klasifikasi-Gambar-Brain-Tumor-BRISC-2025

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebook.ipynb
```

### Data Preparation

```bash
# Download dataset from Kaggle
# Place in: data/brain_tumor/

# Dataset structure should be:
# data/brain_tumor/
# ├── glioma/
# ├── meningioma/
# ├── pituitary/
# └── no_tumor/
```

---

## 🎓 What I Learned

### Deep Learning Concepts
- ✅ **CNN Architecture** - How convolutional layers extract features
- ✅ **Transfer Learning** - Potential for using pre-trained models
- ✅ **Data Augmentation** - Rotating, flipping, zooming to increase dataset
- ✅ **Regularization** - Preventing overfitting with dropout and batch norm
- ✅ **Hyperparameter Tuning** - Learning rate, batch size, epochs
- ✅ **Model Evaluation** - Confusion matrices, precision-recall, ROC curves

### Real-World ML Challenges
- **Class Imbalance** - Some tumor types appear more frequently (handled with stratification)
- **Data Quality** - Medical images have varying quality and angles
- **Generalization** - Models trained on one dataset may not work on another
- **Computational Cost** - Training large models is expensive
- **Interpretability** - Understanding what the model "sees" in images

### Model Deployment
- **Multiple Formats:**
  - **SavedModel** - For TensorFlow serving
  - **TFLite** - For mobile and embedded devices
  - **TFJS** - For browser-based predictions
- Why this matters: Different deployment targets need different formats

### What I'd Do Differently Now
- ✅ Add explainability (Grad-CAM) to visualize what model focuses on
- ✅ Use ensemble methods (combine multiple models)
- ✅ Implement class weighting for imbalanced data
- ✅ Add comprehensive logging and monitoring
- ✅ Create a web/mobile app for inference
- ✅ Compare with transfer learning approaches
- ✅ Add uncertainty estimation for less confident predictions

---

## 🔍 How to Use the Model

### For Inference (Python)

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('submission/saved_model')

# Load and preprocess image
img = Image.open('path/to/mri_scan.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

print(f"Predicted: {classes[class_idx]}")
print(f"Confidence: {predictions[0][class_idx]:.2%}")
```

### For Web (JavaScript with TensorFlow.js)

```javascript
// Load model
const model = await tf.loadGraphModel('tfjs_model/model.json');

// Preprocess image
const canvas = document.getElementById('imageCanvas');
const prediction = await model.predict(tf.browser.fromPixels(canvas).expandDims(0));

// Get result
const classIdx = prediction.argMax(-1).dataSync()[0];
const classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'];
console.log(`Predicted: ${classes[classIdx]}`);
```

### For Mobile (TensorFlow Lite)

```swift
// Swift example for iOS
import TensorFlowLite

let interpreter = try Interpreter(modelPath: modelPath)
let output = try interpreter.invoke(inputData)
```

---

## 📈 Evaluation Results

- **Training accuracy:** ~95%
- **Validation accuracy:** ~92%
- **Test accuracy:** ~90%+

(See confusion matrix in notebook for per-class breakdown)

---

## 🚧 Future Improvements

- [ ] Implement Grad-CAM for explainability
- [ ] Build web app for inference
- [ ] Create mobile app (iOS/Android)
- [ ] Use transfer learning (ResNet50, EfficientNet)
- [ ] Ensemble multiple models
- [ ] Real-time prediction dashboard
- [ ] Integrate with DICOM viewer for clinical use
- [ ] Add uncertainty quantification
- [ ] Compare with radiologist accuracy

---

## 📚 Key Files to Review

- **Full Analysis:** [notebook.ipynb](./notebook.ipynb) - Step-by-step walkthrough
- **Model Details:** [submission/saved_model/](./submission/saved_model/)
- **Deployment Formats:** [submission/tflite/](./submission/tflite/) & [submission/tfjs_model/](./submission/tfjs_model/)

---

## 💡 Reflection

This project showed me that **machine learning isn't just about building high-accuracy models—it's about solving real problems responsibly.** In medical imaging, a 90% accurate model that misses tumors is worse than useful. This taught me to think about:

- **False negatives vs. false positives** - Which error is costlier?
- **Model interpretability** - Doctors need to trust the predictions
- **Deployment considerations** - How will this model actually be used?
- **Continuous improvement** - Models degrade over time with new data

This is why foundations matter: proper validation, testing, and documentation ensure AI systems are reliable and trustworthy.

---

## 📖 References & Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Guide](https://keras.io/)
- [Deep Learning Book - Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)
- [Medical Image Analysis Papers](https://arxiv.org/list/eess.IV/recent)

---

## 📞 Contact

Questions about this project?

- **GitHub:** [@Ini-Amin](https://github.com/Ini-Amin)
- **Email:** [your-email@example.com]

---

**Last Updated:** 2026-06-04  
**Status:** BRISC 2025 Submission Complete
