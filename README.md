# Handwritten Digit Recognition using CNN (MNIST)

## 📌 Overview
A TensorFlow/Keras **Convolutional Neural Network (CNN)** that classifies handwritten digits (0–9) from the **MNIST** dataset loaded from CSV files. The script handles data loading, preprocessing, model training, evaluation, visualization, prediction export, and model saving.

---

## ✨ Features
- Load & validate MNIST data from `train.csv` / `test.csv`
- Normalize & reshape to `(28, 28, 1)` tensors
- One‑hot encode labels
- Split training/validation (80/20)
- Visualize class distribution & sample images
- CNN with 3 conv‑pool blocks + dense head + dropout
- EarlyStopping & ReduceLROnPlateau callbacks
- Accuracy & loss training curves
- Confusion matrix + classification report
- Random sample predictions with confidence
- Export `predictions.csv`
- Save trained model (`digit_recognition_model.h5`)
- Generate performance summary

---

## 🏗 Model Architecture
```
Input: 28x28x1
Conv2D(32,3x3) + ReLU + MaxPool
Conv2D(64,3x3) + ReLU + MaxPool
Conv2D(128,3x3) + ReLU + MaxPool
Flatten
Dense(128) + ReLU
Dropout(0.5)
Dense(10) + Softmax
```

---

## ⚙️ Requirements
Install dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---

## 📂 Data Format
Your CSVs must follow the Kaggle MNIST digit-recognizer style:

**train.csv**
- 785 columns total: `label` + 784 pixel values (`pixel0`..`pixel783`)
- One row per 28x28 image

**test.csv**
- 784 pixel columns; optional `label` column (if present, evaluation will include metrics)

---

## ▶️ Quick Start
```bash
git clone https://github.com/your-username/digit-recognition-cnn.git
cd digit-recognition-cnn

# Place train.csv and test.csv in this folder

python digit_recognition_cnn.py
```

---

## 📊 What You Get After Running
Output artifacts (created in the project folder):
| File | Description |
|------|-------------|
| `predictions.csv` | ImageId → predicted digit for test set |
| `digit_recognition_model.h5` | Saved trained Keras model |
| Training plots | Accuracy & loss curves (displayed; save manually if needed) |
| Confusion matrix | Displayed heatmap; save manually if needed |

---

## 🔧 Configuration Tips
You can edit defaults in **`digit_recognition_cnn.py`**:
- Change file names in `load_and_preprocess_data()`
- Adjust validation split percentage
- Modify CNN layers or dropout
- Set training `epochs` in `train_model()`
- Tune callback patience or min LR

---

## 🚀 Ideas to Extend
- Data augmentation (random shifts/rotations)
- Learning rate schedulers / optimizer experiments
- Batch normalization layers
- Model quantization / TFLite export
- Simple Gradio or Streamlit demo app

---

## 👩‍💻 Authors
**Ramisha Ikram (FA22-BCE-024)**  
**Tahreem Shahid (FA22-BCE-035)**

---

## 📜 License
MIT License (recommended) — update below if different.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---


