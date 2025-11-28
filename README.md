# shoe_classification

---

# 👟 Shoe Brand Classification using MobileNetV2

A deep learning project for classifying shoe brands (Nike, Adidas, Converse) using transfer learning with MobileNetV2, implemented in TensorFlow/Keras.

---

## 📌 Project Overview

This project implements an image classification system to recognize three shoe brands:  
- **Nike**  
- **Adidas**  
- **Converse**  

The model is built using **MobileNetV2** as the base architecture and fine-tuned with a custom classifier head. It achieves **75.56% accuracy** on the validation set.

---

## 📁 Dataset

The dataset consists of 600 training images and 90 validation images (200 and 30 per brand, respectively).  
All images are RGB JPEGs resized to **240×240 pixels**.

### Data Augmentation
To improve generalization, the following augmentations were applied during training:
- Random rotation (±40°)
- Width/height shift (±30%)
- Shear (20%)
- Zoom (±30%)
- Horizontal flip
- Brightness adjustment (70%–130%)
- Pixel normalization ([0, 1])

---

## 🧠 Model Architecture

The model is based on **MobileNetV2** pre-trained on ImageNet, with the following custom top layers:

- Global Average Pooling
- Dropout (0.5)
- Dense (128 units, ReLU)
- Batch Normalization
- Dense (3 units, Softmax)

### Training Configuration
- **Optimizer**: Adam (learning rate = 0.0001)  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 30  
- **Batch Size**: 32  
- **Callbacks**: Early Stopping, ReduceLROnPlateau

---

## 📊 Results

### Validation Performance
- **Accuracy**: 75.56%
- **Loss**: 0.5259

### Hyperparameter Tuning
| Learning Rate | Dropout | Dense Units | Epochs | Accuracy |
|---------------|---------|-------------|--------|----------|
| 0.0001        | 0.5     | 128         | 30     | 75.56%   |
| 0.001         | 0.5     | 128         | 20/30  | 80.00%   |
| 0.00001       | 0.5     | 128         | 11/30  | 34.44%   |
| 0.0001        | 0.3     | 128         | 27/30  | 77.78%   |
| 0.0001        | 0.7     | 128         | 19/30  | 70.00%   |
| 0.0001        | 0.5     | 64          | 30     | 74.44%   |
| 0.0001        | 0.5     | 256         | 30     | 73.33%   |

---

## 🛠️ How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/shoe-classification.git
cd shoe-classification
```

### 2. Install Dependencies
```bash
pip install tensorflow matplotlib numpy
```

### 3. Prepare Data
Organize your dataset as follows:
```
dataset/
  train/
    nike/
    adidas/
    converse/
  validation/
    nike/
    adidas/
    converse/
```

### 4. Train the Model
Run the training script:
```python
python train.py
```

### 5. Test the Model
Update the path in the test script and run:
```python
python test.py
```

---

## 📈 Training Curves

The training process includes visualization of accuracy and loss curves for both training and validation sets.

---

## 👨‍💻 Author

**FAN Yutao**  
- Student ID: 13024637  
- Course: COMP S492F  

---

## 📄 License

This project is for academic purposes only.

---
