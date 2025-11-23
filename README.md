# 🌾 Rice Leaf Disease Prediction

This project focuses on building a deep learning–based image classification system to detect **rice leaf diseases** using Convolutional Neural Networks (CNN) and Transfer Learning.  
The notebook (`Rice_leaf.ipynb`) walks through dataset preparation, data augmentation, model building, training, evaluation, and reporting of challenges.

---

## 📘 Problem Statement

Rice crops are highly susceptible to various diseases that can significantly reduce yield and quality.  
Early detection of rice leaf diseases helps farmers take immediate action and prevent large-scale damage.

This project aims to:
- Classify rice leaf images into **diseased** or **healthy** categories  
- Use CNN and Transfer Learning models for accurate prediction  
- Compare model performance and identify the best approach  

---

## 🧰 Importing Necessary Libraries

The notebook uses:
- TensorFlow / Keras  
- OpenCV  
- Matplotlib, Seaborn  
- NumPy  
- Scikit-learn  

---

## 📂 Loading Dataset

The dataset includes multiple rice leaf images categorized into different classes.  
Images are loaded from folders using TensorFlow's image preprocessing utilities.

---

## 🔄 Data Augmentation

To prevent overfitting and enrich the dataset, the following augmentation techniques are applied:

- Rotation  
- Zoom  
- Horizontal & vertical flips  
- Shifts  
- Rescaling  

---

## 🔍 Exploratory Data Analysis (EDA)

EDA includes:
- Visualizing sample images  
- Understanding class distribution  
- Checking for imbalance  
- Observing variations between healthy and diseased leaves  

---

## 🧠 Building CNN Model

A custom CNN model is built consisting of:
- Convolution layers  
- MaxPooling  
- Dropout  
- Dense layers  

Key steps include:
- Defining architecture  
- Compiling with appropriate loss function  
- Training on augmented data  

---

## 🔁 Transfer Learning

The notebook implements Transfer Learning using pre-trained models such as:
- VGG16  
- ResNet  
- MobileNetV2  
(or whichever is used in the notebook)

Steps include:
- Freezing base layers  
- Adding custom classification head  
- Fine-tuning dense layers  

Transfer learning generally provides higher accuracy with fewer epochs.

---

## 🏋️ Model Training & Compilation

Training settings typically include:
- Optimizer (Adam)  
- Loss function (categorical crossentropy)  
- Metrics (accuracy)  
- Epochs (based on notebook)  

Training and validation accuracy/loss curves are analyzed.

---

## 📊 Model Evaluation

The notebook includes:
- Classification accuracy  
- Confusion matrix  
- Precision, recall, F1-score  
- Loss/accuracy plots across epochs  

---

## 👁️ Visualizing Predictions

The notebook also visualizes:
- Actual vs Predicted labels  
- Sample images with model predictions  
- Misclassified examples  

This helps understand how well the model distinguishes between rice leaf diseases.

---

## 🥇 Model Comparison Report

Models compared include:
- Custom CNN  
- Transfer Learning models  

The comparison evaluates:
- Accuracy  
- Training time  
- Validation performance  
- Generalization  

Transfer Learning usually outperforms base CNN models.

---

## 📝 Challenges Faced

Key challenges covered in the notebook:
- Dataset imbalance  
- Overfitting in CNN baseline model  
- Image noise and variation  
- Long training time for large architectures  
- Choosing optimal augmentation strategies  

---

## ▶️ How to Run the Notebook

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Download and place the image dataset in the project folder.
3. Open the notebook:
   ```bash
   jupyter notebook Rice_leaf.ipynb
4. Run all cells sequentially.
