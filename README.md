# 🌿 Plant Disease Classification System

An end-to-end Computer Vision project using Deep Learning and Transfer Learning to classify plant diseases from leaf images.

## 🚀 Overview
Plant diseases cause severe yield losses in agriculture. Early diagnosis is crucial for mitigating damage. This project provides a web-based diagnosis tool powered by **MobileNetV2** and **Streamlit** to instantly classify plant leaf images into **38 different categories** (healthy vs. diseased across various plant species).

## 🛠️ Tech Stack & Tools
* **Deep Learning Framework:** TensorFlow / Keras
* **Pre-trained Model:** MobileNetV2 (Transfer Learning)
* **Web App Framework:** Streamlit
* **Deployment:** Hugging Face Spaces / GitHub
* **Dataset:** PlantVillage (54,305 images across 38 classes)

## 📊 Model Performance
* **Training Accuracy:** ~97.5%
* **Validation Accuracy:** ~96.0%
* **Key Architecture Features:** Rescaling Layer, MobileNetV2 Base (Frozen), GlobalAveragePooling2D, Dense(128, ReLU), Dropout(0.3), Dense(38, Softmax).

## 📁 Repository Structure
