# American Sign Language (ASL) Classification

## Project Overview
This project aims to classify isolated American Sign Language (ASL) signs, serving as a fundamental step towards various downstream applications such as sign language translation and fingerspelling. This classification model can assist the deaf and hard-of-hearing community in better communication through computer vision applications.

## Team Members
- James Zampa
- JeongYoon Lee
- Sailaja Bobburi
- Meixi Lu

## Table of Contents
- [Task Description](#task-description)
- [Dataset & Features](#dataset--features)
- [Preprocessing Methods](#preprocessing-methods)
- [Model Specifications](#model-specifications)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Analysis & Limitations](#analysis--limitations)
- [Future Improvements](#future-improvements)

## Task Description
The goal is to build a model to classify individual ASL signs using various machine learning models. The project involves preprocessing image data, feature extraction, and implementing different classifiers to compare performance.

## Dataset & Features
- **Image Data**: 200x200x3 pixel images of ASL signs captured in real-life environments.
- **Total Samples**: 34,190 images across 26 classes (A-Z).
- **Training & Testing Split**:
  - Training: 80% (27,352 samples)
  - Testing: 20% (6,838 samples)

### Preprocessing Methods
#### Image Processing
- **PCA Analysis**
- **Resizing** (30x30 for CNN models)
- **Grayscale Conversion**
- **Histogram Equalization**
- **Binary Image Conversion**
- **RGB Histogram Analysis**
- **YOLO-based Object Detection**

#### Hand Landmark Extraction
- **Google MediaPipe** used to extract 21x2 (x, y) hand landmarks.
- **Scaling and Centroid Alignment**
- **Feature Discretization** using a 10x10 grid, one-hot encoding, and value clipping.

## Model Specifications
We implemented and evaluated the following models:

### Classical ML Models
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Decision Tree (ID3, C4.5, CART)**

### Deep Learning Models
- **Convolutional Neural Networks (CNN)**
- **Deep Neural Networks (DNN)**

#### CNN Architecture
- **Convolutional Layers**
- **Max-Pooling Layers**
- **Fully Connected Layers**
- **Dropout Regularization (0.35)**
- **Learning Rate**: 0.001
- **Batch Size**: 100
- **Training Epochs**: 70 (Early Stopping at 5 Patience)

#### DNN Architecture
- **Fully Connected Dense Layers**
- **Activation Functions**: ReLU, Softmax
- **Input Size**: 784 nodes

## Evaluation Metrics
The models were evaluated using:
- **Precision**
- **Recall**
- **F1-Score**

## Results
| Model                  | Input Type   | Precision | Recall | F1-Score |
|------------------------|-------------|-----------|--------|----------|
| Baseline VGG16        | Images      | 99.22%    | 99.24% | 99.23%   |
| Decision Tree (ID3)   | Key Points  | 96.01%    | 96.01% | 96.00%   |
| Decision Tree (CART)  | Images      | 80.77%    | 80.73% | 80.70%   |
| Decision Tree (Bagging, n=50) | Images | 94.30%  | 94.28% | 94.27%   |
| KNN                   | Key Points  | 97.79%    | 97.79% | 97.79%   |
| KNN                   | Images      | 94.73%    | 94.73% | 94.73%   |
| Naive Bayes           | Key Points  | 94.37%    | 93.47% | 93.80%   |
| Naive Bayes           | Images      | 45.91%    | 30.98% | 32.64%   |
| CNN                   | Images      | 96.00%    | 96.00% | 96.00%   |
| CNN (Binary)          | Images      | 41.60%    | 41.60% | 41.60%   |
| CNN (YOLO)            | Images      | 96.33%    | 96.33% | 96.33%   |
| DNN                   | Key Points  | 96.23%    | 95.91% | 95.88%   |
| DNN                   | Images      | 93.00%    | 93.00% | 93.00%   |

## Analysis & Limitations
### Key Factors Contributing to High Performance
- **Effective Data Preprocessing**: Standardization improved model accuracy.
- **Balanced Dataset**: Equal representation of each sign class helped avoid bias.
- **Feature Similarity**: Signs with similar features were easier to classify with ML models.
- **CNN and DNN**: Captured spatial relationships and hierarchical representations effectively.

### Limitations
- **Computational Intensity**: CNN/DNN models require substantial GPU resources.
- **Curse of Dimensionality**: Affects KNN in high-dimensional spaces.
- **Independence Assumption in Naive Bayes**: ASL signs have interdependent features, reducing its performance on images.
- **Overfitting Risks**: Especially in DNN models if regularization is not properly applied.

## Future Improvements
- **Ensemble Learning**: Combining models using bagging or boosting.
- **SHAP Value Analysis**: To improve model interpretability.
- **Dual-Input CNN Framework**: Processing images and key points together for better classification.
- **Super-Resolution Models**: Enhancing image clarity before classification.
- **Segmentation Model Instead of YOLO**: Using semantic segmentation for better feature extraction.

## References
- Dataset and additional resources: [Kaggle Notebook](https://www.kaggle.com/code/harits/vgg16-asl-recognition-model-explainability/notebook)


---

### How to Use
1. Clone the repository:
   ```bash
   git clone <repo_link>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

For further questions or contributions, please open an issue in the repository!
