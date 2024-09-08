# Cat vs Dog Image Classification

## Project Overview
This project implements image classification using different techniques to classify images of cats and dogs. The project explores traditional feature extraction methods like Histogram of Oriented Gradients (HOG) and deep learning techniques like the VGG16 convolutional neural network. The goal is to compare these approaches and evaluate their performance on a standard dataset.

## Techniques Used
| **Technique** | **Description** | **Accuracy** |
|---------------|-----------------|--------------|
| **HOG (Histogram of Oriented Gradients)** | Feature extraction method that emphasizes edge detection and orientation | **67.14%** |
| **VGG16 (Pretrained CNN)** | Deep learning model pretrained on the ImageNet dataset | **65.71%** |
| **Scaling** | Standard feature scaling to normalize input data before classification | **67.14%** |

## How It Works
1. **Data Preprocessing**:
   - The dataset consists of labeled images of cats and dogs.
   - Data is loaded, resized, and preprocessed for each technique.
   
2. **HOG**:
   - The HOG method is applied to extract features that capture the edges and gradient orientations in the images.
   - These features are then passed to a classifier (SVM) to predict whether an image is a cat or a dog.
   
3. **VGG16**:
   - The VGG16 model, pretrained on ImageNet, is used to extract features from the images. The top layers are removed, and the remaining layers are used to extract convolutional features.
   - A custom classifier is trained on these features for binary classification.

4. **Scaling**:
   - The images are scaled using standard scaling techniques before being passed into the classifier.

## Requirements
To run this project, you'll need the following libraries:
- `tensorflow`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`

Install them using the following command:

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn
