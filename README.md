# DHC-intern-phase-2-task-4
# Multimodal Housing Price Prediction using Deep Learning

## Project Overview

This project implements a **Multimodal Deep Learning Model** for housing price prediction using both:

1. **Tabular Data**

   * Square Footage (SqFt)
   * Number of Bedrooms
   * Number of Bathrooms
   * House Age

2. **Image Data**

   * Synthetic house images generated for each property

The model combines information from both modalities through a **Late Fusion Architecture**, where image features extracted by a Convolutional Neural Network (CNN) are merged with tabular features extracted by a Multi-Layer Perceptron (MLP).

---

## Objectives

* Generate a synthetic multimodal housing dataset.
* Process numerical and image-based features.
* Build a multimodal neural network using PyTorch.
* Train the model to predict housing prices.
* Evaluate performance using regression metrics.
* Visualize predicted versus actual house prices.

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* PIL (Python Imaging Library)
* Matplotlib

---

## Dataset Generation

The notebook automatically creates a synthetic housing dataset consisting of:

### Tabular Features

* Square Footage (`sqft`)
* Bedrooms (`bedrooms`)
* Bathrooms (`bathrooms`)
* House Age (`age`)

### Target Variable

* House Price (`price`)

A deterministic pricing formula with Gaussian noise is used to simulate realistic housing prices.

### Image Features

For each property, a synthetic RGB image (64×64 pixels) is generated and stored in:

```text
housing_dataset/images/
```

The tabular data is stored in:

```text
housing_dataset/tabular_data.csv
```

---

## Model Architecture

### 1. CNN Branch (Image Processing)

The image branch consists of:

* Convolution Layer (3 → 16 channels)
* ReLU Activation
* Max Pooling
* Convolution Layer (16 → 32 channels)
* ReLU Activation
* Max Pooling
* Flatten Layer
* Fully Connected Layer

### 2. MLP Branch (Tabular Processing)

The tabular branch includes:

* Fully Connected Layer
* ReLU Activation
* Fully Connected Layer
* ReLU Activation

### 3. Fusion Layer

Features from both branches are concatenated and passed through:

* Fully Connected Layer
* ReLU Activation
* Output Layer

The output is a single predicted house price.

---

## Data Preprocessing

### Tabular Data

* Loaded using Pandas
* Split into training and testing sets
* Standardized using `StandardScaler`

### Image Data

* Resized to 64×64 pixels
* Converted to tensors
* Normalized using ImageNet normalization values

---

## Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Optimizer     | Adam                     |
| Learning Rate | 0.005                    |
| Loss Function | Mean Squared Error (MSE) |
| Epochs        | 50                       |
| Batch Size    | Defined in DataLoader    |
| Framework     | PyTorch                  |

---

## Evaluation Metrics

The model performance is measured using:

### Mean Absolute Error (MAE)

Measures the average prediction error.

### Root Mean Squared Error (RMSE)

Measures the standard deviation of prediction errors.

---

## Output Visualization

A scatter plot is generated showing:

* Actual House Prices
* Predicted House Prices
* Perfect Fit Reference Line

This visualization helps assess model accuracy and regression quality.

---

## Project Structure

```text
project/
│
├── task 4.ipynb
│
├── housing_dataset/
│   ├── tabular_data.csv
│   └── images/
│       ├── house_0.jpg
│       ├── house_1.jpg
│       └── ...
│
└── README.md
```

---

## Learning Outcomes

After completing this project, you will understand:

* Multimodal machine learning concepts
* CNN-based image feature extraction
* MLP-based tabular feature learning
* Feature fusion techniques
* Regression using deep neural networks
* Model evaluation and visualization in PyTorch

---

## Author

Deep Learning Project: Multimodal Housing Price Prediction

Developed using PyTorch for educational purposes and multimodal learning experimentation.
