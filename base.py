
# Part 1: Setup and Data Preparation

# Import necessary libraries
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# Data Loading
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset into training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Part 2: Model Building

# Use LogisticRegression to fit the model
model = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='auto')
model.fit(X_train, y_train)

# Part 3: Report model accuracy, Model size, Inference time of Logistic regression model

# Measure the model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Measure the model size by converting model parameters to numpy arrays and calculating memory usage
model_size = sum(param.nbytes for param in [model.coef_, model.intercept_]) / 1024  # size in KB

# Measure inference time
start_time = time.time()
model.predict(X_test)
inference_time = time.time() - start_time

print(f"Original Model Accuracy: {accuracy * 100:.2f}%")
print(f"Original Model Size: {model_size:.2f} KB")
print(f"Original Inference Time: {inference_time:.4f} seconds")

# Part 4: Create a function to quantize the model

def quantize_model(model, scale_factor=2**7):
    quantized_weights = {}
    # Extract coefficients and intercept
    weights = model.coef_
    intercept = model.intercept_

    # Quantize weights
    for i in range(weights.shape[0]):  # For each class
        param = weights[i]  # Get the coefficients for class i
        # Normalize the parameters to the range [-1, 1]
        param_max = np.max(np.abs(param))
        if param_max != 0:  # Avoid division by zero
            normalized_param = param / param_max
        else:
            normalized_param = param
        # Scale to [-scale_factor, scale_factor] and convert to int8
        scaled_param = np.clip(normalized_param * scale_factor, -128, 127)
        quantized_weights[f'class_{i}_coef'] = torch.tensor(scaled_param, dtype=torch.int8)

    # Quantize intercept
    for i in range(len(intercept)):
        intercept_max = np.max(np.abs(intercept[i]))
        if intercept_max != 0:
            normalized_intercept = intercept[i] / intercept_max
        else:
            normalized_intercept = intercept[i]
        scaled_intercept = np.clip(normalized_intercept * scale_factor, -128, 127)
        quantized_weights[f'class_{i}_intercept'] = torch.tensor(scaled_intercept, dtype=torch.int8)

    return quantized_weights

# Quantize the model
quantized_model = quantize_model(model)

# Part 5: Create a function to inference using the quantized model

def quantized_inference(quantized_model, X, scale_factor=2**7):
    # Rescale quantized weights for inference
    weights = {}
    for name, param in quantized_model.items():
        param = param.float() / scale_factor  # Rescale the weights
        weights[name] = param.numpy()  # Convert back to numpy for use in scikit-learn
    
    # Prepare model coefficients and intercepts for scikit-learn
    coef_list = []
    intercept_list = []
    for i in range(len(weights) // 2):
        coef_list.append(weights[f'class_{i}_coef'])
        intercept_list.append(weights[f'class_{i}_intercept'])
    
    # Create a new Logistic Regression model with the quantized weights
    quantized_model_sk = LogisticRegression(multi_class='auto', max_iter=2000)
    quantized_model_sk.coef_ = np.array(coef_list)
    quantized_model_sk.intercept_ = np.array(intercept_list)
    quantized_model_sk.classes_ = np.unique(y)

    return quantized_model_sk.predict(X)

# Measure accuracy of quantized model
quantized_y_pred = quantized_inference(quantized_model, X_test)
quantized_accuracy = accuracy_score(y_test, quantized_y_pred)

# Measure the quantized model size by summing the memory of quantized weights
quantized_model_size = sum(param.numel() for param in quantized_model.values())  # Number of elements in all tensors

# Convert size from elements to KB, considering int8 values (1 byte each)
quantized_model_size = (quantized_model_size * 1) / 1024  # size in KB

# Measure inference time of the quantized model
start_time = time.time()
quantized_inference(quantized_model, X_test)
quantized_inference_time = time.time() - start_time

# Part 6: Report Quantized model accuracy, size, and inference time
print(f"Quantized Model Accuracy: {quantized_accuracy * 100:.2f}%")
print(f"Quantized Model Size: {quantized_model_size:.2f} KB")
print(f"Quantized Inference Time: {quantized_inference_time:.4f} seconds")

# Model Size Comparison
print(f"Size Reduction: {(model_size - quantized_model_size) / model_size * 100:.2f}%")

