# CUDA Accelerated Logistic Regression for Email Classification

## Overview

This project implements a spam email classifier using logistic regression with CUDA acceleration via CuPy. The model is trained using scikit-learn for feature extraction. It offers functionality to check the model's accuracy through an 80/20 testing split and can also classify new emails.

## Features

- Utilizes logistic regression for email classification.
- CUDA acceleration with CuPy for faster computation.
- Feature extraction using scikit-learn's CountVectorizer.
- Ability to evaluate model accuracy with an 80/20 testing split.
- Capability to classify new emails.

## Requirements

- Python 3.x
- CuPy
- NumPy
- pandas
- scikit-learn

