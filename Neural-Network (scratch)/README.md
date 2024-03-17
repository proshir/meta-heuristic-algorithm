# Neural Network Implementation

This repository contains Python code for implementing a simple neural network from scratch using NumPy. The code demonstrates various functionalities including building the model, plotting decision boundaries, experimenting with different hyperparameters, and extending the network to handle multiple classes.

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Experimentation](#experimentation)
5. [Extension](#extension)
6. [Conclusion](#conclusion)

## Introduction
This neural network implementation is primarily aimed at educational purposes to understand the fundamentals of neural networks and their training process. It provides functionalities to build a neural network model, train it using gradient descent, plot decision boundaries, experiment with various hyperparameters, and extend the model for multi-class classification.

## Getting Started
To get started, ensure you have Python installed on your system along with the required libraries listed below:
- NumPy
- scikit-learn
- matplotlib
- pandas

## Usage
The main functionalities of the neural network implementation are encapsulated within the `build_model` and `Plot` functions.

1. **Building the Model**: Use the `build_model` function to create a neural network model with customizable parameters such as the number of nodes in the hidden layer (`nn_hdim`), activation function, number of passes through the training data, and more.

```python
# Example usage to build a model with a 3-dimensional hidden layer
model = build_model(3)
```

2. **Plotting Decision Boundaries**: The `Plot` function allows you to visualize the decision boundaries of the trained model.

```python
# Example usage to plot decision boundaries
Plot()
```

## Experimentation
The repository provides a comprehensive set of experiments to understand the impact of various hyperparameters on the model performance.

### Varying Hidden Layer Size
Experiments are conducted by varying the size of the hidden layer to observe its effect on the decision boundaries and model performance.

### Mini-batch Gradient Descent
An implementation of mini-batch gradient descent is provided as an alternative to batch gradient descent, which typically performs better in practice.

### Annealing Schedule for Learning Rate
An annealing schedule for the gradient descent learning rate is implemented to improve convergence and avoid overshooting.

### Activation Functions
Experimentation with different activation functions such as tanh, logistic, ReLU, and ArcTan is provided to observe their impact on the model's behavior and convergence.

### Multi-class Classification
Extension of the network from binary to multi-class classification is demonstrated using appropriate datasets and model modifications.

## Conclusion
This neural network implementation serves as a valuable resource for understanding the fundamental concepts of neural networks, training algorithms, and hyperparameter tuning. Users can experiment with various configurations to gain insights into the behavior and performance of neural networks.
