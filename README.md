# MNIST Handwritten Digit Recognition with PyTorch CNN

This repository contains the code for training and evaluating a Convolutional Neural Network (CNN) model for handwritten digit recognition using PyTorch. The CNN model is trained on the MNIST dataset, which consists of 60,000 28x28 grayscale images of handwritten digits from 0 to 9.

## Introduction

Welcome to the MNIST Handwritten Digit Recognition project! In this project, we have developed a deep learning model using PyTorch that can accurately recognize handwritten digits. By training our CNN model on the popular MNIST dataset, we aim to achieve high accuracy in classifying different handwritten numbers. Here are the key steps we followed to build and train our CNN model:

- **Dataset Preparation**: We obtained the MNIST dataset, a widely-used benchmark dataset for handwritten digit recognition tasks. The dataset consists of 60,000 training images and 10,000 test images. Each image is grayscale and has a resolution of 28x28 pixels. We split the dataset into training and test sets to evaluate the performance of our model.

- **Model Architecture**: We designed a CNN model architecture using PyTorch. The model consists of convolutional layers, pooling layers, and fully connected layers. This architecture allows the model to learn meaningful representations of the input images and make accurate predictions.

- **Data Preprocessing**: Before training the model, we performed data preprocessing steps such as normalization and resizing. Normalization ensures that all pixel values fall within a consistent range, which helps in stabilizing the learning process. Resizing the images to a standardized size enables efficient processing and reduces computational requirements.

- **Training the Model**: To train the CNN model, we used stochastic gradient descent (SGD) optimization with backpropagation. We used the cross-entropy loss function to measure the discrepancy between the predicted and true labels. By iteratively updating the model's weights based on the calculated gradients, the model learned to make better predictions over time.

- **Evaluation and Testing**: Once the model was trained, we evaluated its performance on the test dataset. We measured metrics such as accuracy, which indicates the percentage of correctly classified digits. This allowed us to assess how well the model generalizes to unseen data and determine its effectiveness in recognizing handwritten digits.

- **Saving and Loading the Model**: We provided functionality to save the trained model's weights and architecture using PyTorch's serialization capabilities. This allows us to reuse the model for future predictions or further fine-tuning. The saved model can be loaded and used to classify new handwritten digits efficiently.

## Requirements

- Python 3.7 or above
- PyTorch 1.8 or above
- TorchVision 0.9 or above

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/FarzadMalik/MNIST_with_pytorch_CNN.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the MNIST dataset and extract it into the `data` directory.
4. Train the CNN model.
5. Evaluate the trained model.

Feel free to explore the code and modify it according to your needs. You can adjust the hyperparameters, add more layers to the network, or even experiment with different datasets for training.

## Project Structure

The repository is organized as follows:

- `data/`: Directory to store the MNIST dataset.
- `models/`: Contains the CNN model architecture and functions for saving/loading models.
- `utils/`: Utility functions for data preprocessing and the training loop.
- `train.py`: Script to train the CNN model.
- `evaluate.py`: Script to evaluate the trained model.
- `requirements.txt`: List of required Python packages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you have any questions or suggestions, feel free to reach out. Happy coding!
