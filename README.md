# MNIST Handwritten Digit Classification Using Convolutional Neural Networks (CNN)

Developed a Convolutional Neural Network (CNN) to classify grayscale handwritten digit images from the MNIST dataset, achieving high accuracy through model architecture optimization, normalization, and regularization techniques.

## Project Overview

This project applies deep learning techniques to the MNIST dataset, a foundational dataset in computer vision. By building and refining a CNN architecture, the model accurately classifies handwritten digits into 10 distinct categories, showcasing the effectiveness of CNNs in image recognition tasks.

## Dataset

The **MNIST dataset** contains:
- 60,000 training images and 10,000 test images.
- Each image is a **28x28 grayscale pixel** representation.
- 10 digit classes:
  - 0 – Zero
  - 1 – One
  - 2 – Two
  - 3 – Three
  - 4 – Four
  - 5 – Five
  - 6 – Six
  - 7 – Seven
  - 8 – Eight
  - 9 – Nine

## Objectives

- Preprocess image data with normalization techniques.
- Design, build, and train a Convolutional Neural Network (CNN).
- Optimize performance using regularization and learning rate adjustments.
- Evaluate the model using accuracy metrics.
- Visualize model training progress and performance.

## Methods

### Data Preprocessing:
- Normalized pixel values to range [0, 1].
- Reshaped input data to add a single grayscale channel.
- One-hot encoded the target labels.
- Visualized random samples for exploratory analysis.

### Model Development:
- Built a CNN using Keras Sequential API with the following architecture:
  - **Input**: Flattened 28x28 grayscale images.
  - **Hidden Layer**: Dense layer with 100 neurons and ReLU activation.
  - **Output Layer**: 10-neuron softmax layer for multi-class classification.
- Compiled with **Stochastic Gradient Descent (SGD)** optimizer (learning rate = 0.01, momentum = 0.9).
- Trained the model for **15 epochs** with batch size **64** and validation split of **10%**.

### Evaluation:
- Tracked training and validation accuracy.
- Visualized learning curves to assess model convergence.
- Evaluated final model performance on test data.

## Results

- Achieved **~97% validation accuracy**.
- Stable convergence of training and validation accuracy with minimal overfitting.
- Demonstrated the strong baseline performance of CNNs on handwritten digit recognition.
- Highlighted potential improvements, including deeper architectures and advanced regularization (e.g., Batch Normalization, Dropout).

## Business/Scientific Impact

- Demonstrated the capability of CNNs for digit recognition tasks, foundational in OCR (Optical Character Recognition) systems.
- Supports automation of tasks like postal code recognition, bank check processing, and digitized forms analysis.
- Provides a transferable architecture for similar grayscale image classification tasks.

## Technologies Used

- Python
- TensorFlow (Keras)
- NumPy
- Matplotlib
- Seaborn

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mnist-digit-classification-cnn.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to:
   - Preprocess the dataset.
   - Train the CNN model.
   - Evaluate model performance.
   - Visualize training progress and test predictions.

## Future Work

- Experiment with **deeper CNN architectures** for improved accuracy.
- Integrate **Batch Normalization** and **Dropout** layers to enhance generalization.
- Test alternative optimizers (e.g., Adam, RMSprop) and learning rate schedules.
- Perform **hyperparameter tuning** on network architecture and training parameters.
- Extend the project to similar datasets, such as **Fashion MNIST** or **Kuzushiji-MNIST**.
