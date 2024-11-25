
# CNN Image Classifier (MNIST Dataset)
This project is a simple Convolutional Neural Network (CNN) built using PyTorch for image classification on the MNIST dataset, a well-known dataset containing handwritten digits (0-9). The model is trained to classify these digits with high accuracy and deployed as a web service using Flask.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Training the Model](#training-the-model)
- [Web App Deployment](#web-app-deployment)
- [Usage](#usage)

## Installation
To get started with this project, follow these steps:

1. Clone the repository
```bash
    git clone https://github.com/monikarkashyap/Data-Science-cnn.git
```

2. Create and activate a virtual environment
It's a good practice to use a virtual environment:

```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

3. Install the required dependencies
You can install all the required packages using requirements.txt:

```bash
    pip install -r requirements.txt
```

4. Download the MNIST dataset
The MNIST dataset will be downloaded automatically during the training process if it doesn't already exist in the ./data directory.

## Project Structure
```plaintext
cnn/
│
├── app/
│   ├── inference.py           # Script to load the trained model and perform inference
│   ├── app.py                 # Main script to run the Flask web server
├── model/
│   ├── config.py              # Configuration file containing model settings
│   ├── model.py                # Model architecture definition (CNN)
│   ├── train.py                # functions to train the model
│   ├── main.py                # Script to train the model
├── templates/
│   └── index.html              # HTML template for the web interface
│
├── .gitignore                  # Git ignore configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview and instructions
├── mnist_cnn.pth               # Trained model file
```

## Model Details
The CNN model used for this project has the following architecture:

- Input Layer: 28x28 grayscale image (MNIST digits).
- Convolutional Layer 1: 16 filters, kernel size of 3x3, stride 1, padding 1.
- Convolutional Layer 2: 32 filters, kernel size of 3x3, stride 1, padding 1.
- Pooling Layer: Max pooling with a 2x2 kernel.
- Fully Connected Layer 1: 128 neurons.
- Fully Connected Layer 2: 10 output neurons (one for each digit class).
- Dropout: Dropout rate of 50% to avoid overfitting.

## Training Parameters:
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Epochs: 10
- Batch Size: 64
- Training the Model

To train the model on the MNIST dataset, run the main.py script.
```bash
    python model/main.py
```
This will train the model and save the trained weights as mnist_cnn.pth file.

## Web App Deployment
## Requirements
The web app is built using Flask, which serves the trained CNN model to predict handwritten digits.


## Running the Flask Server
To start the Flask web server for model inference, run:

```bash
    python -m app.app
```

The server will start, and you can open your browser and go to http://127.0.0.1:5000 to access the web app.

## Image Prediction
The web interface allows you to upload an image and get predictions for the handwritten digit. The uploaded image is processed and passed to the trained CNN model for classification.

## API Endpoint:
- Endpoint: /predict
- Method: POST
- Input: A file upload (image file in PNG, JPG, or JPEG format).
- Output: Predicted class of the digit.
Example request using Postman or any API client:
- POST to http://127.0.0.1:5000/predict
- Attach the image under the key image.

## Sample Input:
Upload any image containing a handwritten digit (28x28 pixels, grayscale).

## Sample Output:
```json
    {
        "predicted_class": 5
    }
```
## Usage
## Model Inference in Python
If you want to use the trained model for inference in Python, you can run it as module using -m flag:

```bash
    python3 -m app.inference  
```

Simply provide the image path of a digit to get the prediction. You can use any 28x28 grayscale image of a handwritten digit.

## Example Usage:
```python
    predicted_class = predict('path_to_image.png')
    print(f"Predicted class: {predicted_class}")
```
