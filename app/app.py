import torch
from PIL import Image
from model.model import MNIST_CNN
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app=Flask(__name__,template_folder='../templates')
@app.route('/')
def index():
    return render_template('index.html')
    

# Load the trained model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=MNIST_CNN()
model.load_state_dict(torch.load("mnist_cnn.pth",map_location=device))
model.to(device)
model.eval()

# Define data preprocessing steps
transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize((28,28)),
    transforms.Normalize((0.5,),(0.5,))
])

# Define the prediction function
def predict(image):
    image=transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output=model(image)
        _,predicted=torch.max(output,1)
        return predicted.item()
    
@app.route('/predict',methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file=request.files['image']
    if file.filename=='':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Open the image
        image=Image.open(file.stream)
        predicted_class=predict(image)
        return render_template('index.html', predicted_class=predicted_class)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
