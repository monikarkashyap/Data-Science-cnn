import torch
from model.model import MNIST_CNN
from PIL import Image
import torchvision.transforms as transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path="mnist_cnn.pth"

model=MNIST_CNN()
model.load_state_dict(torch.load(model_path,map_location=device))
model.to(device)
model.eval()

#data preprocessing
transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

def predict(image_path):
    image=Image.open(image_path)
    image=transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output=model(image)
        print(output)
        _,predicted=torch.max(output,1)
        return predicted.item()
    
#print("Predicted class: ",predict('converted_sample_digit.png'))   
print("Predicted class: ",predict('sample_digit_4.png'))


