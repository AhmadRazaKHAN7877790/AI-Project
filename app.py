from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import cv2
from NET12 import Net  # Update with the correct import path
from camera import detect_emotion


app = Flask(__name__)

# Load the pre-trained model
try:
    model_state_dict = torch.load("model.pt", map_location=torch.device("cpu"))
except FileNotFoundError:
    print("Error: 'model.pt' not found in the current directory.")
    exit()

model = Net()
model.load_state_dict(model_state_dict)
model.eval()

# Initialize the face detector using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define emotion class names
EMOTION_CLASSES = ['happy', 'neutral', 'sad']

def predict_emotion(face_roi):
    # Convert the face ROI to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])
    input_tensor = transform(pil_image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output).item()
        emotion = EMOTION_CLASSES[predicted_class]

    return emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']

    # Preprocess the image
    image = Image.open(file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output).item()
        emotion = EMOTION_CLASSES[predicted_class]

    # Return the prediction as JSON
    return jsonify({'prediction': emotion})

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    # Call the detect_emotion function
    emotion = detect_emotion()

    # # Return the emotion prediction as JSON
    # return jsonify({'emotion': emotion})
if __name__ == '__main__':
    # Initialize the camera capture (0 corresponds to the default camera in the machine)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    app.run(debug=True)
