import torch
from torchvision import transforms
from PIL import Image
import cv2
from NET12 import Net

# Load the model state_dict
try:
    model_state_dict = torch.load("model.pt", map_location=torch.device("cpu"))
except FileNotFoundError:
    print("Error: 'model.pt' not found in the current directory.")
    exit()

model = Net()

# Load the state_dict into the model
model.load_state_dict(model_state_dict)
model.eval()

# Initialize the face detector using Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize the camera capture (0 corresponds to the default camera in the machine)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame (ret shows if the frame opened successfully)
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for x, y, w, h in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) - the face
        face_roi = frame[y : y + h, x : x + w]

        # Convert the face ROI to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        transform = transforms.Compose(
            [transforms.Resize((48, 48)), transforms.ToTensor()]
        )
        input_tensor = transform(pil_image).unsqueeze(0)  # add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)

        # Get the predicted class
        predicted_class = torch.argmax(output).item()
        class_name = ["happy", "neutral", "sad"][predicted_class]

        # Display the result on the frame
        cv2.putText(
            frame,
            f"Predicted: {class_name}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Display the resulting frame
    cv2.imshow("Live Camera with Face Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
