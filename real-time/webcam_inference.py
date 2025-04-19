import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models

class EmotionEfficientNetB0(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionEfficientNetB0, self).__init__()

        # Load pretrained EfficientNet-B0
        self.base_model = models.efficientnet_b0(pretrained=True)

        

        # Freeze all feature extractor layers initially
        for param in self.base_model.features.parameters():
            param.requires_grad = False

        # Replace the classifier with a custom head
        self.base_model.classifier = nn.Identity()

        self.custom_classifier = nn.Sequential(
            nn.Linear(1280, 128),  # 1280 is the EfficientNet-B0 output feature size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # Output logits
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.custom_classifier(x)
        return x


# Load model
model = EmotionEfficientNetB0(num_classes=7)
model.load_state_dict(torch.load("model/best.pt", map_location=torch.device('cpu')))
model.eval()

# Emotion labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Set up matplotlib
plt.ion()
fig, ax = plt.subplots()

# Capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(face).convert("RGB")
        input_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = class_names[pred]

        # Draw bounding box and label
        cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(orig, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title("Real-Time Emotion Detection")
    ax.axis("off")
    plt.pause(0.01)
    ax.clear()

try:
    cap.release()
    plt.ioff()
    plt.show()
except:
    pass
