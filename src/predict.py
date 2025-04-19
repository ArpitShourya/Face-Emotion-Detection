import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch
from src.model import EmotionEfficientNetB0
from src.preprocessing import PreprocessData
import matplotlib.pyplot as plt

class PredictEmotion:
    def __init__(self,model):
        self.model=model
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.transform=PreprocessData().preprocess_val()

    def predict_emotion(self,img_path):
        self.img_path=img_path
        self.model.eval()
        img_bgr = cv2.imread(self.img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = img_rgb[y:y+h, x:x+w]
            face_pil = Image.fromarray(face).convert("RGB")
            input_tensor = self.transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                output = self.model(input_tensor)
                pred = torch.argmax(output, 1).item()
                emotion = self.class_names[pred]

            # Draw bounding box and label
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_bgr, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        2.2, (255, 0, 0), 4)

        # Show result


        # BGR to RGB for matplotlib
        img_rgb_disp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb_disp)
        plt.title("Emotion Detection")
        plt.axis("off")
        plt.show()



