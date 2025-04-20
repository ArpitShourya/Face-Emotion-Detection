# Facial Emotion Recognition

A deep learning-based project to classify human facial expressions into seven distinct emotions using grayscale images. Trained on the FER2013 dataset with both a custom CNN and a modified MobileNetV2 architecture.

## 📁 Project Structure

```
Facial-Emotion-Recognition/
├── data/               # Contains image and labels 
├── models/             # Saved model checkpoints and architectures
├── src/                # Training, evaluation, and utility scripts
├── notebooks/          # Colab notebooks for training & experiments
└── README.md
```

## 🎯 Objective

To accurately detect emotions from facial images using deep learning techniques. The model classifies images into the following 7 emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## 🧠 Models Used

1. **EffecientNetB0**
2. **MobileNetV2**    

## 🧪 Performance

| Model         | Validation Accuracy |
|---------------|---------------------|
| EffecientNet  | ~64%                |
| MobileNetV2   | ~62%                |

> Trained using Google Colab with a free T4 GPU.

## 🧰 Tech Stack

- Python
- PyTorch
- NumPy, Pandas, Matplotlib
- Google Colab (T4 GPU)

## 🧪 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/facial-emotion-recognition.git
   cd facial-emotion-recognition
   ```

2. Install dependencies (via Colab or locally):
   ```bash
   pip install -r requirements.txt
   ```

3. Run training:
   ```bash
   python main.py        
   ```


## 📊 Sample Results
![image](https://github.com/user-attachments/assets/6fb1beaa-9225-483d-adf2-bdf0e73ea464)
![image](https://github.com/user-attachments/assets/3e8527fc-a1d2-4e42-a0db-c783454a9f62)
![image](https://github.com/user-attachments/assets/cc4cc2aa-7f5b-40d0-a12e-f0c2b4fb0cb0)
![image](https://github.com/user-attachments/assets/4c74d9dc-1bae-446f-a743-557a325f7d49)


## 🚀 Future Plans

- Add a Streamlit web app for real-time inference
- Convert to ONNX and deploy with FastAPI
- Train on larger or more diverse datasets
- Improve accuracy with ensemble models or attention mechanisms

## 🙌 Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- Google Colab for compute support
