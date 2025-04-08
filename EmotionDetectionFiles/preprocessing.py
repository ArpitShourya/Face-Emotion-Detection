import os
import numpy as np
from PIL import Image
import albumentations as A
import csv
from sklearn.preprocessing import LabelEncoder



class PreprocessData:
    def __init__(self):
        self.NUM_AUGMENTS = 5
        self.source_train_dir = r"E:\EmotionDetection\Data\images\train"
        self.save_train_dir = r"E:\EmotionDetection\Data\processed\train"
        self.csv_path = os.path.join(self.save_train_dir, "data.csv")
        self.augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
    ], additional_targets={'image': 'image'})

    def preprocess(self):
        #Create scaled Numpy Arrays
        for emotion in os.listdir(self.source_train_dir):
            emotion_folder = os.path.join(self.source_train_dir, emotion)
            save_emotion_folder = os.path.join(self.save_train_dir, emotion)

            if os.path.isdir(emotion_folder):
                os.makedirs(save_emotion_folder, exist_ok=True)  # Create target folder

                for file in os.listdir(emotion_folder):
                    file_path = os.path.join(emotion_folder, file)
                    file_name, _ = os.path.splitext(file)
                    save_path = os.path.join(save_emotion_folder, file_name + ".npy")

                    try:
                        with Image.open(file_path) as img:
                            img = img.convert('L')  # Keep grayscale
                            pixels = np.array(img, dtype=np.float32) / 255.0  # Normalize
                            np.save(save_path, pixels)  # Save as .npy file
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        # Augmentation
        for emotion in os.listdir(self.save_train_dir):
            emotion_folder = os.path.join(self.save_train_dir, emotion)
            if os.path.isdir(emotion_folder):
                for file in os.listdir(emotion_folder):
                    if not file.endswith(".npy"):
                        continue
                    file_path = os.path.join(emotion_folder, file)
                    base_name = os.path.splitext(file)[0]

                    try:
                        img = np.load(file_path)  # Shape: (H, W), float32 in [0.0, 1.0]

                        # Scale to 0–255 for albumentations, then back later
                        img_scaled = (img * 255).astype(np.uint8)

                        for i in range(1, self.NUM_AUGMENTS + 1):
                            augmented = self.augment(image=img_scaled)["image"]

                            # Normalize and save
                            aug_normalized = augmented.astype(np.float32) / 255.0
                            aug_save_path = os.path.join(emotion_folder, f"{base_name}_aug{i}.npy")
                            np.save(aug_save_path, aug_normalized)

                        print(f"Augmented {file} → {self.NUM_AUGMENTS} copies created.")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        #creating csv data
        emotions = sorted([d for d in os.listdir(self.save_train_dir) if os.path.isdir(os.path.join(self.save_train_dir, d))])
        label_encoder = LabelEncoder()
        label_encoder.fit(emotions)

        # Optional: print label mapping
        label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print("Label Mapping:", label_map)

        # Step 3: Write CSV
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['path', 'label'])  # header

            for emotion in emotions:
                emotion_dir = os.path.join(self.save_train_dir, emotion)
                label = label_encoder.transform([emotion])[0]

                for filename in os.listdir(emotion_dir):
                    if filename.endswith(".npy"):
                        rel_path = os.path.join(emotion, filename)
                        writer.writerow([rel_path, label])

        print(f"CSV saved at: {self.csv_path}")


if __name__=="__main__":
    pd=PreprocessData()
    pd.preprocess()

        


    
