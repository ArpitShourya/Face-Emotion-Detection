from src.preprocessing import PreprocessData
from src.train import TrainModel
from src.predict import PredictEmotion
import torch
from src.model import EmotionEfficientNetB0
if __name__=="__main__":
    preprocessing_obj=PreprocessData()
    train_dataset,train_loader,val_loader=preprocessing_obj.initiate_data_preprocessing()
    training_obj=TrainModel(train_dataset=train_dataset,train_loader=train_loader,val_loader=val_loader)
    trained_model=training_obj.initialize_training(50)
    trained_model = EmotionEfficientNetB0(num_classes=7)  
    trained_model.load_state_dict(torch.load("model/best.pt", map_location='cpu'))    
    predict_emotion_obj=PredictEmotion(trained_model)
    predict_emotion_obj.predict_emotion("Data/test/happy.jpg")