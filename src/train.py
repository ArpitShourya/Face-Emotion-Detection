import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from src.model import EmotionEfficientNetB0


    
class TrainModel:
    def __init__(self,train_dataset,train_loader,val_loader):
        self.train_dataset=train_dataset
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionEfficientNetB0(num_classes=7).to(self.device)
        self.targets = [label for _, label in self.train_dataset]
        self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.targets), y=self.targets)
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=25)
        

        
    def train_one_epoch(self,loader):
        self.model.train()
        total_loss, total_correct = 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = total_correct / len(loader.dataset)
        return total_loss / len(loader), accuracy
    
    def validate_one_epoch(self,loader):
        self.model.eval()  
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  
            for images, labels in loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / total
        accuracy = correct / total

        return avg_loss, accuracy
    
    def unfreeze_mobilenet(self):
        for param in self.model.base_model.features[5:].parameters():  
            param.requires_grad = True

    
    def initialize_training(self,num_epochs:int):
        best_acc = 0
        history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
        patience=3
        patience_counter=0
        for epoch in range(1, num_epochs+1):
            if epoch == 10:  
                self.unfreeze_mobilenet(self.model)
                print("✅ Unfroze last few layers of Efficientnet")
            train_loss, train_acc = self.train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion)
            val_loss, val_acc = self.validate_one_epoch(self.model, self.val_loader, self.criterion)
            self.scheduler.step(val_loss)



            print(f"Epoch {epoch} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                os.makedirs("model", exist_ok=True)
                torch.save(self.model.state_dict(), "model/best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        trained_model = EmotionEfficientNetB0(num_classes=7)  
        trained_model.load_state_dict(torch.load("model/best.pt", map_location='cpu'))
        return trained_model
                    


