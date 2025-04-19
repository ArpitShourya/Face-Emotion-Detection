from torchvision import transforms, datasets
from torch.utils.data import DataLoader



class PreprocessData:
    def __init__(self):
        self.train_dir = r"E:\EmotionDetection\Data\images\train"
        self.val_dir = r"E:\EmotionDetection\Data\images\validation"

        
    def preprocess_train(self):
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  
                                [0.229, 0.224, 0.225])
        ])
        return train_transforms

    def preprocess_val(self):
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return val_transforms
    
    def initiate_data_preprocessing(self):
        train_transforms=self.preprocess_train()
        val_transforms=self.preprocess_val()

        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(root=self.val_dir, transform=val_transforms)


        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_dataset,train_loader,val_loader
