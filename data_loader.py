# data_loader.py
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class MelanomaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,self.data_frame.iloc[idx, 1], self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(train_csv, test_csv, train_dir, test_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = MelanomaDataset(csv_file=train_csv, root_dir=train_dir, transform=transform)
    test_dataset = MelanomaDataset(csv_file=test_csv, root_dir=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
