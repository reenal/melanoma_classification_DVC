# main.py
import os
from data_utils import setup_data
from data_loader import get_dataloaders
from model import train_model, evaluate_model
#from model_pusher import push_model_to_s3
import mlflow
import torch

if __name__ == '__main__':
    # Data setup
    setup_data()

    # Directories and files
    print("data downlaoding and creating metadata started")
    base_dir = 'data/melanoma_cancer_dataset/'
    train_csv = os.path.join(base_dir, 'train_metadata.csv')
    test_csv = os.path.join(base_dir, 'test_metadata.csv')
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    print("data downlaoding and creating metadata finished")

    print("creating train test dataloader started")
    # Dataloaders
    train_loader, test_loader = get_dataloaders(train_csv, test_csv, train_dir, test_dir, batch_size=32)
    print("creating train test dataloader finished")

    print("Training model stated")
    # Train model
    model = train_model(train_loader, test_loader, num_epochs=1)
    print("Training model finished")

    print("Evaluate model stated")
    # Evaluate model
    evaluate_model(model, test_loader)
    print("Evaluate model finished")
  
    # Save model
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)

    # Push model to S3
    #bucket_name = "your-s3-bucket-name"
    #push_model_to_s3(model_path, bucket_name, model_path)
    print("all done")