# data_utils.py
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import logging

def download_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images', path='./data', unzip=True)

def create_metadata():
    base_dir = 'data/melanoma/'
    train_dir = os.path.join(base_dir, 'train/')
    test_dir = os.path.join(base_dir, 'test/')

    train_metadata = []
    test_metadata = []
   
    for category in os.listdir(train_dir):
        if category == '.DS_Store':
            continue
        category_path = os.path.join(train_dir, category)
        for img_name in os.listdir(category_path):

            train_metadata.append([img_name, category])

    for category in os.listdir(test_dir):
        if category == '.DS_Store':
            continue
        category_path = os.path.join(test_dir, category)
        for img_name in os.listdir(category_path):
            test_metadata.append([img_name, category])

    train_df = pd.DataFrame(train_metadata, columns=['image_name', 'class'])
    test_df = pd.DataFrame(test_metadata, columns=['image_name', 'class'])

    train_df.to_csv(os.path.join(base_dir, 'train_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(base_dir, 'test_metadata.csv'), index=False)

def setup_data():
    #download_dataset()
    create_metadata()

if __name__ == '__main__':
    setup_data()
