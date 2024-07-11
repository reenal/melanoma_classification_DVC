# model_pusher.py
'''
import boto3
import os
from logger import get_logger

logger = get_logger(__name__)

def push_model_to_s3(model_path, bucket_name, model_name):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(model_path, bucket_name, model_name)
        logger.info(f"Model pushed to S3: s3://{bucket_name}/{model_name}")
    except Exception as e:
        logger.error(f"Error in uploading model to S3: {e}")
'''