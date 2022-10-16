import boto3
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

class amazons3:
    def __init__(self):
        self.ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
        self.SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")
        self.conn = boto3.client('s3', aws_access_key_id=self.ACCESS_KEY_ID, aws_secret_access_key=self.SECRET_ACCESS_KEY)

    def upload_s3(self, df, bucket, path):
        df.to_csv(
            f"s3://{bucket}/{path}",
            index=False,
            storage_options={
                "key": self.ACCESS_KEY_ID,
                "secret": self.SECRET_ACCESS_KEY
            }
        )

    def download_s3(self, bucket, path):
        pd.read_csv(
            f"s3://{bucket}/{path}",
            index=False,
            storage_options={
                "key": self.ACCESS_KEY_ID,
                "secret": self.SECRET_ACCESS_KEY
            }
        )

    def upload_model_s3(self, filename, bucket, path):
        client = self.conn
        client.upload_file(filename, bucket, path)

    def download_model_s3(self, bucket, path, filename):
        client = self.conn
        client.download_file(bucket, path, filename)