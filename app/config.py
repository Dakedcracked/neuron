import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Resolve and load .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

REQUIRED_VARS = [
    "DATABASE_URL",
    "JWT_SECRET",
    "NEURON_CLINIC_SALT",
    "NEURON_ADMIN_USERNAME",
    "NEURON_ADMIN_PASSWORD",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_BUCKET_NAME",
    "S3_ENDPOINT_URL"
]

missing_vars = [var for var in REQUIRED_VARS if not os.environ.get(var)]

if missing_vars:
    print(f"❌ FATAL CONFIGURATION ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print(f"Please populate them in the environment or in the .env file at {env_path}.")
    sys.exit(1)

# Centralized clean configuration properties
DATABASE_URL = os.environ["DATABASE_URL"]
JWT_SECRET = os.environ["JWT_SECRET"]
NEURON_CLINIC_SALT = os.environ["NEURON_CLINIC_SALT"]
NEURON_ADMIN_USERNAME = os.environ["NEURON_ADMIN_USERNAME"]
NEURON_ADMIN_PASSWORD = os.environ["NEURON_ADMIN_PASSWORD"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_ENDPOINT_URL = os.environ["S3_ENDPOINT_URL"]
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_PUBLIC_URL_PREFIX = os.environ.get("S3_PUBLIC_URL_PREFIX", f"https://{S3_BUCKET_NAME}.s3.amazonaws.com")
print("✓ Enterprise Configuration properties loaded successfully.")
