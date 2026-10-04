"""Hugging Face dataset accessors, including a parquet fallback for Datasets 5.x."""
from datasets import load_dataset

from .config import settings

def load_split(split: str):
    token=settings.huggingface_api_key or None
    if settings.dataset_name=="microsoft/ms_marco" and settings.dataset_config=="v1.1":
        file_url=f"https://huggingface.co/datasets/{settings.dataset_name}/resolve/main/v1.1/{split}-00000-of-00001.parquet"
        return load_dataset("parquet",data_files={split:file_url},split=split,streaming=True,token=token)
    return load_dataset(settings.dataset_name,settings.dataset_config,split=split,streaming=True,token=token)
