import os
from dotenv import dotenv_values
import gdown
import zipfile

config = dotenv_values(".env")


def model_downloaded(model_filename: str) -> bool:
    """Checks if model is downloaded"""
    if not os.path.exists(config["DOWNLOAD_PATH"]):
        return False
    if not os.path.exists(f"{config['DOWNLOAD_PATH']}/{model_filename}"):
        return False
    return True


def download_model(id: str, model_filename: str) -> None:
    if not os.path.exists(config["DOWNLOAD_PATH"]):
        os.mkdir(config["DOWNLOAD_PATH"])
    gdown.download(
        id=id,
        output=f"{config['DOWNLOAD_PATH']}/{model_filename}.zip",
        quiet=False,
    )
    with zipfile.ZipFile(
        file=f"{config['DOWNLOAD_PATH']}/{model_filename}.zip", mode="r"
    ) as zip_ref:
        zip_ref.extractall(path=f"{config['DOWNLOAD_PATH']}")
    os.remove(path=f"{config['DOWNLOAD_PATH']}/{model_filename}.zip")
