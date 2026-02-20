import os
import zipfile

import gdown
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Загружает и распаковывает датасет с Google Drive."""
    gdrive_url = config["data_load"]["url"]
    output_dir = config["data_load"]["data_path"]
    zip_filename = config["data_load"]["data_zip_name"]

    os.makedirs(output_dir, exist_ok=True)

    print("Загрузка ZIP-файла...")
    gdown.download(gdrive_url, zip_filename, quiet=False, fuzzy=True)

    print("Распаковка ZIP-файла...")
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("Удаление ZIP-файла...")
    os.remove(zip_filename)
    print(f"Распакованные данные сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
