from typing import Any, Optional, Tuple

import cv2
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class SignLanguageMNISTDataset(Dataset):
    """Датасет для распознавания языка жестов на основе MNIST.

    Этот класс загружает данные из CSV файла, преобразует их в изображения
    и подготавливает для использования в PyTorch.

    Attributes:
        csv (pd.DataFrame): Загруженные данные из CSV файла.
        img_size (int): Размер изображения после ресайза.
        train (bool): Флаг, указывающий, используется ли датасет для обучения.
        images (torch.Tensor): Тензор с изображениями.
        labels (pd.Series): Метки классов.
    """

    def __init__(self, csv_path: str, img_size: int = 224, train: bool = True):
        """Инициализирует SignLanguageMNISTDataset.

        Args:
            csv_path (str): Путь к CSV файлу с данными.
            img_size (int, optional): Размер изображения после ресайза. По умолчанию 224.
            train (bool, optional): Флаг обучения (True для обучающей выборки). По умолчанию True.
        """
        self.csv = pd.read_csv(csv_path)
        self.img_size = img_size
        self.train = train

        text = "pixel"
        self.images = torch.zeros((self.csv.shape[0], 1))

        for i in range(1, 785):
            temp_text = text + str(i)
            temp = self.csv[temp_text]
            temp = torch.FloatTensor(temp).unsqueeze(1)
            self.images = torch.cat((self.images, temp), 1)

        self.labels = self.csv["label"]
        self.images = self.images[:, 1:]
        self.images = self.images.view(-1, 28, 28)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """Возвращает элемент датасета по индексу.

        Args:
            index (int): Индекс элемента.

        Returns:
            Tuple[torch.Tensor, Any]: Кортеж (изображение, метка) для обучения,
                или только изображение для предсказания.
        """
        img = self.images[index]
        img = img.numpy()
        img = cv2.resize(img, (self.img_size, self.img_size))

        tensor_image = torch.FloatTensor(img)
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image /= 255.0

        if self.train:
            return tensor_image, self.labels[index]
        return tensor_image

    def __len__(self) -> int:
        """Возвращает размер датасета.

        Returns:
            int: Количество элементов в датасете.
        """
        return self.images.shape[0]


class SignLanguageMNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule для загрузки данных языка жестов.

    Этот класс управляет загрузкой, разделением и подготовкой данных
    для обучения, валидации, тестирования и предсказаний.

    Attributes:
        train_csv_path (Optional[str]): Путь к CSV файлу с обучающими данными.
        val_csv_path (Optional[str]): Путь к CSV файлу с валидационными данными.
        test_csv_path (Optional[str]): Путь к CSV файлу с тестовыми данными.
        predict_csv_path (Optional[str]): Путь к CSV файлу с данными для предсказания.
        img_size (int): Размер изображений.
        train_batch_size (int): Размер батча для обучения.
        predict_batch_size (int): Размер батча для предсказаний.
        num_workers (int): Количество процессов для загрузки данных.
        persistent_workers (bool): Флаг постоянных рабочих процессов.
        train_dataset (Optional[SignLanguageMNISTDataset]): Обучающий датасет.
        val_dataset (Optional[SignLanguageMNISTDataset]): Валидационный датасет.
        test_dataset (Optional[SignLanguageMNISTDataset]): Тестовый датасет.
        predict_dataset (Optional[SignLanguageMNISTDataset]): Датасет для предсказаний.
    """

    def __init__(
        self,
        train_csv_path: Optional[str] = None,
        val_csv_path: Optional[str] = None,
        test_csv_path: Optional[str] = None,
        predict_csv_path: Optional[str] = None,
        img_size: int = 224,
        train_batch_size: int = 128,
        predict_batch_size: int = 64,
        num_workers: int = 4,
        persistent_workers: bool = True,
    ):
        """Инициализирует SignLanguageMNISTDataModule.

        Args:
            train_csv_path (Optional[str]): Путь к CSV файлу с обучающими данными.
            val_csv_path (Optional[str]): Путь к CSV файлу с валидационными данными.
            test_csv_path (Optional[str]): Путь к CSV файлу с тестовыми данными.
            predict_csv_path (Optional[str]): Путь к CSV файлу с данными для предсказания.
            img_size (int, optional): Размер изображений. По умолчанию 224.
            train_batch_size (int, optional): Размер батча для обучения. По умолчанию 128.
            predict_batch_size (int, optional): Размер батча для предсказаний. По умолчанию 64.
            num_workers (int, optional): Количество процессов для загрузки данных. По умолчанию 4.
            persistent_workers (bool, optional): Флаг постоянных рабочих процессов. По умолч. True.
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.predict_csv_path = predict_csv_path
        self.num_workers = num_workers

        self.img_size = img_size
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.persistent_workers = persistent_workers

    def prepare_data(self):
        """Подготавливает данные (загрузка, предобработка).

        В данном случае данные уже загружены в CSV файлах.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Настраивает датасеты в зависимости от этапа.

        Args:
            stage (Optional[str]): Этап ('fit', 'validate', 'test', 'predict').
                Если None, настраивает все возможные датасеты.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = SignLanguageMNISTDataset(
                csv_path=self.train_csv_path, img_size=self.img_size, train=True
            )
            self.val_dataset = SignLanguageMNISTDataset(
                csv_path=self.val_csv_path, img_size=self.img_size, train=True
            )
        elif stage == "validate":
            self.val_dataset = SignLanguageMNISTDataset(
                csv_path=self.val_csv_path, img_size=self.img_size, train=True
            )
        elif stage == "test":
            self.test_dataset = SignLanguageMNISTDataset(
                csv_path=self.test_csv_path, img_size=self.img_size, train=True
            )
        elif stage == "predict" and self.predict_csv_path:
            self.predict_dataset = SignLanguageMNISTDataset(
                csv_path=self.predict_csv_path, img_size=self.img_size, train=False
            )

    def train_dataloader(self) -> DataLoader:
        """Создает DataLoader для обучающей выборки.

        Returns:
            DataLoader: DataLoader для обучения.
        """
        return DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            shuffle=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Создает DataLoader для валидационной выборки.

        Returns:
            DataLoader: DataLoader для валидации.
        """
        return DataLoader(
            dataset=self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.predict_batch_size,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Создает DataLoader для тестовой выборки.

        Returns:
            DataLoader: DataLoader для тестирования.
        """
        return DataLoader(
            dataset=self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.predict_batch_size,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Создает DataLoader для предсказаний.

        Returns:
            DataLoader: DataLoader для предсказаний.
        """
        return DataLoader(
            dataset=self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.predict_batch_size,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    @property
    def num_classes(self) -> int:
        """Возвращает количество классов.

        Returns:
            int: Количество классов (25 для языка жестов).
        """
        return 25

    @property
    def in_channels(self) -> int:
        """Возвращает количество входных каналов.

        Returns:
            int: Количество каналов (1 для черно-белых изображений).
        """
        return 1
