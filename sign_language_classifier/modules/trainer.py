from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from sign_language_classifier.modules.model import ConvClassifier


class SignLanguageCNNModule(pl.LightningModule):
    """PyTorch Lightning модуль для классификации языка жестов с использованием CNN.

    Этот модуль управляет циклами обучения, валидации и тестирования для модели
    распознавания языка жестов. Включает отслеживание метрик и возможности визуализации.

    Attributes:
        model (ConvClassifier): CNN модель для классификации.
        criterion (nn.CrossEntropyLoss): Функция потерь для обучения.
        lr (float): Скорость обучения для оптимизации.
        num_classes (int): Количество классов языка жестов.
        train_accuracy (torchmetrics.Accuracy): Метрика точности для обучения.
        train_f1 (torchmetrics.F1Score): Метрика F1 для обучения.
        test_accuracy (torchmetrics.Accuracy): Метрика точности для тестирования.
        test_f1 (torchmetrics.F1Score): Метрика F1 для тестирования.
        val_accuracy (torchmetrics.Accuracy): Метрика точности для валидации.
        val_f1 (torchmetrics.F1Score): Метрика F1 для валидации.
        train_losses (list): История значений функции потерь при обучении.
        train_accuracies (list): История значений точности при обучении.
        train_f1scores (list): История значений F1 при обучении.
        val_losses (list): История значений функции потерь при валидации.
        val_accuracies (list): История значений точности при валидации.
        val_f1scores (list): История значений F1 при валидации.
    """

    def __init__(self, lr: float = 1e-3, num_classes: int = 25, in_channels: int = 1):
        """Инициализирует SignLanguageCNNModule.

        Args:
            lr (float, optional): Скорость обучения для оптимизатора. По умолчанию 1e-3.
            num_classes (int, optional): Количество классов для классификации. По умолчанию 25.
            in_channels (int, optional): Количество входных каналов изображения. По умолчанию 1.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = ConvClassifier(num_classes, in_channels)
        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr
        self.num_classes = num_classes

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )

        self.train_losses = []
        self.train_accuracies = []
        self.train_f1scores = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1scores = []

    def forward(self, inputs):
        """Прямой проход через модель.

        Args:
            inputs (torch.Tensor): Входной тензор с изображениями.

        Returns:
            torch.Tensor: Выходные логиты модели.
        """
        return self.model(inputs)

    def visualize(self, data, title, ylabel, filename, color="blue"):
        """Визуализирует и сохраняет график метрик.

        Args:
            data (list): Список значений для визуализации.
            title (str): Заголовок графика.
            ylabel (str): Подпись оси Y.
            filename (str): Имя файла для сохранения.
            color (str, optional): Цвет линии графика. По умолчанию "blue".

        Note:
            Графики сохраняются в директорию 'plots/'.
        """
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)

        plt.figure()
        plt.plot(data, label=title, color=color)
        plt.xlabel("Step" if "train" in filename else "Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / filename)
        plt.close()

    def training_step(self, batch: Any, batch_idx: int):
        """Выполняет один шаг обучения.

        Args:
            batch (Any): Батч данных, содержащий изображения и метки.
            batch_idx (int): Индекс текущего батча.

        Returns:
            torch.Tensor: Значение функции потерь для текущего шага.

        Note:
            Логирует: train_loss, train_f1_score, train_accuracy
        """
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)

        preds = torch.softmax(outputs, dim=1)
        self.train_f1(preds, labels)
        self.train_accuracy(preds, labels)

        self.train_losses.append(loss.item())
        self.train_accuracies.append(self.train_accuracy.compute().item())
        self.train_f1scores.append(self.train_f1.compute().item())

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_f1_score",
            self.train_f1,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "train_accuracy",
            self.train_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: Any):
        """Выполняет один шаг валидации.

        Args:
            batch (Any): Батч данных, содержащий изображения и метки.

        Note:
            Логирует: val_loss, val_accuracy, val_f1
        """
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)

        preds = torch.softmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)

        self.val_losses.append(loss.item())
        self.val_accuracies.append(self.val_accuracy.compute().item())
        self.val_f1scores.append(self.val_f1.compute().item())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_f1",
            self.val_f1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def on_train_end(self):
        """Вызывается в конце обучения для визуализации метрик обучения.

        Создает и сохраняет графики:
        - train_loss.png: График функции потерь при обучении
        - train_accuracy.png: График точности при обучении
        - train_f1.png: График F1 метрики при обучении
        """
        self.visualize(self.train_losses, "Training Loss", "Loss", "train_loss.png", color="blue")
        self.visualize(
            self.train_accuracies,
            "Training Accuracy",
            "Accuracy",
            "train_accuracy.png",
            color="green",
        )
        self.visualize(
            self.train_f1scores,
            "Training F1 Score",
            "F1 Score",
            "train_f1.png",
            color="orange",
        )

    def on_validation_epoch_end(self):
        """Вызывается в конце каждой эпохи валидации для визуализации метрик.

        Создает и сохраняет графики:
        - val_loss.png: График функции потерь при валидации
        - val_accuracy.png: График точности при валидации
        - val_f1.png: График F1 метрики при валидации
        """
        self.visualize(self.val_losses, "Validation Loss", "Loss", "val_loss.png", color="red")
        self.visualize(
            self.val_accuracies,
            "Validation Accuracy",
            "Accuracy",
            "val_accuracy.png",
            color="blue",
        )
        self.visualize(
            self.val_f1scores,
            "Validation F1 Score",
            "F1 Score",
            "val_f1.png",
            color="purple",
        )

    def test_step(self, batch: Any, batch_idx: int):
        """Выполняет один шаг тестирования.

        Args:
            batch (Any): Батч данных, содержащий изображения и метки.
            batch_idx (int): Индекс текущего батча.

        Note:
            Логирует: test_loss, test_accuracy, test_f1
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = torch.softmax(outputs, dim=1)

        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log(
            "test_accuracy",
            self.test_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "test_f1",
            self.test_f1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        """Настраивает оптимизатор для обучения.

        Returns:
            torch.optim.Optimizer: Оптимизатор Adam с заданной скоростью обучения.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
