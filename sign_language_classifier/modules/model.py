import torch


class ConvClassifier(torch.nn.Module):
    """Сверточная нейронная сеть для классификации изображений языка жестов."""

    def __init__(self, num_classes: int = 25, in_channels: int = 1):
        """Инициализирует ConvClassifier.

        Args:
            num_classes (int, optional): Количество классов для классификации.
                По умолчанию 25 (для алфавита языка жестов).
            in_channels (int, optional): Количество входных каналов изображения.
                По умолчанию 1 (черно-белые изображения).
        """
        super().__init__()

        self.model = torch.nn.Sequential(
            # Conv1: (in_channels, 28, 28) -> (32, 12, 12) после Conv2d(5x5) и MaxPool2d(2)
            torch.nn.Conv2d(in_channels, 32, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            # Conv2: (32, 12, 12) -> (64, 4, 4) после Conv2d(5x5) и MaxPool2d(2)
            torch.nn.Conv2d(32, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            # Conv3: (64, 4, 4) -> (128, 1, 1) после Conv2d(3x3) и MaxPool2d(2)
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            # Conv4: (128, 1, 1) -> (256, -) Требуется вход большего размера
            # Проблема: после предыдущих слоев размер становится слишком маленьким
            # Рекомендуется увеличить входное изображение или изменить архитектуру
            torch.nn.Conv2d(128, 256, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            # Dropout для регуляризации
            torch.nn.Dropout(0.1),
            # Conv5
            torch.nn.Conv2d(256, 512, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            # Классификатор
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 4 * 4, 256),  # Размер зависит от входного изображения
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через модель.

        Args:
            x (torch.Tensor): Входной тензор изображений.
                Форма: (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Выходные логиты модели.
                Форма: (batch_size, num_classes)

        Note:
            Ожидаемый размер входного изображения: 28x28 пикселей.
        """
        return self.model(x)
