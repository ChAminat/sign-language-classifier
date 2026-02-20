import fire
import pytorch_lightning as pl
from modules.data import SignLanguageMNISTDataModule
from modules.trainer import SignLanguageCNNModule


def main(
    test_csv_path: str,
    checkpoint_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
) -> None:
    """
    Запускает инференс на тестовом датасете используя обученный чекпоинт.

    Args:
        test_csv_path: Путь к CSV файлу с тестовыми данными
        checkpoint_path: Путь к чекпоинту модели (.ckpt)
        batch_size: Размер батча для инференса
        num_workers: Количество рабочих процессов для загрузчика данных
        img_size: Размер изображения для ресайза
    """
    dm = SignLanguageMNISTDataModule(
        train_csv_path=None,
        val_csv_path=None,
        test_csv_path=test_csv_path,
        img_size=img_size,
        train_batch_size=batch_size,
        predict_batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )

    module = SignLanguageCNNModule.load_from_checkpoint(checkpoint_path)
    module.eval()

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
    )

    test_results = trainer.test(module, datamodule=dm)

    print("\n" + "=" * 50)
    print("Результаты тестирования:")
    print(f"  Точность: {test_results[0]['test_accuracy']:.4f}")
    print(f"  F1 мера: {test_results[0]['test_f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    fire.Fire(main)
