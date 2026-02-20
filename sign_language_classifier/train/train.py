import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.loggers import MLFlowLogger

from sign_language_classifier.modules.data import SignLanguageMNISTDataModule
from sign_language_classifier.modules.trainer import SignLanguageCNNModule


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Запускает обучение модели классификации языка жестов."""
    datamodule = SignLanguageMNISTDataModule(
        train_csv_path=config["data_load"]["train_csv_path"],
        val_csv_path=config["data_load"]["val_csv_path"],
        test_csv_path=config["data_load"]["test_csv_path"],
        img_size=config["data_load"]["img_size"],
        train_batch_size=config["training"]["train_batch_size"],
        predict_batch_size=config["training"]["predict_batch_size"],
        num_workers=config["training"]["num_workers"],
        persistent_workers=config["training"]["pers_workers"],
    )

    loggers = [
        MLFlowLogger(
            experiment_name=config["logging"]["experiment_name"],
            run_name=config["logging"]["run_name"],
            save_dir=config["logging"]["save_dir"],
            tracking_uri=config["logging"]["tracking_uri"],
        )
    ]

    module = SignLanguageCNNModule(
        lr=config["training"]["lr"],
        num_classes=config["model"]["num_classes"],
        in_channels=config["model"]["in_channels"],
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        DeviceStatsMonitor(),
        RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        ModelCheckpoint(
            dirpath=config["model"]["model_local_path"],
            filename="{epoch:02d}-{val_accuracy:.4f}",
            monitor="val_accuracy",
            mode="max",
            save_top_k=config["model"]["save_top_k"],
            every_n_epochs=config["model"]["every_n_epochs"],
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
