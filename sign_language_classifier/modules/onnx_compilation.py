from pathlib import Path

import fire
import numpy as np
import onnxruntime
import torch
from trainer import SignLanguageCNNModule


def main(
    checkpoint_path: str,
    onnx_output_path: str = "./models/model.onnx",
    img_size: int = 224,
    batch_size: int = 1,
) -> None:
    """
    Преобразует чекпоинт .ckpt модели в ONNX.

    Args:
        checkpoint_path: путь до чекпоинта модели (.ckpt)
        onnx_output_path: куда сохранять ONNX
        img_size: размер изображения (default: 224)
        batch_size: размер батча для ONNX (default: 1)
    """
    module = SignLanguageCNNModule.load_from_checkpoint(checkpoint_path)
    print("Модель загружена")
    module.eval()

    Path(onnx_output_path).parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(batch_size, 1, img_size, img_size)

    module.to_onnx(
        onnx_output_path,
        dummy_input,
        export_params=True,
        input_names=["INPUT_IMAGE"],
        output_names=["CLASS_LOGITS"],
        dynamic_axes={
            "INPUT_IMAGE": {0: "batch_size"},
            "CLASS_LOGITS": {0: "batch_size"},
        },
        opset_version=11,
    )

    print(f"Модель преобразована в ONNX и лежит тут: {onnx_output_path}")

    ort_session = onnxruntime.InferenceSession(onnx_output_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.numpy().astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)

    if ort_outs:
        print("Все ок!")
        print(f"   Output shape: {ort_outs[0].shape}")
        print(f"   Sample output: {ort_outs[0][0][:5]} ...")
    else:
        print("Что-то пошло не так :(")


if __name__ == "__main__":
    fire.Fire(main)
