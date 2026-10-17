import io
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import create_dataloader
from networks.freqnet import freqnet
from options.test_options import TestOptions
from validate import metrics_from_predictions


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def jpeg_transform(images, quality):
    device = images.device
    mean, std = MEAN.to(device), STD.to(device)
    rgb = (images * std + mean).clamp(0, 1)
    transformed = []
    for tensor in rgb:
        array = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as image:
            decoded = np.asarray(image.convert("RGB")).copy()
        tensor_out = torch.from_numpy(decoded).permute(2, 0, 1).float().to(device) / 255.0
        transformed.append((tensor_out - mean[0]) / std[0])
        buffer.close()
    return torch.stack(transformed, dim=0)


def resize_transform(images, scale=0.8):
    height, width = images.shape[-2:]
    smaller = F.interpolate(images, size=(max(8, int(height * scale)), max(8, int(width * scale))),
                            mode="bilinear", align_corners=False)
    return F.interpolate(smaller, size=(height, width), mode="bilinear", align_corners=False)


def main():
    opt = TestOptions().parse()
    if not opt.model_path:
        raise ValueError("Pass --model_path pointing to a trained checkpoint.")
    device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
    checkpoint = torch.load(opt.model_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = freqnet()
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()

    eval_opt = opt
    eval_opt.isTrain = False
    eval_opt.serial_batches = True
    eval_opt.no_flip = True
    loader = create_dataloader(eval_opt)
    if len(loader) == 0:
        raise RuntimeError("Evaluation loader is empty; check --dataroot and --classes.")

    transforms = {
        "clean": lambda x: x,
        "jpeg_q95": lambda x: jpeg_transform(x, 95),
        "jpeg_q75": lambda x: jpeg_transform(x, 75),
        "jpeg_q50": lambda x: jpeg_transform(x, 50),
        "resize_down_up": lambda x: resize_transform(x, 0.8),
    }
    all_results = {}
    for name, transform in transforms.items():
        labels_all, scores_all = [], []
        inference_seconds, image_count = 0.0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = transform(images.to(device, non_blocking=True))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                logits = model(images).sigmoid().flatten()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - start
                image_count += int(labels.numel())
                labels_all.extend(labels.flatten().cpu().tolist())
                scores_all.extend(logits.cpu().tolist())
        metrics = metrics_from_predictions(labels_all, scores_all)
        metrics["inference_ms_per_image"] = 1000.0 * inference_seconds / max(1, image_count)
        all_results[name] = metrics
        print(f"{name}: accuracy={metrics['accuracy']:.5f}, AP={metrics['average_precision']:.5f}, AUROC={metrics['auroc']:.5f}, F1={metrics['f1']:.5f}")

    all_results["_metadata"] = {
        "checkpoint": os.path.abspath(opt.model_path),
        "dataroot": os.path.abspath(opt.dataroot),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "note": "JPEG conditions use actual JPEG encode/decode. Resize robustness uses bilinear downsample/upscale. Evaluate on held-out data only.",
    }
    with open(opt.output_json, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, allow_nan=True)
    print(f"Wrote results to {os.path.abspath(opt.output_json)}")


if __name__ == "__main__":
    main()
