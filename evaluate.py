import json
import os
import torch

from data import create_dataloader
from networks.freqnet import freqnet
from options.test_options import TestOptions
from validate import metrics_from_predictions


def main():
    opt = TestOptions().parse()
    if not opt.model_path:
        raise ValueError("Pass --model_path to a trained checkpoint.")
    device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
    state = torch.load(opt.model_path, map_location="cpu", weights_only=False)
    weights = state.get("model", state) if isinstance(state, dict) else state
    model = freqnet()
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()

    opt.isTrain = False
    opt.serial_batches = True
    opt.no_flip = True
    loader = create_dataloader(opt)
    labels, scores = [], []
    with torch.no_grad():
        for images, targets in loader:
            probabilities = model(images.to(device)).sigmoid().flatten()
            labels.extend(targets.flatten().cpu().tolist())
            scores.extend(probabilities.cpu().tolist())
    metrics = metrics_from_predictions(labels, scores)
    metrics["parameter_count"] = sum(p.numel() for p in model.parameters())
    metrics["checkpoint"] = os.path.abspath(opt.model_path)
    with open(opt.output_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
