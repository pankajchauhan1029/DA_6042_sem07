import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from validate import validate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path, trainer, epoch, metrics):
    torch.save({
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "total_steps": trainer.total_steps,
        "epoch": epoch,
        "metrics": metrics,
        "fcr_enabled": trainer.fcr_enabled,
        "fcr_lambda": trainer.fcr_lambda,
    }, path)


def main():
    opt = TrainOptions().parse()
    set_seed(opt.seed)
    if opt.niter < 1:
        raise ValueError("--niter must be at least 1")
    if not os.path.isdir(opt.dataroot):
        raise FileNotFoundError(f"Dataset root does not exist: {opt.dataroot}")

    root = os.path.abspath(opt.dataroot)
    train_opt = copy.deepcopy(opt)
    train_opt.dataroot = os.path.join(root, opt.train_split)
    train_opt.isTrain = True
    val_opt = copy.deepcopy(opt)
    val_opt.dataroot = os.path.join(root, opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    if not os.path.isdir(train_opt.dataroot):
        raise FileNotFoundError(f"Training split not found: {train_opt.dataroot}")
    if not os.path.isdir(val_opt.dataroot):
        raise FileNotFoundError(f"Validation split not found: {val_opt.dataroot}")

    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    run_dir = os.path.join(opt.checkpoints_dir, opt.name)
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "seed": opt.seed, "dataset_root": root, "train_split": opt.train_split,
            "val_split": opt.val_split, "classes": opt.classes, "batch_size": opt.batch_size,
            "niter": opt.niter, "lr": opt.lr, "fcr_enabled": opt.fcr_enabled,
            "fcr_lambda": opt.fcr_lambda, "gpu_ids": opt.gpu_ids,
        }, handle, indent=2)

    train_loader = create_dataloader(train_opt)
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Check dataset directory structure.")
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    trainer = Trainer(opt)
    best_ap = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(opt.niter):
        trainer.train()
        running_loss, batches = 0.0, 0
        for batch in train_loader:
            trainer.set_input(batch)
            loss_info = trainer.optimize_parameters()
            trainer.total_steps += 1
            running_loss += loss_info["loss"]
            batches += 1
            if trainer.total_steps % opt.loss_freq == 0:
                writer.add_scalar("train/loss", loss_info["loss"], trainer.total_steps)
                writer.add_scalar("train/fcr_kl", loss_info["loss_fcr"], trainer.total_steps)

        train_loss = running_loss / max(1, batches)
        acc, ap, real_acc, fake_acc, y_true, y_score = validate(trainer.model, val_opt)
        val_metrics = {
            "accuracy": float(acc), "average_precision": float(ap),
            "real_accuracy": float(real_acc), "fake_accuracy": float(fake_acc),
        }
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar("validation/" + key, value, epoch)
        print(f"epoch={epoch + 1}/{opt.niter} loss={train_loss:.5f} val_acc={acc:.5f} val_ap={ap:.5f}")

        save_checkpoint(os.path.join(run_dir, "model_epoch_last.pth"), trainer, epoch + 1, val_metrics)
        if ap > best_ap:
            best_ap = ap
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(os.path.join(run_dir, "model_epoch_best.pth"), trainer, epoch + 1, val_metrics)
        else:
            epochs_without_improvement += 1

        if opt.delr_freq > 0 and (epoch + 1) % opt.delr_freq == 0:
            trainer.adjust_learning_rate()
        if opt.patience > 0 and epochs_without_improvement >= opt.patience:
            print(f"Early stopping: validation AP did not improve for {opt.patience} epochs.")
            break

    writer.close()
    elapsed = time.time() - start
    summary = {
        "best_epoch": best_epoch, "best_validation_average_precision": best_ap,
        "elapsed_seconds": elapsed, "checkpoint": os.path.join(run_dir, "model_epoch_best.pth"),
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
