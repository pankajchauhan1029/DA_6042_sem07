import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.freqnet import freqnet
from networks.base_model import BaseModel


def make_label_preserving_view(x):
    """Randomly apply mild downsample/upscale or 3x3 average blur."""
    _, _, height, width = x.shape
    if torch.rand((), device=x.device).item() < 0.5:
        scale = torch.empty((), device=x.device).uniform_(0.82, 0.95).item()
        small_h = max(8, int(height * scale))
        small_w = max(8, int(width * scale))
        view = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
        return F.interpolate(view, size=(height, width), mode="bilinear", align_corners=False)
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)


def bernoulli_kl_from_logits(logits_p, logits_q, eps=1e-6):
    """KL between Bernoulli probabilities induced by two binary logits."""
    p = torch.sigmoid(logits_p).clamp(eps, 1.0 - eps)
    q = torch.sigmoid(logits_q).clamp(eps, 1.0 - eps)
    return (p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))).mean()


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super().__init__(opt)
        self.model = freqnet()
        self.fcr_enabled = bool(getattr(opt, "fcr_enabled", False))
        self.fcr_lambda = float(getattr(opt, "fcr_lambda", 0.05))
        if self.fcr_lambda < 0:
            raise ValueError("--fcr_lambda must be non-negative")

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            trainable = filter(lambda p: p.requires_grad, self.model.parameters())
            if opt.optim == "adam":
                self.optimizer = torch.optim.Adam(trainable, lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == "sgd":
                self.optimizer = torch.optim.SGD(trainable, lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [sgd, adam]")

        self.model.to(self.device)
        if not self.isTrain or getattr(opt, "continue_train", False):
            self.load_networks(opt.epoch)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= 0.8
            if param_group["lr"] < min_lr:
                return False
        self.lr = self.optimizer.param_groups[0]["lr"]
        print(f"Learning rate reduced to {self.lr}")
        return True

    def set_input(self, batch):
        self.input = batch[0].to(self.device, non_blocking=True)
        self.label = batch[1].to(self.device, non_blocking=True).float().view(-1)

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.forward()
        logits = self.output.squeeze(1)
        original_loss = self.loss_fn(logits, self.label)

        if self.fcr_enabled:
            transformed = make_label_preserving_view(self.input)
            transformed_logits = self.model(transformed).squeeze(1)
            transformed_loss = self.loss_fn(transformed_logits, self.label)
            consistency_loss = bernoulli_kl_from_logits(logits, transformed_logits)
            self.loss = 0.5 * (original_loss + transformed_loss) + self.fcr_lambda * consistency_loss
            self.loss_fcr = consistency_loss.detach().item()
        else:
            self.loss = original_loss
            self.loss_fcr = 0.0

        self.loss.backward()
        self.optimizer.step()
        return {
            "loss": float(self.loss.detach().item()),
            "loss_original": float(original_loss.detach().item()),
            "loss_fcr": float(self.loss_fcr),
        }
