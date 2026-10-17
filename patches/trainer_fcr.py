import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.freqnet import freqnet
from networks.base_model import BaseModel


def make_label_preserving_view(x):
    """Mild tensor-only resize/blur perturbation for FCR training.

    JPEG is evaluated separately using a real encode/decode pipeline.
    """
    _, _, height, width = x.shape
    choice = int(torch.randint(0, 2, (1,), device=x.device).item())

    if choice == 0:
        scale = float(torch.empty(1, device=x.device).uniform_(0.82, 0.95).item())
        small_h = max(8, int(height * scale))
        small_w = max(8, int(width * scale))
        view = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
        return F.interpolate(view, size=(height, width), mode="bilinear", align_corners=False)

    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)


def bernoulli_kl_from_logits(logits_p, logits_q, eps=1e-6):
    """KL(Bernoulli(sigmoid(logits_p)) || Bernoulli(sigmoid(logits_q)))."""
    p = torch.sigmoid(logits_p).clamp(eps, 1.0 - eps)
    q = torch.sigmoid(logits_q).clamp(eps, 1.0 - eps)
    return (p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))).mean()


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.model = freqnet()
        self.fcr_enabled = bool(getattr(opt, "fcr_enabled", False))
        self.fcr_lambda = float(getattr(opt, "fcr_lambda", 0.05))

        for name, param in self.model.named_parameters():
            print("=" * 20, "requires_grad", param.requires_grad, name)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            trainable = filter(lambda p: p.requires_grad, self.model.parameters())
            if opt.optim == "adam":
                self.optimizer = torch.optim.Adam(trainable, lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == "sgd":
                self.optimizer = torch.optim.SGD(trainable, lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [sgd, adam]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= 0.8
            if param_group["lr"] < min_lr:
                return False
        self.lr = param_group["lr"]
        print("*" * 25)
        print(f"Changing lr from {param_group['lr'] / 0.8} to {param_group['lr']}")
        print("*" * 25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

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
            self.loss_fcr = float(consistency_loss.detach().item())
        else:
            self.loss = original_loss
            self.loss_fcr = 0.0

        self.loss.backward()
        self.optimizer.step()
