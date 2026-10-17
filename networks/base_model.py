import os
import torch
import torch.nn as nn
from torch.nn import init


class BaseModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
        os.makedirs(self.save_dir, exist_ok=True)

    def save_networks(self, epoch):
        path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Saved model weights: {path}")

    def load_networks(self, epoch):
        path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        state = torch.load(path, map_location=self.device)
        weights = state.get("model", state) if isinstance(state, dict) else state
        self.model.load_state_dict(weights)
        if isinstance(state, dict) and "total_steps" in state:
            self.total_steps = state["total_steps"]
        if self.isTrain and isinstance(state, dict) and "optimizer" in state and not getattr(self.opt, "new_optim", False):
            self.optimizer.load_state_dict(state["optimizer"])
            for item in self.optimizer.state.values():
                for key, value in item.items():
                    if torch.is_tensor(value):
                        item[key] = value.to(self.device)

    def eval(self):
        self.model.eval()

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(module):
        name = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in name or "Linear" in name):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=gain)
            else:
                raise ValueError(f"Unknown initialization: {init_type}")
            if getattr(module, "bias", None) is not None:
                init.constant_(module.bias.data, 0.0)
        elif "BatchNorm2d" in name:
            init.normal_(module.weight.data, 1.0, gain)
            init.constant_(module.bias.data, 0.0)
    net.apply(init_func)
