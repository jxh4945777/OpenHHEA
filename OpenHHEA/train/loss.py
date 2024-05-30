import numpy as np
import torch
import torch.nn as nn


class HHEALoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pairs:np.ndarray, features:torch.Tensor) -> torch.Tensor:
        pass


class SimpleHHEALoss(HHEALoss):
    def __init__(self, gamma:float=1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def l1(self, ll, rr):
        return torch.sum(torch.abs(ll - rr), axis=-1)

    def forward(self, pairs: np.ndarray, features: torch.Tensor) -> torch.Tensor:
        feat = features[pairs]
        l, r, fl, fr = feat[:, 0, :], feat[:, 1, :], feat[:, 2, :], feat[:, 3, :]
        loss = torch.sum(nn.ReLU()(self.gamma + self.l1(l, r) - self.l1(l, fr)) + nn.ReLU()(self.gamma + self.l1(l, r) - self.l1(fl, r))) / features.shape[0] 
        return loss

class XGEALoss(HHEALoss):
    def __init__(self, gamma:float=1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, pairs: np.ndarray, features: torch.Tensor) -> torch.Tensor:
        pass