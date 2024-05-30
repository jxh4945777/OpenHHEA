import numpy as np
import torch
import torch.nn as nn


def l1(ll, rr):
    return torch.sum(torch.abs(ll - rr))