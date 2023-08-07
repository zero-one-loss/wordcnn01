import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def update_bp(net, optimizer, data_loader, criterion, use_cuda, device, dtype):
    net.train()
    for data, target in data_loader:
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        optimizer.zero_grad()
        outputs = net(data, layer=net.layers[-1]+'_projection')
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        break