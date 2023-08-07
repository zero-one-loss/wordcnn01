import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


def sign(x):
    return x.sign_()
    # return torch.sigmoid(x)


def signb(x):
    # return F.relu_(torch.sign(x)).float()
    return x.float()

def softmax_(x):
    return F.softmax(x.float(), dim=1)

# def softmax_(x):
#     return x.float()


def sigmoid_(x):
    return torch.sigmoid(x)


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1)
        # nn.init.uniform_(m.weight, -1, 1)
        m.weight = torch.nn.Parameter(
            m.weight / torch.norm(
                m.weight.view((m.weight.size(0), -1)),
                dim=1).view((-1, 1, 1, 1))
        )
        # m.weight.requires_grad = False
        if m.bias is not None:
            init.constant_(m.bias, 0)
            # m.bias.requires_grad = False
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1)
        # nn.init.uniform_(m.weight, -1, 1)
        m.weight = torch.nn.Parameter(
            m.weight / torch.norm(
                m.weight.view((m.weight.size(0), -1)),
                dim=1).view((-1, 1, 1, 1))
        )
        # m.weight.requires_grad = False
        if m.bias is not None:
            init.constant_(m.bias, 0)
            # m.bias.requires_grad = False
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def _01_init(model):
    for name, m in model.named_modules():
        if 'si' not in name:
            if isinstance(m, nn.Conv2d):
                m.weight = torch.nn.Parameter(
                    m.weight.sign()
                )
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.Linear):
                m.weight = torch.nn.Parameter(
                    m.weight.sign())
                if m.bias is not None:
                    m.bias.data.zero_()


def init_weights(model, kind='normal'):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if kind == 'normal':
                nn.init.normal_(m.weight, mean=0, std=1)
            elif kind == 'uniform':
                nn.init.uniform_(m.weight, -1, 1)
            m.weight = torch.nn.Parameter(
                m.weight / torch.norm(
                    m.weight.view((m.weight.size(0), -1)),
                    dim=1).view((-1, 1, 1, 1))
            )
            # m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.data.zero_()
                # m.bias.requires_grad = False

        if isinstance(m, nn.Linear):
            if kind == 'normal':
                nn.init.normal_(m.weight, mean=0, std=1)
            elif kind == 'uniform':
                nn.init.uniform_(m.weight, -1, 1)
            m.weight = torch.nn.Parameter(
                m.weight / torch.norm(m.weight, dim=1, keepdim=True))
            # m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.data.zero_()
                # m.bias.requires_grad = False


class WordCNN01(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False,
                 ndim=100, drop_p=0, bias=True):
        super(WordCNN01, self).__init__()
        if act == "sign":
            self.act = torch.sign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb
        # if no_embed:
        #     self.embedding = None
        # else:
        #     self.embedding = nn.Embedding(nwords, ndim)
        self.conv1_si = nn.Conv2d(1, 150, kernel_size=(4, ndim), padding=(2, 0), bias=bias)
        self.fc2_si = nn.Linear(150, num_classes, bias=bias)
        self.layers = ["conv1_si", "fc2_si"]
        self.drop = nn.Dropout(drop_p)
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":

                out = self.conv1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = torch.sign(out)
            out.squeeze_(dim=-1)
            # out = F.avg_pool1d(out, out.size(2))
            out = F.relu(out).sum(dim=2)
            out = out.reshape(out.size(0), -1)
            # out = out / out.norm(dim=1, keepdim=True)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.drop(out)
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            # out = out / out.norm(dim=1, keepdim=True)
            out = self.signb(out)

        return out


class WordCNNbp(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False,
                 ndim=100, drop_p=0, bias=True):
        super(WordCNNbp, self).__init__()
        if act == "sign":
            self.act = torch.sign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb
        # if no_embed:
        #     self.embedding = None
        # else:
        #     self.embedding = nn.Embedding(nwords, ndim)
        self.conv1_si = nn.Conv2d(1, 150, kernel_size=(4, ndim), padding=(2, 0), bias=bias)
        self.fc2_si = nn.Linear(150, num_classes, bias=bias)
        self.layers = ["conv1_si", "fc2_si"]
        self.drop = nn.Dropout(drop_p)
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":

                out = self.conv1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = torch.relu(out)
            out.squeeze_(dim=-1)
            out = F.avg_pool1d(out, out.size(2))
            # out = F.relu(out).sum(dim=2)
            out = out.reshape(out.size(0), -1)
            # out = out / out.norm(dim=1, keepdim=True)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.drop(out)
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            # out = out / out.norm(dim=1, keepdim=True)
            out = self.signb(out)

        return out


arch = {}



arch['wordcnn01'] = WordCNN01


if __name__ == '__main__':
    # net = Toy(1, act='sign')
    # x = torch.rand(size=(100, 3, 32, 32))
    # # x = torch.rand(size=(100, 6, 16, 16))
    # output = net(x)  # shape 100, 1
    # Test case
    x = torch.rand(size=(8, 12)).long()
    # net = Toy2FConv(10, act='sign', sigmoid=False, softmax=True, scale=1, bias=True)
    net = arch['wordcnn01'](num_classes=2, act=sign, sigmoid=False, softmax=True,
                 no_embed=False, nwords=40, ndim=100, drop_p=0.3)
    #
    net.eval()
    output = net(x)
    layers = net.layers
    temp_out = x
    for i in range(len(layers)):
        print(f'Running on {layers[i]}')
        out = net(temp_out, input_=layers[i])
        temp_projection = net(temp_out, input_=layers[i], layer=layers[i] + '_projection')
        current_out = net(temp_out, input_=layers[i], layer=layers[i] + '_output')
        temp_out = net(temp_projection, input_=layers[i] + '_ap', layer=layers[i] + '_output')
    # import os
    # path = [os.path.join('checkpoints', 'toy_v3.pt') for i in range(200)]
    # net = Ensemble(structure=Toy(1, 'sign', False, False), path=path)
    # yp = net.predict_proba(x)
