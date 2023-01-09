# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def binary(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.manual_seed(42)
        torch.nn.init.kaiming_normal_(w.weight)
        sigma = w.weight.data.std()
        w.weight.data = torch.sign(w.weight.data) * sigma
        try:
            w.bias.data = torch.zeros_like(w.bias.data)
        except e:
            pass


def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.manual_seed(42)
        torch.nn.init.kaiming_normal_(w.weight)
        try:
            w.bias.data = torch.zeros_like(w.bias.data)
        except:
            pass


def kaiming_uniform(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.manual_seed(42)
        torch.nn.init.kaiming_uniform_(w.weight)
        try:
            w.bias.data = torch.zeros_like(w.bias.data)
        except:
            pass


def orthogonal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.manual_seed(42)
        torch.nn.init.orthogonal_(w.weight)
        try:
            w.bias.data = torch.zeros_like(w.bias.data)
        except:
            pass
