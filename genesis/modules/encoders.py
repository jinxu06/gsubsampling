# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2020 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
import torch.nn.functional as F

import genesis.modules.blocks as B


class MONetCompEncoder(nn.Module):

    def __init__(self, 
                 img_size,
                 input_channels,
                 comp_enc_channels,
                 comp_ldim,
                 act):
        super(MONetCompEncoder, self).__init__()
        nin = input_channels
        c = comp_enc_channels
        self.ldim = comp_ldim
        nin_mlp = 2*c * (img_size//16)**2
        nhid_mlp = max(256, 2*self.ldim)
        self.module = Seq(nn.Conv2d(nin+1, c, 3, 2, 1), act,
                          nn.Conv2d(c, c, 3, 2, 1), act,
                          nn.Conv2d(c, 2*c, 3, 2, 1), act,
                          nn.Conv2d(2*c, 2*c, 3, 2, 1), act,
                          B.Flatten(),
                          nn.Linear(nin_mlp, nhid_mlp), act,
                          nn.Linear(nhid_mlp, 2*self.ldim))

    def forward(self, x):
        return self.module(x)
