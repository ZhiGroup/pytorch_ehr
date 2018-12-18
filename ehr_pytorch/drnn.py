#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:51:34 2018

@author: ginnyzhu
"""
#separate DRNN model related
######Dilated RNN related methods
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import sys, random
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
import math


use_cuda = torch.cuda.is_available()


class Drnn(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(Drnn, self).__init__()
        
    def drnn_layer(self, cell, inputs, rate, hidden=None):

        #n_steps = len(inputs)
        n_steps = inputs.size(0)
        #print('n_steps',n_steps) 
        #batch_size = inputs[0].size(0)
        batch_size = inputs.size(1)
        #print('batch size',batch_size) --verified correct
        hidden_size = cell.hidden_size
        #print('hidden size',hidden_size) --verified correct

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden
       
    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):

        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):

        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):

        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):

        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, Variable(zeros_)))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps


    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs
    
    def init_hidden(self, batch_size, hidden_size):
        c = Variable(torch.zeros(batch_size, hidden_size)) ############## hidden_dim??? hidden_size?? Also where is other batch_size 
        if use_cuda:
            c = c.cuda()
        if self.cell_type == "LSTM":
            m = Variable(torch.zeros(batch_size, hidden_size))  #batch_size should be part of self.batch_size I think 
            if use_cuda:
                m = m.cuda()
            return (c, m)
        else:
            return c        


