import numpy as np

import torch
import torch.nn as nn

def load_npz(path):
    return nn.Parameter(torch.from_numpy(np.load(path)['arr_0']).float())

def predict_struct(network, fwd, back, stack, current_shift_i, lengths):
    batch_size = lengths.shape[0]
    s1_left = stack[:, 1, 0] + 1 # batch; 0 if it's -1
    s1_right = stack[:, 1, 1] + 1
    s0_left = stack[:, 0, 0] + 1
    s0_right = stack[:, 0, 1] + 1
    lefts = torch.stack([torch.ones(batch_size).long().to(network.device),
                         s1_left.clamp(min=1),
                         s0_left.clamp(min=1),
                         current_shift_i + 1], dim=1) # batch x 4
    rights = torch.stack([(s1_left - 1).clamp(min=0),
                          s1_right.clamp(min=0),
                          s0_right.clamp(min=0),
                          lengths], dim=1) # batch x 4
    scores = network.evaluate_struct(fwd, back, lefts, rights) # batch x 2
    _, pred_indices = scores.max(dim=1)
    return scores, pred_indices
    

def predict_label(network, fwd, back, stack, current_shift_i, lengths, step):
    batch_size = lengths.shape[0]
    s0_left = stack[:, 0, 0] + 1 # batch; 0 if it's -1
    s0_right = stack[:, 0, 1] + 1
    lefts = torch.stack([torch.ones(batch_size).long().to(network.device),
                         s0_left.clamp(min=1),
                         current_shift_i + 1], dim=1) # batch x 3
    rights = torch.stack([(s0_left - 1).clamp(min=0),
                          s0_right.clamp(min=0),
                          lengths], dim=1) # batch x 3
    scores = network.evaluate_label(fwd, back, lefts, rights) # batch x labels
    mask_indices = (step >= 2 * lengths - 2)
    scores[mask_indices, 0] = -1e8 # disallow this at the end
    _, pred_indices = scores.max(dim=1)
    return scores, pred_indices


def pad_to_max_length(t1, t2, dim, side='right'):
    if t1.size(dim) < t2.size(dim):
        t1 = pad_to_length(t1, t2.size(dim), dim, side)
    elif t2.size(dim) < t1.size(dim):
        t2 = pad_to_length(t2, t1.size(dim), dim, side)
    return t1, t2


def pad_to_length(tensor, length, dim, side='right'):
    assert side in ['left', 'right']
    assert tensor.size(dim) <= length
    if tensor.size(dim) == length:
        return tensor
    else:
        zeros_shape = list(tensor.shape)
        zeros_shape[dim] = length - tensor.size(dim)
        zeros_shape = tuple(zeros_shape)
        if side == 'right':
            return torch.cat([tensor, torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device)], dim=dim)
        else:
            return torch.cat([torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device), tensor], dim=dim)
