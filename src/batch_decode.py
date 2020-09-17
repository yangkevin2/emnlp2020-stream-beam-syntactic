from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from phrase_tree import PhraseTree, FScore, convert_to_tree
from util import predict_label, predict_struct


def my_predict_struct(network, fwd, back, stack, current_shift_i, lengths): # for tracking time separately in profiler visualization
    return predict_struct(network, fwd, back, stack, current_shift_i, lengths)
    

def my_predict_label(network, fwd, back, stack, current_shift_i, lengths, step):
    return predict_label(network, fwd, back, stack, current_shift_i, lengths, step)


def parse_batch(sentences, fm, network): # greedy decoding
    assert not network.training
    num_indices = 0
    count = 0

    batch_size = len(sentences)
    lengths = [len(sentence) for sentence in sentences]
    lengths = torch.LongTensor(lengths).to(network.device)
    max_length = lengths.max()
    original_index = torch.arange(batch_size).to(network.device)
    current_shift_i = torch.zeros(batch_size).long().to(network.device)
    stack = torch.zeros(batch_size, max_length, 2).long().to(network.device) - 1 # -1 padding. left, right
    label_actions = torch.zeros(batch_size, 0, 3).long().to(network.device) - 1 # -1 padding. left, right label
    all_should_combine = torch.zeros(batch_size, 0).long().to(network.device)
    # label_actions = []

    done_info = [None for _ in range(batch_size)]

    # encode all sentences; afterward only dealing with lefts/rights of spans
    sentence_info = [fm.sentence_sequences(sentence) for sentence in sentences]
    batch_w, batch_t = [si[0] for si in sentence_info], [si[1] for si in sentence_info]
    encode_lengths = torch.LongTensor([len(example) for example in batch_w]).to(network.device)
    assert ((lengths + 2 - encode_lengths)**2).sum() == 0 # should be the same with sos/eos toks
    batch_w = pad_sequence([torch.from_numpy(example) for example in batch_w]).long().to(network.device)
    batch_t = pad_sequence([torch.from_numpy(example) for example in batch_t]).long().to(network.device)
    fwd, back = network.evaluate_recurrent(batch_w, batch_t, encode_lengths)

    # for step in max steps:
    for step in range(2 * max_length - 1):
        num_indices += len(lengths)
        count += 1
        # note sh = 0, comb = 1
        # predict sh or comb action
        can_shift = current_shift_i < lengths
        can_combine = stack[:, 1, 0] != -1 # at least 2 things in stack
        assert (can_shift | can_combine).long().sum() == len(can_shift) # at least 1 action possible
        # create struct features
        both_possible = can_shift & can_combine
        if both_possible.long().sum() > 0:
            _, both_possible_preds = my_predict_struct(network,
                                                 fwd[:, both_possible, :],
                                                 back[:, both_possible, :],
                                                 stack[both_possible],
                                                 current_shift_i[both_possible],
                                                 lengths[both_possible])
        else:
            both_possible_preds = torch.zeros(0).long().to(network.device) # there's nothing here
        should_combine_indicator = can_combine.long() # 1 where combine is possible
        should_combine_indicator[both_possible] = both_possible_preds # replace with 0 where decided to shift instead
        should_shift = (1 - should_combine_indicator).nonzero().flatten()
        should_combine = should_combine_indicator.nonzero().flatten()
        # update stack, current_shift_i
        # shift actions
        stack[should_shift] = torch.roll(stack[should_shift], 1, dims=1)
        stack[should_shift, 0, :] = current_shift_i[should_shift].unsqueeze(1).repeat(1, 2)
        current_shift_i[should_shift] += 1
        # combine actions
        combined_leftright = torch.stack([stack[should_combine][:, 1, 0], stack[should_combine][:, 0, 1]], dim=1)
        stack[should_combine] = torch.roll(stack[should_combine], -1, dims=1)
        stack[should_combine, 0, :] = combined_leftright
        stack[should_combine, -1, :] = -1 # roll a -1 to the end
        all_should_combine = torch.cat([all_should_combine, should_combine_indicator.long().unsqueeze(1)], dim=1)

        # predict label
        _, label_preds = my_predict_label(network,
                                    fwd,
                                    back,
                                    stack,
                                    current_shift_i,
                                    lengths,
                                    step) # batch
        # add label to tensor of (batch x steps x 3 (left, right, label)) actions
        label_actions = torch.cat([label_actions, torch.cat([stack[:, 0, :], label_preds.unsqueeze(1)], dim=1).unsqueeze(1)], dim=1)
        # early stopping for some sources as needed
        # update done info
        done_indices = ((2 * lengths - 2) == step).long()
        done = done_indices.nonzero().flatten()
        not_done = (1-done_indices).nonzero().flatten()
        for i, j in enumerate(original_index[done]):
            k = done[i]
            done_info[j] = (stack[k], label_actions[k], all_should_combine[k])
        # update stack, label_actions, current_shift_i, lengths, original_index, fwd, back
        stack = stack[not_done]
        label_actions = label_actions[not_done]
        current_shift_i = current_shift_i[not_done]
        lengths = lengths[not_done]
        original_index = original_index[not_done]
        all_should_combine = all_should_combine[not_done]
        fwd = fwd[:, not_done, :]
        back = back[:, not_done, :]

    assert all([di is not None for di in done_info])
    # afterward, create tree for each by assembling struct and label actions
    return (sentences, [[di] for di in done_info]), num_indices, count

