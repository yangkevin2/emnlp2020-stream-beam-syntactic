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


def parse_batch_beam(sentences, fm, network, k=5): # beam decoding
    assert not network.training
    num_indices = 0
    count = 0

    batch_size = len(sentences)
    lengths = [len(sentence) for sentence in sentences]
    lengths = torch.LongTensor(lengths).to(network.device) # batch
    max_length = lengths.max()
    original_index = torch.arange(batch_size).to(network.device) # batch x k
    current_shift_i = torch.zeros(batch_size, k).long().to(network.device) # batch x k
    stack = torch.zeros(batch_size, k, max_length, 2).long().to(network.device) - 1 # -1 padding. left, right. batch x k x seq x 2
    label_actions = torch.zeros(batch_size, k, 0, 3).long().to(network.device) - 1 # -1 padding. left, right label. batch x k x seq x 3
    all_should_combine = torch.zeros(batch_size, k, 0).long().to(network.device) # history of actions. batch x k x seq
    log_probs = torch.cat([torch.zeros(batch_size, 1), torch.zeros(batch_size, k-1) - 1e8], dim=1).to(network.device) # batch x k

    done_info = [None for _ in range(batch_size)]

    # encode all sentences; afterward only dealing with lefts/rights of spans
    sentence_info = [fm.sentence_sequences(sentence) for sentence in sentences]
    batch_w, batch_t = [si[0] for si in sentence_info], [si[1] for si in sentence_info]
    encode_lengths = torch.LongTensor([len(example) for example in batch_w]).to(network.device)
    assert ((lengths + 2 - encode_lengths)**2).sum() == 0 # should be the same with sos/eos toks
    batch_w = pad_sequence([torch.from_numpy(example) for example in batch_w]).long().to(network.device)
    batch_t = pad_sequence([torch.from_numpy(example) for example in batch_t]).long().to(network.device)
    fwd, back = network.evaluate_recurrent(batch_w, batch_t, encode_lengths) # seq x batch x hidden

    # for step in max steps:
    for step in range(2 * max_length - 1):
        num_indices += (log_probs > -1e8).sum()
        count += 1
        # note sh = 0, comb = 1
        # predict sh or comb action
        can_shift = current_shift_i < lengths.unsqueeze(1) # batch x k
        can_combine = stack[:, :, 1, 0] != -1 # at least 2 things in stack. batch x k
        assert (can_shift | can_combine).long().sum() == can_shift.shape[0] * k # at least 1 action possible
        # create struct features
        both_possible = can_shift & can_combine # batch x k
        both_possible = both_possible.flatten()
        if both_possible.long().sum() > 0: 
            both_possible_scores, _ = my_predict_struct(network,
                                                 fwd.unsqueeze(2).repeat(1, 1, k, 1).flatten(1, 2)[:, both_possible, :],
                                                 back.unsqueeze(2).repeat(1, 1, k, 1).flatten(1, 2)[:, both_possible, :],
                                                 stack.flatten(0, 1)[both_possible],
                                                 current_shift_i.flatten(0, 1)[both_possible],
                                                 lengths.unsqueeze(1).repeat(1, k).flatten()[both_possible])
            both_possible_log_probs = both_possible_scores.log_softmax(dim=-1) # batch*k filtered x 2
        else:
            both_possible_log_probs = torch.zeros(0, 2).to(network.device) # there's nothing here

        # construct batch x k x 2 of log prob for new shift, log prob for new combine
        new_log_probs = torch.zeros_like(log_probs).unsqueeze(2).repeat(1, 1, 2)
        cannot_shift = (~can_shift).long().flatten().nonzero().flatten()
        cannot_combine = (~can_combine).long().flatten().nonzero().flatten()
        new_log_probs = new_log_probs.flatten(0, 1) # batch*k x 2
        new_log_probs[cannot_shift, 0] = -1e8
        new_log_probs[cannot_combine, 1] = -1e8
        new_log_probs[both_possible] = both_possible_log_probs
        new_log_probs = log_probs.unsqueeze(2) + new_log_probs.view(-1, k, 2) # batch x k x 2
        new_log_probs = new_log_probs.flatten(1) # batch x k*2
        top_probs, top_indices = new_log_probs.topk(k, dim=1) # batch x k, batch x k
        new_decisions = top_indices % 2 # batch x k
        beam_indices = top_indices // 2 # batch x k
        gathered_beams = torch.gather(all_should_combine, 1, beam_indices.unsqueeze(2).expand_as(all_should_combine))
        stack = torch.gather(stack, 1, beam_indices.unsqueeze(2).unsqueeze(3).expand_as(stack))
        label_actions = torch.gather(label_actions, 1, beam_indices.unsqueeze(2).unsqueeze(3).expand_as(label_actions))
        current_shift_i = torch.gather(current_shift_i, 1, beam_indices)

        shift_indices = (1 - new_decisions).flatten().nonzero().flatten()
        combine_indices = new_decisions.flatten().nonzero().flatten()
        stack = stack.flatten(0, 1)
        current_shift_i = current_shift_i.flatten(0, 1)

        # update stack, current_shift_i
        # shift actions
        stack[shift_indices] = torch.roll(stack[shift_indices], 1, dims=1)
        stack[shift_indices, 0, :] = current_shift_i[shift_indices].unsqueeze(1).repeat(1, 2)
        current_shift_i[shift_indices] += 1
        # combine actions
        combined_leftright = torch.stack([stack[combine_indices][:, 1, 0], stack[combine_indices][:, 0, 1]], dim=1)
        stack[combine_indices] = torch.roll(stack[combine_indices], -1, dims=1)
        stack[combine_indices, 0, :] = combined_leftright
        stack[combine_indices, -1, :] = -1 # roll a -1 to the end

        all_should_combine = torch.cat([gathered_beams, new_decisions.unsqueeze(2)], dim=2)
        log_probs = top_probs

        # predict label
        label_scores, _ = my_predict_label(network,
                                    fwd.unsqueeze(2).repeat(1, 1, k, 1).flatten(1, 2),
                                    back.unsqueeze(2).repeat(1, 1, k, 1).flatten(1, 2),
                                    stack,
                                    current_shift_i,
                                    lengths.unsqueeze(1).repeat(1, k).flatten(),
                                    step) # batch*k x vocab
        # reshape stuff properly
        stack = stack.view(-1, k, stack.shape[1], stack.shape[2])
        current_shift_i = current_shift_i.view(-1, k)

        # figure out label beams
        label_probs = label_scores.log_softmax(dim=-1).view(-1, k, label_scores.shape[1]) # batch x k x vocab
        label_probs = label_probs + log_probs.unsqueeze(2)
        label_probs = label_probs.flatten(1) # batch x k*vocab
        top_label_probs, top_label_indices = label_probs.topk(k, dim=1) # batch x k
        new_labels = top_label_indices % network.label_out
        label_beam_indices = top_label_indices // network.label_out
        
        # update vars for label beams
        stack = torch.gather(stack, 1, label_beam_indices.unsqueeze(2).unsqueeze(3).expand_as(stack))
        current_shift_i = torch.gather(current_shift_i, 1, label_beam_indices)
        all_should_combine = torch.gather(all_should_combine, 1, label_beam_indices.unsqueeze(2).expand_as(all_should_combine))

        # add label to tensor of (batch x steps x 3 (left, right, label)) actions
        label_actions = torch.gather(label_actions, 1, label_beam_indices.unsqueeze(2).unsqueeze(3).expand_as(label_actions))
        label_actions = torch.cat([label_actions, torch.cat([stack[:, :, 0, :], new_labels.unsqueeze(2)], dim=2).unsqueeze(2)], dim=2)
        log_probs = top_label_probs
        
        # early stopping for some sources as needed
        # update done info
        done_indices = ((2 * lengths - 2) == step).long()
        done = done_indices.nonzero().flatten()
        not_done = (1-done_indices).nonzero().flatten()
        for i, j in enumerate(original_index[done]):
            done_index = done[i]
            done_info[j] = [(stack[done_index, i], label_actions[done_index, i], all_should_combine[done_index, i]) for i in range(len(stack[done_index]))]
        # update stack, label_actions, current_shift_i, lengths, original_index, fwd, back
        stack = stack[not_done]
        label_actions = label_actions[not_done]
        current_shift_i = current_shift_i[not_done]
        lengths = lengths[not_done]
        original_index = original_index[not_done]
        all_should_combine = all_should_combine[not_done]
        fwd = fwd[:, not_done, :]
        back = back[:, not_done, :]
        log_probs = log_probs[not_done]

    assert all([di is not None for di in done_info])
    return sentences, done_info, num_indices, count

