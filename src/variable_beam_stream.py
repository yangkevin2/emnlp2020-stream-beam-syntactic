from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from phrase_tree import PhraseTree, FScore, convert_to_tree
from util import predict_label, predict_struct, pad_to_length, pad_to_max_length


def my_predict_struct(network, fwd, back, stack, current_shift_i, lengths): # for tracking time separately in profiler visualization
    return predict_struct(network, fwd, back, stack, current_shift_i, lengths)
    

def my_predict_label(network, fwd, back, stack, current_shift_i, lengths, step):
    return predict_label(network, fwd, back, stack, current_shift_i, lengths, step)


def select_source_indices(num_valid_beams, master_progress, max_beams, min_prog=False):
    if min_prog:
        indices = torch.LongTensor(list(range(len(num_valid_beams)))).to(num_valid_beams.device)
        prog_min = master_progress.min()
        mp_indices = (master_progress == prog_min).nonzero().flatten()
        nvb = num_valid_beams[mp_indices]
        num_beams = torch.cumsum(nvb, dim=0)
        allowed_mask = (num_beams <= max_beams).float()
        selected_indices = mp_indices[:int(allowed_mask.sum())]
        unselected_mask = torch.ones_like(master_progress)
        unselected_mask[selected_indices] = 0
        unselected_indices = unselected_mask.nonzero().flatten()
        return selected_indices, unselected_indices
    else:
        total_num_beams = torch.cumsum(num_valid_beams, dim=0)
        allowed_mask = (total_num_beams <= max_beams).float()
        cutoff = int(allowed_mask.sum())
        indices = torch.arange(len(num_valid_beams)).to(num_valid_beams.device)
        return indices[:cutoff], indices[cutoff:] # selected, unselected


def mask_ap(log_probs, ap):
    shape = log_probs.shape
    max_log_probs, _ = log_probs.max(dim=1)
    prune_mask = max_log_probs.unsqueeze(1) - ap > log_probs
    return log_probs - prune_mask * 1e8


def parse_batch_variable_beam_stream(sentences, fm, network, k=5, ap=2.5, mc=3, max_beams=256, encode_batch_size=10, max_si=64, min_prog=False): # beam decoding
    assert not network.training
    num_indices = 0
    count = 0
    encode_batch_diff_limit = 100 * encode_batch_size

    total_num_sentences = len(sentences)

    # init master variables
    master_original_index = torch.arange(0).to(network.device) # batch
    master_current_shift_i = torch.zeros(0, k).long().to(network.device) # batch x k
    master_stack = torch.zeros(0, k, 0, 2).long().to(network.device) - 1 # -1 padding. left, right. batch x k x seq x 2
    master_label_actions = torch.zeros(0, k, 0, 3).long().to(network.device) - 1 # -1 padding. left, right label. batch x k x seq x 3
    master_all_should_combine = torch.zeros(0, k, 0).long().to(network.device) # history of actions. batch x k x seq
    master_log_probs = torch.zeros(0, k).to(network.device) # batch x k
    master_lengths = torch.zeros(0).long().to(network.device) # batch
    master_num_valid_beams = torch.ones(0).long().to(network.device) # batch
    master_steps = torch.zeros(0).long().to(network.device) # batch of current step num

    master_fwd, master_back = torch.zeros(0, 0, 2*network.lstm_units).to(network.device), torch.zeros(0, 0, 2*network.lstm_units).to(network.device)

    done_info = [None for _ in range(len(sentences))]
    current_sentence_idx = 0

    while True:
        starting_new_encode_group = False
        while (len(master_original_index) <= max_si - encode_batch_diff_limit or (starting_new_encode_group and len(master_original_index) <= max_si - encode_batch_size)) and current_sentence_idx < total_num_sentences:
            starting_new_encode_group = True
            sentence_info = [fm.sentence_sequences(sentence) for sentence in sentences[current_sentence_idx:current_sentence_idx + encode_batch_size]]
            batch_w, batch_t = [si[0] for si in sentence_info], [si[1] for si in sentence_info]
            encode_lengths = torch.LongTensor([len(example) for example in batch_w]).to(network.device)
            batch_w = pad_sequence([torch.from_numpy(example) for example in batch_w]).long().to(network.device)
            batch_t = pad_sequence([torch.from_numpy(example) for example in batch_t]).long().to(network.device)
            fwd, back = network.evaluate_recurrent(batch_w, batch_t, encode_lengths) # seq x batch x hidden

            # update master vars
            added_size = len(sentence_info)
            # lengths
            master_lengths = torch.cat([master_lengths, torch.LongTensor([len(sentence) for sentence in sentences[current_sentence_idx:current_sentence_idx + encode_batch_size]]).to(network.device)], dim=0)
            # trim lengths
            max_length = master_lengths.max()
            encode_max_length = max_length + 2
            master_stack = pad_to_length(master_stack[:, :, :max_length], max_length, 2)
            master_label_actions = pad_to_length(master_label_actions[:, :, :2*max_length], 2*max_length, 2)
            master_all_should_combine = pad_to_length(master_all_should_combine[:, :, :2*max_length], 2*max_length, 2)
            master_fwd = pad_to_length(master_fwd[:encode_max_length], encode_max_length, 0)
            master_back = pad_to_length(master_back[:encode_max_length], encode_max_length, 0)

            # original_index
            master_original_index = torch.cat([master_original_index, torch.arange(added_size).to(network.device) + current_sentence_idx], dim=0)
            # current_shift_i
            master_current_shift_i = torch.cat([master_current_shift_i, torch.zeros(added_size, k).long().to(network.device)], dim=0)
            # stack
            master_stack = torch.cat([master_stack, (torch.zeros(added_size, k, max_length, 2) - 1).long().to(network.device)], dim=0)
            # label_actions
            master_label_actions = torch.cat([master_label_actions, (torch.zeros(added_size, k, 2*max_length, 3) - 1).long().to(network.device)], dim=0)
            # all_should_combine
            master_all_should_combine = torch.cat([master_all_should_combine, torch.zeros(added_size, k, 2*max_length).long().to(network.device)], dim=0)
            # log_probs
            master_log_probs = torch.cat([master_log_probs, torch.cat([torch.zeros(added_size, 1), torch.zeros(added_size, k-1) - 1e8], dim=1).to(network.device)], dim=0)
            # fwd
            master_fwd = torch.cat([master_fwd, pad_to_length(fwd, encode_max_length, 0)], dim=1)
            # back
            master_back = torch.cat([master_back, pad_to_length(back, encode_max_length, 0)], dim=1)
            # num valid beams
            master_num_valid_beams = torch.cat([master_num_valid_beams, torch.ones(added_size).long().to(network.device)], dim=0)
            # steps
            master_steps = torch.cat([master_steps, torch.zeros(added_size).long().to(network.device)], dim=0)

            current_sentence_idx += encode_batch_size
        
        # select beams for current iteration
        selected_indices, unselected_indices = select_source_indices(master_num_valid_beams, master_steps, max_beams, min_prog)
        lengths = master_lengths[selected_indices]
        original_index = master_original_index[selected_indices]
        current_shift_i = master_current_shift_i[selected_indices]
        stack = master_stack[selected_indices]
        label_actions = master_label_actions[selected_indices]
        all_should_combine = master_all_should_combine[selected_indices]
        log_probs = master_log_probs[selected_indices]
        fwd = master_fwd[:, selected_indices]
        back = master_back[:, selected_indices]
        num_valid_beams = master_num_valid_beams[selected_indices]
        steps = master_steps[selected_indices]

        num_indices += (log_probs > -1e8).sum()
        count += 1

        # note sh = 0, comb = 1
        # predict sh or comb action
        can_shift = current_shift_i < lengths.unsqueeze(1) # batch x k
        can_combine = stack[:, :, 1, 0] != -1 # at least 2 things in stack. batch x k
        assert (can_shift | can_combine).long().sum() == can_shift.shape[0] * k # at least 1 action possible
        # create struct features
        both_possible = can_shift & can_combine & (log_probs > -1e8) # batch x k
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

        # there are only 2 decisions here so we don't do mc heuristic here
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

        all_should_combine = torch.cat([new_decisions.unsqueeze(2), gathered_beams[:, :, :-1]], dim=2)
        log_probs = top_probs
        log_probs = mask_ap(log_probs, ap)

        # predict label
        current_shift_i = current_shift_i.clamp(max=max_length) # only affects stuff whose prob is -1e8 anyway; avoid a crash
        stack = stack.clamp(max=max_length-1) # only affects stuff whose prob is -1e8 anyway; avoid a crash
        valid_beam_indices = (log_probs > -1e8).long().flatten().nonzero().flatten()
        valid_label_scores, _ = my_predict_label(network,
                                        fwd.unsqueeze(2).repeat(1, 1, k, 1).flatten(1, 2)[:lengths.max()+2, valid_beam_indices, :],
                                        back.unsqueeze(2).repeat(1, 1, k, 1).flatten(1, 2)[:lengths.max()+2, valid_beam_indices, :],
                                        stack[valid_beam_indices, :lengths.max()],
                                        current_shift_i[valid_beam_indices],
                                        lengths.unsqueeze(1).repeat(1, k).flatten()[valid_beam_indices],
                                        steps.unsqueeze(1).repeat(1, k).flatten()[valid_beam_indices]) # batch*k x vocab
        label_scores = torch.zeros(log_probs.shape[0]*log_probs.shape[1], valid_label_scores.shape[1]).to(valid_label_scores.device) - 1e8
        label_scores[valid_beam_indices] = valid_label_scores
        # reshape stuff properly
        stack = stack.view(-1, k, stack.shape[1], stack.shape[2])
        current_shift_i = current_shift_i.view(-1, k)

        # figure out label beams, with mc heuristic
        label_probs = label_scores.log_softmax(dim=-1).view(-1, k, label_scores.shape[1]) # batch x k x vocab
        label_probs = label_probs + log_probs.unsqueeze(2)
        mc_probs, mc_indices = label_probs.topk(mc, dim=2) # batch x k x mc
        min_mc_probs = mc_probs[:, :, -1] # batch x k
        under_min_mc_mask = (label_probs < min_mc_probs.unsqueeze(2)).int() | (label_probs <= -1e8).int()
        label_probs = label_probs - under_min_mc_mask * 1e8
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
        label_actions = torch.cat([torch.cat([stack[:, :, 0, :], new_labels.unsqueeze(2)], dim=2).unsqueeze(2), label_actions[:, :, :-1, :]], dim=2)
        log_probs = top_label_probs
        log_probs = mask_ap(log_probs, ap)

        # early stopping for some sources as needed
        # update done info
        done_indices = ((2 * lengths - 2) == steps).long()
        done = done_indices.nonzero().flatten()
        not_done = (1-done_indices).nonzero().flatten()
        for di_index, j in enumerate(original_index[done]):
            done_index = done[di_index]
            # done_info[j] = [(stack[done_index, i], label_actions[done_index, i, :2*lengths[done_index] - 1].flip(0), all_should_combine[done_index, i, :2*lengths[done_index] - 1].flip(0)) for i in range(len(stack[done_index]))]
            done_info[j] = [stack[done_index], label_actions[done_index, :, :2*lengths[done_index] - 1].flip(1), all_should_combine[done_index, :, :2*lengths[done_index] - 1].flip(1)]
            done_info[j] = [(done_info[j][0][i], done_info[j][1][i], done_info[j][2][i]) for i in range(len(stack[done_index]))]
        # update master variables, stack/label_actions/all_should_combine/fwd/back seq length
        # current_shift_i
        master_current_shift_i[selected_indices] = current_shift_i
        # stack
        master_stack[selected_indices] = stack
        # label_actions
        master_label_actions[selected_indices] = label_actions
        # all_should_combine
        master_all_should_combine[selected_indices] = all_should_combine
        # log_probs
        master_log_probs[selected_indices] = log_probs
        # num valid beams
        master_num_valid_beams[selected_indices] = (log_probs > -1e8).long().sum(dim=1)
        # steps
        master_steps[selected_indices] = steps + 1

        # master update for done indices of stack, label_actions, current_shift_i, lengths, original_index, fwd, back
        master_not_done = torch.cat([selected_indices[not_done], unselected_indices], dim=0)
        master_stack = master_stack[master_not_done]
        master_label_actions = master_label_actions[master_not_done]
        master_current_shift_i = master_current_shift_i[master_not_done]
        master_lengths = master_lengths[master_not_done]
        master_original_index = master_original_index[master_not_done]
        master_all_should_combine = master_all_should_combine[master_not_done]
        master_fwd = master_fwd[:, master_not_done, :]
        master_back = master_back[:, master_not_done, :]
        master_log_probs = master_log_probs[master_not_done]
        master_num_valid_beams = master_num_valid_beams[master_not_done]
        master_steps = master_steps[master_not_done]

        # finished parsing all sentences
        if current_sentence_idx >= total_num_sentences and len(master_not_done) == 0:
            break

    assert all([di is not None for di in done_info])
    # afterward, create tree for each by assembling struct and label actions
    return sentences, done_info, num_indices, count

