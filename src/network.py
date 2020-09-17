import time
import random
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from phrase_tree import PhraseTree, FScore
from features import FeatureMapper
from parser import Parser
from lstm import CH_LSTM
from util import load_npz


class Network(nn.Module):
    def __init__(self,
        word_count,
        tag_count,
        word_dims,
        tag_dims,
        lstm_units,
        hidden_units,
        struct_out,
        label_out,
        droprate=0,
        struct_spans=4,
        label_spans=3,
        device='cuda',
    ):
        super(Network, self).__init__()
        self.word_count = word_count
        self.tag_count = tag_count
        self.word_dims = word_dims
        self.tag_dims = tag_dims
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.struct_out = struct_out
        self.label_out = label_out

        self.droprate = droprate

        self.activation = nn.ReLU()
        self.word_embed = nn.Embedding(word_count, word_dims)
        self.tag_embed = nn.Embedding(tag_count, tag_dims)

        # self.lstm = nn.LSTM(word_dims + tag_dims, lstm_units, num_layers=2, bidirectional=True)
        self.fwd_lstm1 = CH_LSTM(word_dims + tag_dims, lstm_units)
        self.back_lstm1 = CH_LSTM(word_dims + tag_dims, lstm_units, backward=True)

        self.fwd_lstm2 = CH_LSTM(2*lstm_units, lstm_units)
        self.back_lstm2 = CH_LSTM(2*lstm_units, lstm_units, backward=True)

        # self.lstm1 = nn.LSTM(word_dims + tag_dims, lstm_units, num_layers=1, bidirectional=True)
        # self.lstm2 = nn.LSTM(2*lstm_units, lstm_units, num_layers=1, bidirectional=True)

        self.struct_hidden_linear = nn.Linear(4*struct_spans * lstm_units, hidden_units)
        self.struct_output_linear = nn.Linear(hidden_units, struct_out)

        self.label_hidden_linear = nn.Linear(4*label_spans * lstm_units, hidden_units)
        self.label_output_linear = nn.Linear(hidden_units, label_out)

        self.dropout = nn.Dropout(droprate)

        self.device = device
        
    def load_dynet(self, path):
        with torch.no_grad():
            for attr in ['fwd_lstm1', 'back_lstm1', 'fwd_lstm2', 'back_lstm2']:
                module = getattr(self, attr)
                module.c0 = load_npz(os.path.join(path, attr, 'c0.npz'))
                for gate in ['i', 'o', 'f', 'c']:
                    setattr(getattr(module, 'W_' + gate), 'weight', load_npz(os.path.join(path, attr, 'W_' + gate + '.npz')))
                    setattr(getattr(module, 'W_' + gate), 'bias', load_npz(os.path.join(path, attr, 'b_' + gate + '.npz')))
            self.struct_hidden_linear.weight = load_npz(os.path.join(path, 'struct_hidden_W.npz'))
            self.struct_hidden_linear.bias = load_npz(os.path.join(path, 'struct_hidden_b.npz'))
            self.struct_output_linear.weight = load_npz(os.path.join(path, 'struct_output_W.npz'))
            self.struct_output_linear.bias = load_npz(os.path.join(path, 'struct_output_b.npz'))
            self.label_hidden_linear.weight = load_npz(os.path.join(path, 'label_hidden_W.npz'))
            self.label_hidden_linear.bias = load_npz(os.path.join(path, 'label_hidden_b.npz'))
            self.label_output_linear.weight = load_npz(os.path.join(path, 'label_output_W.npz'))
            self.label_output_linear.bias = load_npz(os.path.join(path, 'label_output_b.npz'))
    

    def evaluate_recurrent(self, word_inds, tag_inds, lengths):
        assert lengths.max() == word_inds.shape[0]
        # word_inds and tag_inds are seq x batch, lengths is batch
        word_embeddings = self.word_embed(word_inds)
        tag_embeddings = self.tag_embed(tag_inds)
        sentence = torch.cat([word_embeddings, tag_embeddings], dim=2) # seq x batch x word+tag

        fwd1 = self.fwd_lstm1(sentence, lengths)
        back1 = self.back_lstm1(sentence, lengths)
        # sentence = pack_padded_sequence(sentence, lengths, enforce_sorted=False)

        # lstm1_output, _ = self.lstm1(sentence)
        # padded_lstm1_output, _ = pad_packed_sequence(lstm1_output)
        # fwd1, back1 = padded_lstm1_output.view(padded_lstm1_output.shape[0], word_inds.shape[1], 2, self.lstm_units)[:, :, 0, :], \
        #               padded_lstm1_output.view(padded_lstm1_output.shape[0], word_inds.shape[1], 2, self.lstm_units)[:, :, 1, :]

        

        lstm2_in = self.dropout(torch.cat([fwd1, back1], dim=2))

        fwd2 = self.fwd_lstm2(lstm2_in, lengths)
        back2 = self.back_lstm2(lstm2_in, lengths)

        # lstm2_input = pack_padded_sequence(self.dropout(padded_lstm1_output), lengths, enforce_sorted=False)

        # lstm2_output, _ = self.lstm2(lstm2_input)
        # padded_lstm2_output, _ = pad_packed_sequence(lstm2_output)
        # fwd2, back2 = padded_lstm2_output.view(padded_lstm2_output.shape[0], word_inds.shape[1], 2, self.lstm_units)[:, :, 0, :], \
        #               padded_lstm2_output.view(padded_lstm2_output.shape[0], word_inds.shape[1], 2, self.lstm_units)[:, :, 1, :]

        fwd_out = torch.cat([fwd1, fwd2], dim=2) 
        back_out = torch.cat([back1, back2], dim=2)

        return fwd_out, back_out # each seq x batch x 2*lstm
    
    def evaluate_struct(self, fwd_out, back_out, lefts, rights):
        # fwd_out, back_out each seq x batch x 2*lstm
        # lefts, rights each batch x 4
        # lefts, rights = np.array(lefts), np.array(rights) # TODO convert to torch outside of this method
        lefts, rights = lefts.unsqueeze(2).expand(-1, -1, 2*self.lstm_units), \
                        rights.unsqueeze(2).expand(-1, -1, 2*self.lstm_units)
        fwd_out, back_out = fwd_out.permute(1, 0, 2), back_out.permute(1, 0, 2)
        fwd_span_vec = torch.gather(fwd_out, 1, rights) - torch.gather(fwd_out, 1, lefts - 1)
        back_span_vec = torch.gather(back_out, 1, lefts) - torch.gather(back_out, 1, rights + 1) # each 4 x batch x 2*lstm

        hidden_input = torch.cat([fwd_span_vec, back_span_vec], dim=2).flatten(1)
        hidden_input = self.dropout(hidden_input)

        hidden_output = self.activation(self.struct_hidden_linear(hidden_input))
        scores = self.struct_output_linear(hidden_output)

        return scores # batch x struct_out

    def evaluate_label(self, fwd_out, back_out, lefts, rights):
        # fwd_out, back_out each seq x batch x 2*lstm
        # lefts, rights each batch x 3
        # lefts, rights = np.array(lefts), np.array(rights) # TODO convert to torch outside of this method
        lefts, rights = lefts.unsqueeze(2).expand(-1, -1, 2*self.lstm_units), \
                        rights.unsqueeze(2).expand(-1, -1, 2*self.lstm_units)
        fwd_out, back_out = fwd_out.permute(1, 0, 2), back_out.permute(1, 0, 2)
        fwd_span_vec = torch.gather(fwd_out, 1, rights) - torch.gather(fwd_out, 1, lefts - 1)
        back_span_vec = torch.gather(back_out, 1, lefts) - torch.gather(back_out, 1, rights + 1) # each 3 x batch x 2*lstm

        hidden_input = torch.cat([fwd_span_vec, back_span_vec], dim=2).flatten(1)
        hidden_input = self.dropout(hidden_input)

        hidden_output = self.activation(self.label_hidden_linear(hidden_input))
        scores = self.label_output_linear(hidden_output)

        return scores # batch x label_out