import torch
import torch.nn as nn
import torch.nn.functional as F


class CH_LSTM(nn.Module):
    def __init__(self, input_dims, output_dims, backward=False):
        super(CH_LSTM, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.backward = backward

        self.W_i = nn.Linear(input_dims + output_dims, output_dims)
        self.W_f = nn.Linear(input_dims + output_dims, output_dims)
        self.W_c = nn.Linear(input_dims + output_dims, output_dims)
        self.W_o = nn.Linear(input_dims + output_dims, output_dims)
        self.c0 = nn.Parameter(torch.zeros(output_dims))
    
    def initial_state(self, batch_size):
        c0 = self.c0.unsqueeze(0).repeat(batch_size, 1)
        return F.tanh(c0), c0
    
    def step(self, input_vec, state): # input is batch x input_dim
        h, c = state
        x = torch.cat([input_vec, h], dim=1)
        i = torch.sigmoid(self.W_i(x))
        f = torch.sigmoid(self.W_f(x))
        g = F.tanh(self.W_c(x))
        o = torch.sigmoid(self.W_o(x))

        c = f * c + i * g
        h = o * F.tanh(c)
        return h, (h, c)
    
    def forward(self, batch, lengths): # seq x batch x input
        state = self.initial_state(len(lengths))
        outputs = []
        max_length = lengths.max()
        for i in range(batch.size(0)):
            if self.backward:
                input_vec = batch[(lengths - 1 - i) % max_length, torch.arange(len(lengths)).to(batch.device)]
            else:
                input_vec = batch[i] # batch x input
            new_output, state = self.step(input_vec, state)
            mask_indices = (i >= lengths).nonzero().flatten()
            new_output[mask_indices] = 0
            outputs.append(new_output)
        if self.backward:
            stacked = torch.stack(outputs, dim=0).flip(0) # seq x batch x output
            offset = max_length - lengths # batch
            select = (torch.arange(max_length).to(batch.device).unsqueeze(1) + offset.unsqueeze(0)) % max_length # seq x batch
            return torch.gather(stacked, 0, select.unsqueeze(2).repeat(1, 1, stacked.size(2)))
        else:
            return torch.stack(outputs, dim=0) # seq x batch x output
        # note final hidden states not needed for our purposes
