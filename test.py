# import os
# import time
# for i in range(1000000):
#     time.sleep(1)
#     print(i)

import torch.nn as nn
import torch
class Rnn(nn.Module):
    def __init__(self,input_size=196):
        super(Rnn, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=20,
            num_layers=1,
            batch_first=False
        )
        # num_layers = 1,
        # batch_first = True


    def forward(self, x,h_n,h_c):
        r_out, (h_n, h_c) = self.rnn(input, (h_n, h_c))  # None 表示 hidden state 会用全0的 state
        return r_out,h_n, h_c
lstm = nn.LSTM(196, 20)#[feature_len,hidden_len,num_layers]
rnn = nn.LSTM(
            input_size=196,
            hidden_size=20,
            num_layers=1,
            batch_first=False
        )
input = torch.randn((18, 5, 196),device="cpu")
h0 = (torch.randn(1, 5, 20))
c0 = (torch.randn(1, 5, 20))
output, hn = lstm(input, (h0, c0))
r_out, (h_n, h_c) = rnn(input, (h0,h0))
R=Rnn(input_size=196)
a,b,c=R(input,h0,c0)


