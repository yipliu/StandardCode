import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)

from c2f.config import Config

# Configuration for Encoder test
class TestEncoderConfig(Config):
    """ Configuration for testing Encoder
    """
    NAME = 'Encoder_Test'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR

    RNN_TYPE = 'LSTM'
    BIDRECTIONAL = True
    N_LAYERS = 1


    BATCH_SIZE = 12
    INPUT_DIM = 30
    ENC_EMB_DIM = 50
    ENC_HID_DIM = 24
    E_DROPOUT =  0.5
    DEC_HID_DIM = 24


class Encoder(nn.Module):
    """Module to encoder Language
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        """
        A bidirectional RNN: -> Two RNNs in each layer
        """
        self.hidden_size = config.ENC_HID_DIM
        self.embeddings = nn.Embedding(config.INPUT_DIM, config.ENC_EMB_DIM)

        #assert rnn_type == 'LSTM' or 'GRU'

        self.rnn = getattr(nn, config.RNN_TYPE)(config.ENC_EMB_DIM, config.ENC_HID_DIM, config.N_LAYERS, 
                             dropout=(0 if config.N_LAYERS ==1 else config.E_DROPOUT), bidirectional=config.BIDRECTIONAL)

        # self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, 
        #                     dropout=(0 if n_layers ==1 else dropout), bidirectional=bidrectional)

        """
        input_size  -> The number of expected features in the input x
        hidden_size -> The number of features in the hidden state h
        n_layers    -> Number of recurrent layers. Default is 1
        dropout     -> If non-zero, introduces a Dropout layer on the outputs of each LSTM except the last layer.Default 0 
        """
        self.fc = nn.Linear(config.ENC_HID_DIM * 2, config.DEC_HID_DIM)
        self.dropout = nn.Dropout(config.E_DROPOUT)

    def forward(self, input, src_len):# input: [src_len, Batch_size], src_len:[Batch_size]

        # STEP 1: Convert word indexes to embeddings
        embedded = self.dropout(self.embeddings(input)) # [src_len, B, emb_dim]

        # STEP 2: Pack padded batch of sequences for RNN module
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu())

        # STEP 3: Forward pass through RNN
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        """
        packed_outputs: a packed tensor containing all of the hidden states from the sequence
        hidden: the final hidden state from our sequence
                This is a standard tensor and not packed in anyway

        Initial hidden state h0 to a tensor of all zeros

        output = [seq_len, batch_size, num_directions*hidden_size]
        h_n = [num_layers*num_directions, batch_size, hidden_size]
        c_n = [num_layers*numdirections, batch_size, hidder_size]
        """
        
        # STEP 4: Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        """
        outputs is now a non-packed sequence

        outputs = [src_len, batch_size, enc_hid_dim * num_directions]


        hidden[-2, :, :] is the last of the forwards RNN
        hidden[-1, :, :] is the last of the backwards RNN
        """
        # STEP 5: Return outputs(All encoder hidden states) and the final hidden state which is used to initialize the RNN in Decoder
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))) # [batch_size, dec_hid_dim]
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1)))
        return outputs, (hidden, cell)




"""
A bidirectional RNN

We have two RNNs in each layer:
- A forward RNN going over embeded sentence from left to right   -->
- A backward RNN going over embeded sentence from right to left  <--

# How to initialize the RNN in Decoder
- In the original paper, they use

$$s_{0}=\tanh \left(W_{s} \overleftarrow{h}_{1}\right)$$

- In bentrevett, he use

$$s_{0}=\tanh \left(g\left(h \vec{T}, h_{T}^{\leftarrow}\right)\right)$$

"""

if __name__ == '__main__':
    # --------------------
    # Test for encoder
    # --------------------
    config = TestEncoderConfig()
    batch_size = config.BATCH_SIZE

    src_len = 7
    encoder = Encoder(config)
    x = torch.LongTensor(src_len, batch_size).random_(1, 10) # [7, 12]

    x_len = torch.full([batch_size],7) # [12]
    encoder(x, x_len)

    for n, p in encoder.named_parameters():
        print(n, p.data.dtype, p.shape)