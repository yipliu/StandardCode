import os
import sys

import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)

from c2f.config import Config
from attettion import *

# Configuration for Decoder test
class TestDecoderConfig(Config):
    """ Configuration for testing Decoder
    """
    NAME = 'Decoder_Test'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR

    RNN_TYPE = 'LSTM'
    BIDRECTIONAL = True
    N_LAYERS = 1


    BATCH_SIZE = 12
    OUTPUT_DIM = 15
    ENC_EMB_DIM = 50
    ENC_HID_DIM = 24
    E_DROPOUT =  0.5

"""
Unidirectional LSTM in Decoder

In Decoder, we will feed the decoder one word at a time
"""
class Decoder(nn.Module):
    """
    """
    def __init__(self, output_dim, attention):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.attention = attention

    def forward(self, **kwargs):
        """Must be overrided
        """
        raise NotImplementedError



class BahdanauDecoder(nn.Module):
    """Module to decode features and generate word for sentence sequence using Bahdanau Attention
    """
    def __init__(self, config, attention):
        super().__init__()
        self.attention = attention

        # Define layers
        self.embedding = nn. Embedding(config.OUTPUT_DIM, config.DEC_EMB_DIM)
        #self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.rnn = getattr(nn, config.RNN_TYPE)(config.ENC_HID_DIM * 2 + config.ENC_EMB_DIM, config.DEC_HID_DIM, 
                           config.N_LAYERS, dropout=(0 if config.N_LAYERS ==1 else config.E_DROPOUT), bidirectional=False)
        
        self.fc_out = nn.Linear((config.ENC_HID_DIM * 2) + 
                                 config.DEC_HID_DIM + config.DEC_EMB_DIM, config.OUTPUT_DIM)

        self.dropout = nn.Dropout(config.D_DROPOUT)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        """
        input = [batch_size]
        hidden = [batch_size, dec_hid_dim]
        encoder_outputs = [src_len, batch_size, enc_hid_dim]
        mask = [batch_size, src_len]

        Note: we run this one step (word) at a time
        """
        input = input.unsqueeze(0) # input = [1, batch_size]

        embedded = self.dropout(self.embedding(input)) # embedded = [1, batch_size, emb_dim]

        # Get the Score
        attn_weights = self.attention(hidden, encoder_outputs, mask).unsqueeze(1) # [batch_size, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2) #[batch_size, src_len, enc_hid_dim *2]

        # Context vector C
        context = torch.bmm(attn_weights, encoder_outputs).permute(1, 0, 2) # [1, batch_size, enc_hid_dim*2]
        
        # The input of the  prediction model
        rnn_input = torch.cat((embedded, context), dim=2) # [1, batch_size, (enc_hid_dim*2) + emb_dim]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        """
        output = [seq_len, batch_size, dec_hid_dim * n_directions]
        hidden = [n_layers * n_directions, batch_size, dec_hid_dim]

        seq_len, n_layers and n_directions will always be 1 in this decoder, therefore
        
        output == hidden
        """
        assert (output == hidden).all()

        embedded = embedded.squeeze(0) #[1, B, DEC_EMB_DIM] -> [B, DEC_EMB_DIM]
        output = output.squeeze(0)     # [1, B, DEC_HID_DIM] -> [B, DEC_HID_DIM]
        context = context.squeeze(0)   # [1, B, ENC_HID_DIM * 2] -> [B, ENC_HID_DIM *2]

        # The output of the prediction model
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1)) # prediction = [batch_size, output_dim]

        return prediction, (hidden.squeeze(0), cell.squeeze(0)), attn_weights.squeeze(1) # attn_weights = [batch_size, enc_hid_dim]


# -------------------------------
# Decoder with Luong Attention
# -------------------------------

class LuongDecoder(nn.Module):
    """Module to decode features and generate word for sentence sequence using Luong Attention
    """
    def __init__(self, config, attention):
        super(LuongDecoder, self).__init__()

        self.attention = attention

        # Define Layers
        self.embedding = nn. Embedding(config.OUTPUT_DIM, config.DEC_EMB_DIM)
        """
        nn.Embedding:
        1 -> the size of the dictionary embeddings
        2 -> the size of each embedding vector 
        """
        #self.rnn = nn.LSTM(emb_dim, dec_hid_dim)
        self.rnn = getattr(nn, config.RNN_TYPE)(config.ENC_EMB_DIM, config.DEC_HID_DIM, 
                           config.N_LAYERS, dropout=(0 if config.N_LAYERS ==1 else config.E_DROPOUT), bidirectional=False)
        """
        nn.LSTM
        1 -> The number of expected features in the input
        2 -> The number of features in the hidden state h
        """
        self.concat = nn.Linear(config.ENC_HID_DIM*2 + config.DEC_HID_DIM, config.DEC_HID_DIM)
        self.fc_out = nn.Linear(config.DEC_HID_DIM, config.OUTPUT_DIM)

        self.dropout = nn.Dropout(config.D_DROPOUT)
    def forward(self, input_step, hidden, cell, encoder_outputs, mask):
        """
        input_step: [batch_size]

        last_hidden: s0 is the last hidden state in Encoder [B, dec_hid_dim]
        encoder_outputs: [src_len, batch_size, enc_hid_dim * 2]
        
        Note: we run this one step (word) at a time
        """
        input_step = input_step.unsqueeze(0) # input = [1, batch_size]

        # STEP 1: Get embedding of current input word
        embedded = self.dropout(self.embedding(input_step)) # embedded = [1, batch_size, emb_dim]

        # STEP 2: Forward through RNN
        output, (hidden, cell) = self.rnn(embedded, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        """
        outputs = [seq_len, batch_size, num_directions*hidden_size] = [1, B, hid_size]
        hidden = [n_lays*num_directions, batch_size, hidden_size]   = [1, B, hid_size]

        RNN:
        1 -> The input tensor
        2 -> h_0 the initial hidden state
        """
        assert (output == hidden).all()

        # STEP 3: Calculate attention weights
        attn_weights = self.attention(hidden, encoder_outputs, mask) # [batch_size, 1,src_len]

        # STEP 4: To get context vector
        context = torch.bmm(attn_weights, encoder_outputs.permute(1, 0, 2)) # [batch_size, 1, enc_hid_dim * 2]

        # STEP 5: To get the attentional hidden state using eq. 5
        output = output.squeeze(0)    # [B, dec_hid_dim]
        context = context.squeeze(1)  # [B, enc_hid_dim * 2]
        concat_input = torch.cat((output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input)) # [B, dec_hid_dim]

        # STEP 6: Predict next word using Luong eq. 6
        prediction = F.softmax(self.fc_out(concat_output), dim=1)
        
        # Return output and final RNN hidden state
        return prediction, (hidden.squeeze(0), cell.squeeze(0)), attn_weights.squeeze(1)
        """
        prediction: [B, dec_hid_dim]
        hidden.squeeze(0): [B, dec_hid_dim]
        attn_weights.squeeze(1): [B, src_len]
        """

        
