import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)

from c2f.config import Config

# ------------------------------------
# Additive Attention
# ------------------------------------

class BahdanauAttention(nn.Module):
    """Bahdanau Additive Attention.
    """

    def __init__(self, config):
        super(BahdanauAttention, self).__init__()

        self.attn = nn.Linear((config.ENC_HID_DIM * 2) + config.DEC_HID_DIM, config.DEC_HID_DIM)
        self.v = nn.Linear(config.DEC_HID_DIM, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        """
        Input: 
        hidden=[1, batch_size, dec_hid_dim] -> the previous hidden state in Decoder
                                    h_0 is the vector h_T in Encoder. The last hidden state in Encoder
        
        encoder_outputs=[src_len, batch_size, enc_hid_dim *2 ] -> All hidden state in Encoder

        Output: 
              A score

        Aim: To calculate the score, in order to get the Context vector in the Decoder.
        """
        # STEP 1. Claculating Energy. $e_{t} = a(s_{t-1}, H)
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        # hidden = [src_len, batch_size, dec_hid_dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [src_len, batch_size, dec_hid_dim]

        # STEP 2. Claculating Score
        attention = self.v(energy).squeeze(2).permute(1,0)
        # attention = [batch_size, src_len]
        
        # Ensuring no attention is payed to padding tokens in the source sentence
        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


# -----------------------------
# Multiplicative Attention
# -----------------------------
class LuongGlobalAttention(nn.Module):
    """Luong Mutiplicative Attention(Global)
    """
    def __init__(self, config):
        super(LuongGlobalAttention, self).__init__()

        self.config = config

        if self.config.METHOD == "general":
            self.attn = nn.Linear(config.ENC_HID_DIM *2, config.ENC_HID_DIM *2)

        elif self.config.METHOD == 'concat':
            self.attn = nn.Linear(config.ENC_HID_DIM *2+config.DEC_HID_DIM, config.ENC_HID_DIM *2)
            self.v = nn.Parameter(torch.FloatTensor(config.ENC_HID_DIM*2))

    # This is done in Pytorch Tutorials
    def general_score(self, hidden, encoder_outputs):
        
        #hidden: [1, batch_size, dec_hid_dim] -> The current target hidden state
        #encoder_outputs: [src_len, batch_size, enc_hid_dim * 2] -> All hidden state in Encoder
       
        energy = self.attn(encoder_outputs) # energy = [src_len, batch_size, enc_hid_dim *2]
        return torch.sum(hidden.repeat(1,1,2 if self.config.BIDRECTIONAL else 1) * energy, dim=2) #[src_len, batch_size]

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden.repeat(1,1,2 if self.config.BIDRECTIONAL else 1) * encoder_output, dim=2)

   
    def forward(self, hidden, encoder_outputs, mask):
        """
        # hidden: The current target hidden state    [1, batch_size, dec_hid_dim]
        # encoder_outputs: All encoder hidden state  [src_len, batch_size, enc_hid_dim * 2]
        """
        # Calculate the attention weights (energies) based on the given method
        if self.config.METHOD == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.config.METHOD == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.config.METHOD == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions [src_len, batch_size] -> [batch_size, src_len]
        attn_energies = attn_energies.t()

        # Ensuring no attention is payed to padding tokens in the source sentence
        attn_energies = attn_energies.masked_fill(mask == 0, -1e10)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


