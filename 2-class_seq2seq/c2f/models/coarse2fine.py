import os
import sys
import random

import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)

from c2f.config import Config
from c2f.models.encoder import *
from c2f.models.decoder import *
from c2f.models.attettion import *


class TestSeqSeqConfig(Config):
    """ Configuration for testing Encoder
    """
    NAME = 'Encoder_Test'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR

    RNN_TYPE = 'LSTM'
    BIDRECTIONAL = True
    N_LAYERS = 1

    SRC_PAD_IDX = 1


    BATCH_SIZE = 12
    INPUT_DIM = 30
    OUTPUT_DIM = 20
    DEC_EMB_DIM = ENC_EMB_DIM = 50
    DEC_HID_DIM = ENC_HID_DIM = 24
    D_DROPOUT = E_DROPOUT =  0.5


class Seq2Seq(nn.Module):
    """Train/Eval inference class for C2F model
    """
    def __init__(self, config, encoder, decoder):
        super().__init__()

        self.config = config
        self.encoder = encoder
        self.decoder = decoder
    
    def create_mask(self, src):
        mask = (src != self.config.SRC_PAD_IDX).permute(1, 0)
        return mask
    
    def forward(self, src, src_len, trg):
        """
        src = [src_len, batch_size]
        src_len = [batch_size]

        trg = [trg_len, batch_size]
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]

        trg_vocab_size = self.config.OUTPUT_DIM

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,trg_vocab_size).to(self.config.DEVICE)

        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder(src, src_len)
        """
        encoder_outputs: [src_len, Batch_size, enc_hid_dim * 2]
        hidden: [Batch_size, DEC_HID_DIM]
        """
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]

        mask = self.create_mask(src) #[batch_size, src_len]

        for t in range(1, trg_len):

            output, (hidden, cell), _ = self.decoder(input, hidden, cell, encoder_outputs, mask)

            # Place predictions in a tensor holding predicitions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.config.TEACHER_FORCING_RATIO

            # get the hightest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token

            input = trg[t] if teacher_force else top1
        
        return outputs

if __name__ == '__main__':
    # --------------------
    # Test for SeqSeq
    # --------------------
    config = TestSeqSeqConfig()

    enc = Encoder(config)
    #atten = BahdanauAttention(config)
    #dec = BahdanauDecoder(config, atten)
    
    atten = LuongGlobalAttention(config)
    dec = LuongDecoder(config, atten)


    model = Seq2Seq(config, enc, dec)
    
    batch_size = config.BATCH_SIZE

    src_len = 7
    trg_len = 5

    x = torch.LongTensor(src_len, batch_size).random_(1, 10) # [7, 12]
    y = torch.LongTensor(trg_len, batch_size).random_(1, 10) # [5, 12]

    x_len = torch.full([batch_size],7) # [12]
    
    output = model(x, x_len, y)

    for n, p in model.named_parameters():
        print(n, p.data.dtype, p.shape)    