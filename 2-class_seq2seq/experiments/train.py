"""
Training script
"""
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator, Iterator
import numpy as np
from termcolor import colored

# Root directory of the project
ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)

import preprocess
from c2f.config import Config
from c2f.lang_class import Lang
from c2f.utilis import init_weights_base, count_parameters


# Configuration for hperparameters
class TrainConfig(Config):
    """Configuration for training 
    """
    NAME = 'c2f'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'data','geoqueries')
    TASKS = 'geoqueries'
    K_FOLD = False
    N_EPOCHS = 10
    
def main():    
    # Setup configuration class
    config = TrainConfig()

    # Load dataset iterator
    train_iter, test_iter, config = preprocess.prepare_data(config)
    config.display()


    # Setup and build coarse2fine training inference
    attn = Attention
    enc = Encoder
    dec = Decoder

    model = Seq2Seq(enc, dec,).to(config.DEVICE)

    # Initialize network -> Load the pretrained embeddings onto our model
    ## pretrained_embeddings = quote.vocab.vectors
    ## model.embedding.weight.data.copy_(pretrained_embeddings)


    # initialize the model to a special initialization, and calculate the trainable parameter
    model.apply(init_weights_base)
    print(colored(f'The model has {count_parameters(model):,} trainable parameters'),'red')



    # Initialize the loss function and create an optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=config.TRG_PAD_IDX)
    optimizer = optim.Adam(model.parameters())


    # Save vocabulary at last

    # Start training
    if config.K_FOLD:
        pass
    else:
        for epoch in range(N_EPOCHS):
            best_model, best_epach = 



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print('[STOP]', e)