"""
Utility variables, functions and classes
"""

import os
import sys
import time

import torch
import torch.nn as nn
from tqdm import tqdm

# ------------------------------
# Utility functions for Model
# ------------------------------

def count_parameters(model):
    """Calculate the number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Initializing all biases to zero and all weights from N(0, 0.01)
def init_weights_base(m):
    """A simplified version of the weight initialization
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def epoch_time(start_time, end_time):
    """Calcuting time
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# ---------------------------------------------
# Utility functions for Model Train and Test
# ---------------------------------------------

def model_train(model, iterator, optimizer, criterion, epoch, config):
    """Train the Seq2Seq model: One train step to calculte the train loss
    """
    # Set the model into "training mode"
    model.train()
    epoch_loss = 0

    for i, batch in iterator:
        """
        SRC =  Field(include_lengths = true), it makes batch.src = (tensor, the length)
        """
        # src -> tensor
        # src_len -> Its length
        src, src_len = batch.Src
        
        trg = batch.Trg

        optimizer.zero_grad()

        output = model(src, src_len, trg) 
        """
        src = [src_len, batch_size]
        src_len [batch_size]
        trg = [trg_len, batch_size]
        
        output = [trg_len, batch_size, output_dim]
        """

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters of our model
        optimizer.step()

        epoch_loss += loss.item()

        average_loss = epoch_loss / iterator.total

        # update progress bar
        iterator.set_description(f"Epoch [{epoch+1:02}/{num_epochs}]")
        iterator.set_postfix(loss = average_loss)

    return average_loss

def model_evaluate(model,):
    """
    """



def epoch_train(model, train_iter, val_iter, optimizer, criterion, epoch, config):
    """
    """

    best_valid_loss = config.BEST_VALID_LOSS

    start_time = time.time()
    loop_train = tqdm(enumerate(train_iter), total=len(train_iter))
    
    train_loss = model_train(model, loop_train, optimizer, criterion, epoch, config)

    loop_valid = tqdm(enumerate(val_iter), total=len(val_iter))
    valid_loss = model_evaluate(model, loop_valid, criterion, epoch, n_epochs)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        config.BEST_VALID_LOSS = valid_loss
        model_best = model
        epoch_best = epoch

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    return(model_best, epoch_best,best_valid_loss)



