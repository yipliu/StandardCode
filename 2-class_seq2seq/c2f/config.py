"""
Base configuration class

Reference from RS
"""

import os
import multiprocessing

import torch

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change

class Config(object):
    """Base configuration class. For custom configurations, create a 
    sub-class that inherits from this one and override properties
    that need to be changed
    """
    # -------------------------------
    # Basics of the configurations.
    # --------------------------------
    NAME = None             # Override in sub-classes
    MODE = 'train'          # Mode (train/eval)
    ROOT_DIR = None         # Root project directory, Override in sub-classes
    CHECKPOINT_PATH = os.path.join('chechpoints') # Save run path
    DATASET_PATH = os.path.join('data') # Path to dataset files

    # ------------------------
    # Parameters for NLP
    # ------------------------

    # Special tokes to be add into vocabulary
    START_WORD = '<sos>'
    END_WORD   = '<eos>'
    UNK_WORD   = None

    # Maximum sequence length allowed
    MAXLEN     = 10

    # Word frequency
    FREQUENCY = None

    # -------------------------------------
    # Parameters for dataset configuration
    # -------------------------------------
    
    # Size for Vocabulary 
    SRC_VOCAB_SIZE = None       
    TRG_VOCAB_SIZE = None 
    SRC_PAD_IDX = None
    TRG_PAD_IDX = None

    # Dataset
    TRAIN_NAME = 'train.json'
    TEST_NAME = 'test.json'
    TASK = 'geoqueries'

    # -------------------------
    # Parameters for Advance
    # -------------------------
    K_FOLD = False   # Whether to use Cross_Validation
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    RNN_TYPE = 'LSTM'

    BATCH_SIZE = 12
    METHOD = 'general'
    # --------------------------------------------
    # Hyperparameters for Encoder Initialization
    # --------------------------------------------
    BIDRECTIONAL = True # Wheter to use birectional in Encoder
    N_LAYERS = 1


    # Input Dimension. This is equal to the size of vocab_size
    INPUT_DIM = SRC_VOCAB_SIZE 

    # Encoder embedding dim  
    ENC_EMB_DIM = 50

    # Encoder hidden dim
    ENC_HID_DIM = 24

    # Encoder dropout
    E_DROPOUT =  0.5



    # --------------------------------------------
    # Hyperparameters for Decoder Initialization
    # ---------------------------------------------
    N_LAYERS = 1
    OUTPUT_DIM = TRG_VOCAB_SIZE
    DEC_EMB_DIM = 50
    DEC_HID_DIM = 24
    D_DROPOUT = 0.5


    # -------------------------------------
    # Hyperparameters for Model Train
    # -------------------------------------
 
    # Epochs
    NUM_EPOCHS = 50

    # Learning rate
    LEARNING_RATE = 1e-4

    # Learning rate decay, decay by 0.1 every ? epoch
    LR_DECAY_EVERY = [5, 40]

    # Whether to clip gradient or not
    CLIP_NORM = 5.0  # None

    # Weight decay
    WEIGHT_DECAY = 1e-4

    # Teacher forcing
    TEACHER_FORCING_RATIO = 0.5


    # ----------------------
    # Training Parameters
    # Display every ? steps
    DISPLAY_EVERY = 20

    # Save model every ? epoch
    SAVE_EVERY = 1





    # # -------------------
    # # Parameters in c2f
    # RNN_SIZE = 300
    # WORD_VEC_SIZE = 150
    # DECODER_INPUT_SIZE 150
    
    # # Model
    # DECODER_INPUT_SIZE = 150 # layout embedding size
    # N_LAYERS = 1

    # learning_rate = 0.005
    # start_decay_at = 0
    # epochs = 100
    # RNN_TYPE = 'LSTM'



    def __init__(self):
        """Set values of computed attributes."""
        # Workers used for dataset object...
        if os.name == 'nt':
            self.WORKERS = 0
        else:
            self.WORKERS = multiprocessing.cpu_count()
        
        # Some force assertion
        assert self.MODE in ['train', 'eval']
        assert self.TASK in ['atis','django', 'geoqueries', 'wikiql']
        assert self.RNN_TYPE in ['LSTM', 'GRU']
        assert self.METHOD in ['dot', 'general', 'concat']

    def display(self):
        """Display Configuration values."""
        print("-"*30)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print()
