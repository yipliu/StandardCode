"""
Script for data preprocess

STEPS:
1. Specify how preprocessing should be done -> Fields
2. Use Dataset to loac the data             -> TabularDataset (JSON/CSV/TSV Files)
3. Construct an iterator to do batching & padding  -> BucketIterator
"""
import os
import sys

import torch
from torchtext.data import BucketIterator, Iterator

# Root directory of this project
ROOT_DIR = os.path.abspath("")  # .../my_c2f

# Import C2f

# To find local version of the library
sys.path.append(ROOT_DIR)
from c2f.config import Config
from c2f.lang_class import Lang

class DataConfig(Config):
    """Configuration for Data
    """
    NAME     = "Data_preprocess" 
    ROOT_DIR = ROOT_DIR
    DATASET_PATH =  os.path.join(ROOT_DIR, 'data', 'geoqueries')
    #WEIGHTS_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    BATCH_SIZE = 256

    TRAIN_NAME = 'train.json'
    TEST_NAME = 'test.json'


def create_folder(path):
    if not os.path.exists(path):
        os.makedires(path)

def prepare_data(config):
    """Helper function to prepaer training/evaluating

    config: Configuration class
    vocab: Vocabulary class
    """

    lang = Lang(config.DATASET_PATH, config.TRAIN_NAME, config.TEST_NAME)
    train_data, test_data = lang.load_data()

    print(vars(train_data[0]))
    
    # Build vocab
    lang.fields_vocab(train_data)

    INPUT_DIM, OUTPUT_DIM = len(lang.SRC.vocab), len(lang.TRG.vocab)


    print("The length of src_vocab is {}; \n The length of trg_vocab is {}".\
            format(INPUT_DIM, OUTPUT_DIM))




    torch.save(train_data.fields, 'fields.pkl')
    print('The Field: fields.pkl is stored in local')

    if config.K_FOLD:
        pass
    else:
        train_iter, test_iter = BucketIterator.splits(
                                                    (train_data, test_data),
                                                    sort_within_batch = True,
                                                    sort_key = lambda x: len(x.Src),
                                                    batch_size = config.BATCH_SIZE,
                                                    device = config.DEVICE
                                                    )
        """
        for batch in train_iter:
            print(batch.Src)
        
        All elements in the batch need to be sorted by their non-padded lengths in descending order.
        I.e. the first sentence in the batch needs to be the longest 

        sort_within_batch: Tells the iterator that the contents of the batch need to be sorted

        sort_key: Tells the iterator how to sort the elements in the batch.
         Here, we sort by the length of src sentence
        """



    # Reset some key parameters inside config before going futher
    # Reset vocab_size
    config.SRC_VOCAB_SIZE, config.TRG_VOCAB_SIZE = len(lang.SRC.vocab), len(lang.TRG.vocab)
    config.SRC_PAD_IDX, config.TRG_PAD_IDX = lang.SRC.vocab.stoi[lang.TRG.pad_token], lang.SRC.vocab.stoi[lang.SRC.pad_token]

    return train_iter, test_iter, config




        
def main():
    config = DataConfig()
    train_iter, test_iter = prepare_data(config)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print('[STOP]', e)