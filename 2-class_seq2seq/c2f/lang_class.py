"""
Class for language
"""
import os
import sys

ROOT_DIR = os.path.abspath("")

sys.path.append(ROOT_DIR)

import torch
import numpy as np
import pandas as pd
from torchtext.data import Field, TabularDataset, Dataset, BucketIterator
from sklearn.model_selection import KFold

from joblib import Memory

class Lang():
    """Language Class
    """
    def __init__(self, dataset_path, train_name, test_name):
        self.dataset_path = dataset_path
        self.train_name = train_name
        self.test_name = test_name

        self.SRC = Field(init_token = '<sos>',
                        eos_token = '<eso>',
                        lower = True,
                        include_lengths = True)
        
        # include_lengths: this will cause 'batch.src' to be a tuple
        # 1. a batch of numericalized source sentence as a tensor
        # 2. the second element is the non-padded lengths of each source sentence within the batch
        
        self.TRG = Field(init_token='<sos>',
                        eos_token='<eso>',
                        lower=True)
    
    def load_data(self):
        """
        """
        fields = {'src': ('Src',self.SRC), 'tgt':('Trg',self.TRG)}

        train_data, test_data = TabularDataset.splits(
                                                path=self.dataset_path,
                                                train=self.train_name,
                                                test=self.test_name,
                                                format='json',
                                                fields = fields)

        # print(train_data[0].__dict__.keys())
        # print(train_data[0].__dict__.values())
        return train_data, test_data


    def fields_vocab(self, dataset):
        self.SRC.build_vocab(dataset, vectors = "glove.6B.100d", min_freq=2, unk_init= torch.Tensor.normal_)
        self.TRG.build_vocab(dataset, vectors = "glove.6B.100d", min_freq=2, unk_init= torch.Tensor.normal_)

    # if do Cross_Valid, we get cv dataset
    def get_cvdatasets(self, n_folds, SEED):
        
        train_data, test_data = self.load_data()

        train_exs, test_exs = train_data.examples, test_data.examples

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

        fields = [('Src', self.SRC), ('Trg', self.TRG)]

        def iter_folds():
            train_exs_arr = np.array(train_exs)
            for train_idx, val_idx in kf.split(train_exs_arr):
                yield(
                    Dataset(train_exs_arr[train_idx], fields),
                    Dataset(train_exs_arr[val_idx], fields),
                )
        
        test_d = Dataset(test_exs, fields)
        return iter_folds(), test_d

def main():
    lang = Lang(os.path.join(ROOT_DIR, 'data/geoqueries'), 'train.json', 'test.json')

    train_data, test_data = lang.load_data()
    print(vars(train_data[0]))
    lang.fields_vocab(train_data)
    print(lang.SRC.vocab.itos[2])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print('[STOP]', e)