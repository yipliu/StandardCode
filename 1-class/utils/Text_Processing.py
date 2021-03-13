import os
import sys

import numpy as np

# --------------------------------
# Vocabulary and Text Processing
# --------------------------------

class Vocabulary(object):
    """Simple vocabulary wrapper.
    """
    def __init__(self, 
                 start_word='<sos>',
                 end_word='<eos>',
                 unk_word=None):
        # Store word_index
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_counts = {}

        # Add special tokens
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        for special_token in [start_word, end_word, unk_word]:
            if special_token is not None:
                self.add_word(special_token, freq=0)

    def word_to_idx(self, 
                    word):
        """Return the integer word index of a word token.
        """
        if not word in self.word2idx:
            if self.unk_word is None:
                return None   # Return None if no unknown word's defined
            else:
                return self.word2idx[self.unk_word]
        else:
            return self.word2idx[word]

    def idx_to_word(self, 
                    idx):
        """Return the word string of an integer word index.
        """
        if idx >= len(self.idx2word):
            if self.unk_word is None:
                return None
            else:
                return self.unk_word
        else:
            return self.idx2word[idx]

    def __len__(self):
        """Return the length of the vocabulary.
        """
        return len(self.word2idx)
    
    def get_vocab(self):
        """Get all words in the vocabulary.
        """
        # Safely extract the word vocab in order using idx
        words = []
        for i in range(self.idx):
            words.append(self.idx2word[i])
        return words

    def add_word(self, 
                 word, 
                 freq=None):
        """Add individual word to the vocabulary.
        """
        if not word in self.word2idx and word is not None:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

        # Populate word frequency count as well
        if freq is not None:
            self.word_counts[word] = freq
        else:
            try:
                self.word_counts[word] += 1
            except:
                self.word_counts[word] = 1

    def get_bias_vec(self):
        """Calculate bias vector from word frequency distribution.
        NOTE: Frequency need to be properly stored.
        From NeuralTalk.
        """
        words = self.get_vocab()    # Make sure bias_vec is aligned with word indices
        bias_vec = np.array([1.0*self.word_counts[word] for word in words])
        bias_vec /= np.sum(bias_vec) # Normalize to frequencies
        bias_vec = np.log(bias_vec + 1e-16) # Add a small offset
        bias_vec -= np.max(bias_vec) # Shift to nice numeric range
        return bias_vec

def build_vocab(texts, 
                frequency=None, 
                filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                lower=True,
                split=" ", 
                start_word='<sos>',
                end_word='<eos>',
                unk_word=None):
    """Build vocabulary over a set of texts.
    """
    # Load annotations
    counter = Counter()
    for i, text in enumerate(texts):
        tokens = word_tokenize(text, filters, lower, split)
        #print(tokens)
        counter.update(tokens)
        if (i+1) % 5000 == 0:
            print('{} texts tokenized...'.format(i+1))

    # Filter out words lower than the defined frequency
    if frequency is not None:
        counter = {word: cnt for word, cnt in counter.items() if cnt >= frequency}
    else:
        counter = counter

    # Create a vocabulary warpper
    vocab = Vocabulary(start_word=start_word,
                       end_word=end_word,
                       unk_word=unk_word)

    words = sorted(counter.keys())
    for word in words:
        vocab.add_word(word, counter[word])
    print('Vocabulary ready.')
    return vocab

def get_maxlen(texts):
    """Calculate the maximum document length for a list of texts.
    """
    return max([len(x.split(" ")) for x in texts])

def word_tokenize(text,
                  filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                  lower=True, 
                  split=" "):
    """Convert a text to a sequence of words (or tokens).
    """
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [i for i in seq if i]

def text_to_sequence(text,
                     vocab,
                     filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                     lower=True, 
                     split=" "):
    """Convert a text to numerical sequence.
    """
    tokens = word_tokenize(text, filters, lower, split)
    seq = []
    for token in tokens:
        word_index = vocab.word_to_idx(token)
        if word_index is not None:  # Filter out unknown words
            seq.extend([word_index])
    return seq

def sequence_to_text(seq, 
                     vocab, 
                     filter_specials=True, 
                     specials=['<pad>', '<sos>', '<eos>']):
    """Restore sequence back to text.
    """
    tokens = []
    for idx in seq:
        token = vocab.idx_to_word(idx)
        tokens.append(token)
    if filter_specials:
        tokens =  filter_tokens(tokens, specials)
    return ' '.join(tokens)

def texts_to_sequences(texts,
                       vocab,
                       filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                       lower=True, 
                       split=" "):
    """Wrapper to convert batch of texts to sequences.
    """
    seqs = []
    for text in texts:
        seqs.append(text_to_sequence(text, vocab, filters, lower, split))
    return np.array(seqs, dtype='object')

def filter_tokens(tokens, 
                  specials=['<pad>', '<sos>', '<eos>']):
    """Filter specified words.
    """
    filtered = []
    for token in tokens:
        if token not in specials:
            filtered.append(token)
    return filtered

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """(Same from Tensorflow) Pads sequences to the same length.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
