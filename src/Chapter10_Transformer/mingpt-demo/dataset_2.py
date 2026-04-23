#%%
import math
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long) # x = "hello" -> "hell"
        y = torch.tensor(dix[1:], dtype=torch.long) # y = "hello" -> "ello"
        return x, y

class RNATokenizer:
    def __init__(self, k=1):
        self.k = k
        self.vocab = self.build_vocab()
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.pad_token_id = self.vocab[self.pad_token]

    def build_vocab(self):
        import itertools
        bases = ['A', 'C', 'G', 'U']
        kmers = [''.join(p) for p in itertools.product(bases, repeat=self.k)]
        vocab = {kmer: idx for idx, kmer in enumerate(kmers)}
        vocab['[PAD]'] = len(vocab)
        vocab['[CLS]'] = len(vocab)
        vocab['[SEP]'] = len(vocab)
        return vocab

    def tokenize(self, seq):
        tokens = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]
        return [self.cls_token] + tokens + [self.sep_token]

    def encode(self, seq, max_length=None):
        tokens = self.tokenize(seq)
        token_ids = [self.vocab.get(t, self.vocab[self.pad_token]) for t in tokens]
        if max_length:
            token_ids = token_ids[:max_length]
            token_ids += [self.pad_token_id] * (max_length - len(token_ids))
        return token_ids

#%%
class RNADataset(Dataset):

    def __init__(self, data, block_size, k = 1):
        import itertools
        chars = sorted(list(set(data)))
        self.k = k
        kmers = [''.join(p) for p in itertools.product(chars, repeat=self.k)]
        data_size, vocab_size = len(data), len(kmers)
        print(f'data has {len(chars)} characters, {vocab_size} kmers.')
        self.PAD_token = '[PAD]'
        self.START_token = '[START]'
        self.END_token = '[END]'
        kmers.append(self.PAD_token)
        kmers.append(self.START_token)
        kmers.append(self.END_token)
        vocab_size = len(kmers)
        
        self.stoi = { kmer:i for i,kmer in enumerate(kmers) }
        self.itos = { i:kmer for i,kmer in enumerate(kmers) }
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def tokenize(self, seq):
        tokens = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]
        return tokens

    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        tokens = self.tokenize(chunk)
        # encode every character to an integer
        dix = [self.stoi[s] for s in tokens]
        x = torch.tensor(dix[:-1], dtype=torch.long) # x = "hello" -> "hell"
        y = torch.tensor(dix[1:], dtype=torch.long) # y = "hello" -> "ello"
        return x, y

#%%
data = 'ACGTACGTACGTACGTACGTACGTACGTACGTACGT'
block_size = 10
dataset = RNADataset(data, block_size, k=3)
x, y = dataset[0]
print(x)
print(y)
# %%
