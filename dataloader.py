
import os
import numpy as np
import pandas as pd
import torchtext
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from torchtext.data import get_tokenizer   # for tokenization
from collections import Counter, OrderedDict     # for tokenizer
from PIL import Image

class Vocab:
    def __init__(self, min_freq = 1):
        self.itos = {0:'<PAD>',1:'<START>',2:'<END>',3:'<UNK>'}
        self.stoi = {'<PAD>':0,'<START>':1,'<END>':2,'<UNK>':3}

        self.min_freq = min_freq

        self.tokenizer = get_tokenizer('basic_english')
        self.frequencies = Counter()

    def __len__(self):
        return len(self.itos)
    
    def build_vocab(self, sentence_list):
        idx = 4
        for sentence in sentence_list:
            sentence_tokens = self.tokenizer(str(sentence))
            self.frequencies.update(sentence_tokens)
            #TODO: handle null sentences/instructions
            for token in sentence_tokens:
                if token not in self.stoi.keys() and self.frequencies[token] >= self.min_freq:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx +=1

    def numericalize(self, sentence):
        sentence_tokens = self.tokenizer(str(sentence))
        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in sentence_tokens]



class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        recipes = [item[1] for item in batch]
        recipes = pad_sequence(recipes, batch_first=False, padding_value=self.pad_idx)

        return imgs, recipes

class CustomDataset(Dataset):
    def __init__(self, img_dir, recipe_csv, transform = None, min_freq = 3):
        self.img_dir = img_dir
        self.transform = transform
        init_df = pd.read_csv(recipe_csv)
        ###TODO cleaning function :
        # clean_df = init_df.dropna(subset=['Instructions'], inplace=True)
        clean_df = init_df[init_df["Image_Name"] != '#NAME?']
        
        self.recipe_df = clean_df
        self.tokenizer = get_tokenizer('basic_english')

        self.vocab = Vocab(min_freq)
        self.vocab.build_vocab(self.recipe_df["Instructions"].tolist())


    def __len__(self):
        return len(self.recipe_df)
    
    def __getitem__(self, index):
        img_file = os.path.join(self.img_dir, self.recipe_df.iloc[index,4] + '.jpg')
        img = Image.open(img_file)
        img = img.convert('RGB')
        img_name = self.recipe_df.iloc[index,1]
        recipe = self.recipe_df.iloc[index,3]

        recipe_tokens = []
        recipe_tokens += [self.vocab.stoi['<START>']]
        recipe_tokens += self.vocab.numericalize(str(recipe))
        recipe_tokens += [self.vocab.stoi['<END>']] 


        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(recipe_tokens)
