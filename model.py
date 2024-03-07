import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from PIL import Image
import seaborn as sns
import splitfolders
from IPython.display import display, HTML



class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.arch = models.resnet101(pretrained = True)
        self.arch.fc = nn.Linear(self.arch.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.arch(images)
        return self.dropout(self.relu(features))
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, recipe_tokens):
        embeddings = self.dropout(self.embed(recipe_tokens))
        #Add features as a dimension for embeddings as the 'first' input in the sequence
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, recipe_tokens):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, recipe_tokens)
        return outputs

    def recipe_generate(self, image, vocabulary, max_length=50):
        result_recipe = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                
                output_dist = output.data.view(-1).div(0.8).exp()
                top_i = torch.multinomial(output_dist, 1)[0]
                
                result_recipe.append(top_i.item())

                x = self.decoderRNN.embed(torch.unsqueeze(top_i, 0)).unsqueeze(0)

                if vocabulary.itos[top_i.item()] == "<END>":
                    break

        return [vocabulary.itos[idx] for idx in result_recipe]