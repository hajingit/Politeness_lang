import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html



class RNN(nn.Module):
  # Hint: Think about what the various numbers mean. Which ones make sense and which ones don't?
  def __init__(self, in_dim=174, hidden_dim=128, out_dim=3):
    super(RNN, self).__init__()
    self.hidden_dim = hidden_dim
    #self.embeds = nn.Embedding(vocab_size, hidden_dim)
    self.encoder = nn.LSTM(in_dim, hidden_dim, bidirectional=False)
    self.loss = nn.CrossEntropyLoss()
    self.out = nn.Linear(hidden_dim, out_dim)

  def compute_Loss(self, pred_vec, gold_output):
    return self.loss(pred_vec, gold_output)
      
  def forward(self, input_vectors):
    #input_vectors = self.embeds(torch.tensor(input_seq)) # create embeddings,
    input_vectors = input_vectors.unsqueeze(1)
    _, hidden = self.encoder(input_vectors)
    forward, backward = hidden[0], hidden[1]
    concat = torch.cat([forward, backward], 1)
    output = torch.nn.functional.relu(concat)
    prediction = self.out(output)
    prediction = prediction.squeeze()
    val, idx = torch.max(prediction, 0)
    return prediction, idx.item()

class BidirRNN(RNN):
  def __init__(self, in_dim=174, hidden_dim=128, out_dim=3):
    super(BidirRNN, self).__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    self.encoder = nn.LSTM(hidden_dim, 2*hidden_dim, bidirectional=True)
        
