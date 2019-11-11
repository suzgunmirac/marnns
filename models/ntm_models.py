## Import relevant libraries and dependencies
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random

# GPU/CPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

## Baby Neural Turing Machine
class BNTM_Softmax (nn.Module): 
    def __init__(self, hidden_dim, output_size, vocab_size, n_layers=1, memory_size=104, memory_dim=5): 
        super(BNTM_Softmax, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.opnum = 5

        self.rnn = nn.RNN(self.vocab_size, self.hidden_dim, self.n_layers)

        self.W_m = nn.Linear (self.memory_dim, self.hidden_dim)
        self.W_y = nn.Linear (hidden_dim, output_size)
        self.W_n = nn.Linear (hidden_dim, self.memory_dim)
        self.W_a = nn.Linear (hidden_dim, self.opnum)
        
        self.softmax = nn.Softmax (dim=0)
        self.sigmoid = nn.Sigmoid ()
        
        self.simple_addition = torch.zeros (memory_size, 1)
        self.simple_addition [0][0] = 1.0
        
        # Monoid operations
        self.nomove = torch.eye (memory_size, memory_size).to(device)
        self.leftmove = torch.zeros (memory_size, memory_size).to(device)
        self.rightmove = torch.zeros (memory_size, memory_size).to(device)
        self.rightinvmove = torch.zeros (memory_size, memory_size).to(device)
        self.leftinvmove = torch.zeros (memory_size, memory_size).to(device)
        for i in range (memory_size):
            self.leftmove [i][(i+1)%memory_size] = 1.
            self.rightinvmove [i][(i+1)%memory_size] = 1.
            self.rightmove [(i+1)%memory_size][i] = 1.
            self.leftinvmove [(i+1)%memory_size][i] = 1.
        self.leftinvmove [0][memory_size-1] = 0.
        self.rightinvmove [memory_size-1][0] = 0.


    def init_hidden (self):
        return torch.zeros (self.n_layers, 1, self.hidden_dim).to(device)
    
    def forward(self, input, hidden0, memory, temperature=1.):
        hidden_bar = self.W_m (memory[0]).view (1, 1, -1) + hidden0
        ht, hidden = self.rnn(input, hidden_bar)
        output = self.sigmoid(self.W_y (ht))
        self.action_weights = self.softmax (self.W_a (ht).view(-1))
        self.new_elt  = self.sigmoid(self.W_n (ht)).view(1, self.memory_dim)
        memory = torch.matmul (self.leftmove, memory) * self.action_weights [0] + torch.matmul (self.rightmove, memory) * self.action_weights [1] + torch.matmul (self.nomove, memory) * self.action_weights[2] + torch.matmul (self.leftinvmove, memory) * self.action_weights[3] + torch.matmul (self.rightinvmove, memory) * self.action_weights[4]
        memory = memory + torch.matmul(self.simple_addition, self.new_elt)
        return output, hidden, memory.view(-1, self.memory_dim)



class BNTM_SoftmaxTemperature (nn.Module): 
    def __init__(self, hidden_dim, output_size, vocab_size, n_layers=1, memory_size=104, memory_dim=5):
        super(BNTM_SoftmaxTemperature, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.opnum = 5

        self.rnn = nn.RNN(self.vocab_size, self.hidden_dim, self.n_layers)

        self.W_m = nn.Linear (self.memory_dim, self.hidden_dim)
        self.W_y = nn.Linear (hidden_dim, output_size)
        self.W_n = nn.Linear (hidden_dim, self.memory_dim)
        self.W_a = nn.Linear (hidden_dim, self.opnum)

        self.sigmoid = nn.Sigmoid ()
        
        self.simple_addition = torch.zeros (memory_size, 1).to(device)
        self.simple_addition [0][0] = 1.0
        
        # Monoid operations
        self.nomove = torch.eye (memory_size, memory_size).to(device)
        self.leftmove = torch.zeros (memory_size, memory_size).to(device)
        self.rightmove = torch.zeros (memory_size, memory_size).to(device)
        self.rightinvmove = torch.zeros (memory_size, memory_size).to(device)
        self.leftinvmove = torch.zeros (memory_size, memory_size).to(device)
        for i in range (memory_size):
            self.leftmove [i][(i+1)%memory_size] = 1.
            self.rightinvmove [i][(i+1)%memory_size] = 1.
            self.rightmove [(i+1)%memory_size][i] = 1.
            self.leftinvmove [(i+1)%memory_size][i] = 1.
        self.leftinvmove [0][memory_size-1] = 0.
        self.rightinvmove [memory_size-1][0] = 0.


    def init_hidden (self):
        return torch.zeros (self.n_layers, 1, self.hidden_dim).to(device)

    def softmax_temp (self, arr, temp):
        probs = torch.zeros (arr.shape).to(device)
        for i in range (self.opnum):
            probs [i] = torch.exp(arr[i]/temp)
        probs = probs / probs.sum(dim=0)
        return probs
    
    def forward(self, input, hidden0, memory, temperature):
        hidden_bar = self.W_m (memory[0]).view (1, 1, -1) + hidden0
        hidden, ht = self.rnn(input, hidden_bar)
        output = self.sigmoid(self.W_y (ht))
        self.action_weights = self.softmax_temp (self.W_a (ht).view(-1), temperature)
        self.new_elt  = self.sigmoid(self.W_n (ht)).view(1, self.memory_dim)
        memory = torch.matmul (self.leftmove, memory) * self.action_weights [0] + torch.matmul (self.rightmove, memory) * self.action_weights [1] + torch.matmul (self.nomove, memory) * self.action_weights[2] + torch.matmul (self.leftinvmove, memory) * self.action_weights[3] + torch.matmul (self.rightinvmove, memory) * self.action_weights[4]
        memory = memory + torch.matmul(self.simple_addition, self.new_elt)
        return output, hidden, memory.view(-1, self.memory_dim)


class BNTM_GumbelSoftmax (nn.Module): 
    def __init__(self, hidden_dim, output_size, vocab_size, n_layers=1, memory_size=104, memory_dim=5): 
        super(BNTM_GumbelSoftmax, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.opnum = 5

        self.rnn = nn.RNN(self.vocab_size, self.hidden_dim, self.n_layers)

        self.W_m = nn.Linear (self.memory_dim, self.hidden_dim)
        self.W_y = nn.Linear (hidden_dim, output_size)
        self.W_n = nn.Linear (hidden_dim, self.memory_dim)
        self.W_a = nn.Linear (hidden_dim, self.opnum)

        self.sigmoid = nn.Sigmoid ()
        
        self.simple_addition = torch.zeros (memory_size, 1).to(device)
        self.simple_addition [0][0] = 1.0
        
        # Monoid operations
        self.nomove = torch.eye (memory_size, memory_size).to(device)
        self.leftmove = torch.zeros (memory_size, memory_size).to(device)
        self.rightmove = torch.zeros (memory_size, memory_size).to(device)
        self.rightinvmove = torch.zeros (memory_size, memory_size).to(device)
        self.leftinvmove = torch.zeros (memory_size, memory_size).to(device)
        for i in range (memory_size):
            self.leftmove [i][(i+1)%memory_size] = 1.
            self.rightinvmove [i][(i+1)%memory_size] = 1.
            self.rightmove [(i+1)%memory_size][i] = 1.
            self.leftinvmove [(i+1)%memory_size][i] = 1.
        self.leftinvmove [0][memory_size-1] = 0.
        self.rightinvmove [memory_size-1][0] = 0.

    def init_hidden (self):
        return torch.zeros (self.n_layers, 1, self.hidden_dim).to(device)

    
    def forward(self, input, hidden0, memory, temperature):
        hidden_bar = self.W_m (memory[0]).view (1, 1, -1) + hidden0
        hidden, ht = self.rnn(input, hidden_bar)
        output = self.sigmoid(self.W_y (ht))
        self.action_weights = torch.nn.functional.gumbel_softmax (self.W_a (ht).view(1, -1), temperature)[0]
        self.new_elt  = self.sigmoid(self.W_n (ht)).view(1, self.memory_dim)
        memory = torch.matmul (self.leftmove, memory) * self.action_weights [0] + torch.matmul (self.rightmove, memory) * self.action_weights [1] + torch.matmul (self.nomove, memory) * self.action_weights[2] + torch.matmul (self.leftinvmove, memory) * self.action_weights[3] + torch.matmul (self.rightinvmove, memory) * self.action_weights[4]
        memory = memory + torch.matmul(self.simple_addition, self.new_elt)
        return output, hidden, memory.view(-1, self.memory_dim)