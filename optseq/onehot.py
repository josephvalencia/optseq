import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHotToEmbedding(nn.Module):
    '''Converts one-hot encoding into dense embedding'''    
    
    def __init__(self,embedding : nn.Embedding):
        super(OneHotToEmbedding,self).__init__()
        self.embedding = embedding
    
    @property
    def vocab_size(self):
        return self.embedding.weight.shape[0]
    
    @property
    def out_dim(self):
        return self.embedding.weight.shape[1]
    
    def forward(self,one_hot_indexes):
        return F.linear(one_hot_indexes,self.embedding.weight.T) 
    
class TensorToOneHot(nn.Module):
    '''Converts Tensor index into intermediate one-hot encoding'''    
    def __init__(self,embedding : nn.Embedding):
        super(TensorToOneHot,self).__init__()
        self.embedding = embedding

    @property
    def vocab_size(self):
        return self.embedding.weight.shape[0]

    def forward(self,indexes):
        one_hot_indexes = F.one_hot(indexes,self.vocab_size).type(torch.float).requires_grad_(True)
        return one_hot_indexes
