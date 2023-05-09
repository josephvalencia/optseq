from torch.distributions import Categorical, Exponential, Independent
import torch
import torch.nn.functional as F
import torch.nn as nn

''' Wraps torch.distributions.Categorical to make it differentiable.
It will accomodate sampling from the discrete distribution and computing derivatives with respect to the logits.
This will be used for activation maximization to optimize the sequence for a particular oracle function.'''

class DifferentiableCategorical(nn.Module):

    def __init__(self,seed_sequence,class_dim,onehot_fn,grad_method='straight_through',normalize_method=None):

        super().__init__()
        self.seed_sequence = seed_sequence
        self.class_dim = class_dim
        self.to_onehot = onehot_fn
        self.logits = nn.Parameter(self.init_logits(self.to_onehot(seed_sequence)))
        self.grad_method = grad_method

        if normalize_method == 'instance':
            self.normalize = nn.InstanceNorm1d(num_features=self.logits.shape[self.class_dim],affine=True)
        elif normalize_method == 'layer':
            self.normalize = nn.LayerNorm(normalized_shape=self.logits.size())
        else:
            self.normalize = nn.Identity()

    def init_logits(self,onehot,true_val=1.0,other_val=0.01):
        ''' Initialize logits based on a relaxation of the seed sequence one-hot encoding'''
        logits = torch.where(onehot == 1,true_val,other_val) 
        return logits
    
    def onehot_sample(self,n=1):
        sample = self.sample_n(128)
        #sample = self.__dist__().sample()
        #print(f'sample shape: {sample.shape}') 
        onehot_sample = self.to_onehot(sample.squeeze(1))
        #onehot_sample = self.to_onehot(sample)
        return onehot_sample

    def sample(self):
        return self.__call__()

    def forward(self):
        # normalize logits as in Linder and Seelig 2021 https://doi.org/10.1186/s12859-021-04437-5
        sampled = self.onehot_sample() 
        logits = self.normalize(self.logits) 

        if self.grad_method == 'normal':
            surrogate = straight_through_surrogate(logits,sampled)
            return ForwardBackwardWrapper.apply(sampled,surrogate)
        elif self.grad_method == 'softmax':
            surrogate = softmax_straight_through_surrogate(logits)
            return ForwardBackwardWrapper.apply(sampled,surrogate)
        elif self.grad_method == 'gumbel_softmax':
            surrogate = gumbel_softmax_straight_through_surrogate(logits)
            return ForwardBackwardWrapper.apply(sampled,surrogate)
        elif self.grad_method == 'gumbel_rao_softmax':
            surrogate = gumbel_rao_softmax_straight_through_surrogate(logits,sampled)
            return ForwardBackwardWrapper.apply(sampled,surrogate)
        elif self.grad_method == 'reinforce':
            return sampled
        
    def __dist__(self):
        logits = self.normalize(self.logits) 
        dim_order = [x for x in range(logits.dim()) if x != self.class_dim] + [self.class_dim]
        return Independent(Categorical(logits=logits.permute(*dim_order)),1)
    
    def sample_n(self,n):
        return self.__dist__().sample((n,))

    def log_prob(self,onehot_sample):
        dense_sample = torch.argmax(onehot_sample,dim=1)
        return self.__dist__().log_prob(dense_sample)

    def probs(self):
        logits = self.normalize(self.logits) 
        return F.softmax(logits,dim=1)
     
class ForwardBackwardWrapper(torch.autograd.Function):
    ''' Trick from Thomas Viehmann https://discuss.pytorch.org/t/relu-with-leaky-derivative/32818/10.
    Behaves like x_forward on the forward pass, and like x_backward on the backward pass.'''
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

def straight_through_surrogate(logits,onehot_sample):
    '''Original STE from Bengio et. al 2013 http://arxiv.org/abs/1308.3432'''
    return logits * onehot_sample

def softmax_straight_through_surrogate(logits,temperature=1.0):
    '''Softmax STE from Chung et. al 2017 http://arxiv.org/abs/1609.01704'''
    return  F.softmax(logits / temperature,dim=2)

def gumbel_softmax_straight_through_surrogate(logits,temperature=10,hard=False):
    '''Gumbel softmax STE from Jang et. al 2017 http://arxiv.org/abs/1611.01144'''
    return F.gumbel_softmax(logits,tau=temperature,hard=False,dim=2)

def gumbel_rao_softmax_straight_through_surrogate(logits,onehot_sample,temperature=0.1,n_samples=64):
    '''Rao-Blackwellized Gumbel softmax STE from Paulus et. al 2020 http://arxiv.org/abs/2010.04838
    with code taken from https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py'''
    adjusted_logits = logits + conditional_gumbel(logits, onehot_sample, k=n_samples)
    return F.softmax(adjusted_logits / temperature,dim=2).mean(dim=0)

@torch.no_grad()
def conditional_gumbel(logits, D, k=1):
    """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
    + Q) is given by D (one hot vector)."""
    # iid. exponential
    E = Exponential(rate=torch.ones_like(logits)).sample([k])
    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True)
    # partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    # Sampled gumbel-adjusted logits
    adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
                (1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
    return adjusted - logits













