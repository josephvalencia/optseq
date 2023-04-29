import torch
from typing import Union, Callable, Any
import torch.nn.functional as F
import random
import numpy as np

class DiscreteMCMCSampler():
    ''' Parent class for MCMC sampler from a discrete distribution using a gradient-informed proposal'''

    def __init__(self, start_seq : torch.tensor ,
                    num_classes : int,
                    output_fn : Union[Callable,torch.nn.Module],
                    metropolis_adjust=True,
                    constraint_functions=[]):
        
        self.output_fn = output_fn
        self.curr_seq = start_seq
        self.num_classes = num_classes
        self.metropolis = metropolis_adjust
        self.constraint_functions = constraint_functions
        self.step = 0
    
    def to_onehot(self,src):
        onehot = F.one_hot(src,self.num_classes).permute(0,2,1).float().requires_grad_(True)
        return onehot
    
    def input_grads(self,inputs):
        ''' per-sample gradient of outputs wrt. inputs '''
        outputs = self.output_fn(inputs)
        total_pred = outputs.sum()
        total_pred.backward(torch.ones_like(total_pred),inputs=inputs)
        grads = inputs.grad
        return grads

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.sample()

    def sample(self):
        raise NotImplementedError

class LangevinSampler(DiscreteMCMCSampler):
    '''Implements Zhang et al. ICML 2022 https://proceedings.mlr.press/v162/zhang22t.html'''

    def __init__(self,start_state,num_classes,output_fn,stepsize=0.01,beta=0.999,epsilon=1e-8,metropolis_adjust=True):
        super().__init__(start_state,num_classes,output_fn,metropolis_adjust)
        self.stepsize = stepsize
        self.beta = beta
        self.epsilon = epsilon
        self.curr_seq = start_state 
        self.preconditioner = self.curr_seq.new_zeros(self.to_onehot(self.curr_seq).shape,dtype=torch.float32)
    
    def sample(self):
        
        # calc q(x'|x)
        q_fwd = self.forward_proposal_dist(src=self.to_onehot(self.curr_seq))
        next_state = q_fwd.sample() 
        # metropolis adjustment
        if self.metropolis:
            curr_seq = self.metropolis_adjustment(q_fwd,next_state)
        else:
            curr_seq = next_state

        # apply constraints
        if len(self.constraint_functions):
            for constraint_fn in self.constraint_functions:
                curr_seq = constraint_fn(curr_seq)

        self.curr_seq = curr_seq
        self.step += 1
        return self.curr_seq
    
    def metropolis_adjustment(self,q_fwd,next_state):
        # calc q(x|x')
        next_state_onehot = self.to_onehot(next_state)
        q_rev = self.reverse_proposal_dist(src=next_state_onehot)
        # compute the acceptance probabilities and accept/reject 
        fwd_log_prob = q_fwd.log_prob(next_state)
        rev_log_prob = q_rev.log_prob(self.curr_seq)
        score_diff = self.output_fn(next_state_onehot) - self.output_fn(self.to_onehot(self.curr_seq))
        accept_probs = self.acceptance_probs(score_diff,fwd_log_prob,rev_log_prob) 
        return self.accept_reject(accept_probs,next_state,self.curr_seq)

    def acceptance_probs(self,score_diff,forward_log_prob,reverse_log_prob):
        acceptance_log_prob = score_diff + reverse_log_prob - forward_log_prob
        return torch.minimum(acceptance_log_prob,torch.tensor([0.0],device=score_diff.device))
    
    def accept_reject(self,accept_probs,next_state,curr_seq):
        ''' All positions are potentially resampled, so we need to accept/reject each position independently'''
        random_draw = torch.log(torch.rand(accept_probs.shape)) 
        acceptances = accept_probs > random_draw
        return torch.where(acceptances,next_state,curr_seq)
    
    def forward_proposal_dist(self,src):
        grads = self.input_grads(src)
        self.update_preconditioner(grads)
        return self.langevin_proposal_dist(grads,src)
         
    def reverse_proposal_dist(self,src):
        grads = self.input_grads(src)
        self.update_preconditioner(grads) 
        return self.langevin_proposal_dist(grads,src)
    
    def langevin_proposal_dist(self,grads,onehot):
        # Taylor approx 
        grad_current_char = (grads * onehot).sum(dim=1)
        mutation_scores = grads - grad_current_char
        # stepsize term
        #temperature_term = (1.0 - onehot)**2 / (2 * self.stepsize * self.scaled_preconditioner()**2) 
        temperature_term = (1.0 - onehot)**2 / (2 * self.stepsize)
        logits = 0.5 * mutation_scores - temperature_term 
        proposal_dist = torch.distributions.Categorical(logits=logits.permute(0,2,1))
        return torch.distributions.Independent(proposal_dist,0)

    def mask_rare_tokens(self,logits):
        '''Non-nucleotide chars may receive gradient scores, set them to a small number'''
        mask1 = torch.tensor([0,0,1,1,1,1,0,0],device=logits.device)
        mask2 = torch.tensor([-30,-30,0,0,0,0,-30,-30],device=logits.device)
        return logits*mask1 + mask2
    
    def update_preconditioner(self,grads):
        ''' Inspired by Adam diagonal preconditioner '''
        diagonal = grads.pow(2)
        self.preconditioner = self.beta*self.preconditioner + (1.0-self.beta)*diagonal 

    def scaled_preconditioner(self):
        ''' Bias correction for preconditioner ''' 
        bias_corrected = self.preconditioner / (1.0 - self.beta**(self.step+1)) 
        return bias_corrected.sqrt() + self.epsilon

class PathAuxiliarySampler(DiscreteMCMCSampler):
    '''Implements Sun et al. ICLR 2022 https://openreview.net/pdf?id=JSR-YDImK95 '''

    def __init__(self,start_state,output_fn,grad_fn,metropolis_adjust=True,constraint_functions=[],max_path_length=5):
        super().__init__(start_state,output_fn,grad_fn,metropolis_adjust,constraint_functions)
        self.max_path_length = max_path_length
        self.current_length = 1
        self.src_cache = []

    def sample(self):
        # calc q(x'|x)
        curr_seq_onehot = self.to_onehot(self.curr_seq)
        final_src_onehot,fwd_log_prob = self.forward_proposal_dist(curr_seq_onehot)
        # calc q(x'|x) 
        rev_log_prob = self.reverse_proposal_dist(final_src_onehot)
        # compute the acceptance probabilities and accept/reject
        score_diff = self.output_fn(final_src_onehot) - self.output_fn(curr_seq_onehot)
        accept_probs = self.acceptance_probs(score_diff,fwd_log_prob,rev_log_prob) 
        final_src_dense = final_src_onehot.argmax(dim=1,keepdims=False) 
        self.curr_seq =  self.accept_reject(accept_probs,final_src_dense,self.curr_seq)
        self.step += 1
        return self.curr_seq
    
    def update_seq(self,next_state,curr_src):
        char_idx = torch.floor_divide(next_state,curr_src.shape[2])
        pos_idx = next_state % curr_src.shape[2]
        # update character at position pos_idx
        next_seq = curr_src.clone() 
        pos_idx = pos_idx.long()
        char_idx = char_idx.long()
        next_seq[0,:, pos_idx] = F.one_hot(char_idx,num_classes=curr_src.shape[1])
        return next_seq
    
    def forward_proposal_dist(self,src):

        self.current_length = random.randint(1,1)
        # gradients at initial state 
        grads = self.input_grads(src)
        log_prob = 0.0
        for i in range(self.current_length): 
            q_i = self.path_auxiliary_proposal_dist(src,grads)
            next_state = q_i.sample() 
            log_prob += q_i.log_prob(next_state)
            # store the endogenous character in the position of the proposed next state  
            next_state_pos = torch.floor_divide(next_state,self.curr_seq.shape[1]).long()
            curr_state_char = self.curr_seq[0,next_state_pos] 
            curr_state = next_state_pos * self.curr_seq.shape[1] + curr_state_char
            self.src_cache.append((src,curr_state)) 
            src = self.update_seq(next_state,src)
        return src,log_prob

    def reverse_proposal_dist(self,src):

        # gradients at final state 
        src,state = self.src_cache[-1] 
        grads = self.input_grads(src)
        log_prob = 0.0

        for i in range(self.current_length-2,0,-1): 
            q_i = self.path_auxiliary_proposal_dist(src,grads)
            # log prob of cached previous state rather than sampled next state 
            src,state = self.src_cache[i]
            log_prob += q_i.log_prob(state) 

        return log_prob

    def path_auxiliary_proposal_dist(self,src,grads):
        grad_current_char = (grads * src).sum(dim=1)
        mutation_scores = grads - grad_current_char
        # V*(L-1) -way softmax 
        return torch.distributions.Categorical(logits=mutation_scores.reshape(-1) / 2)

    def acceptance_probs(self,score_diff,forward_log_prob,reverse_log_prob):
        print(f'reverse_prob: {reverse_log_prob}, forward_prob: {forward_log_prob}, score_diff: {score_diff}')
        acceptance_log_prob = score_diff + reverse_log_prob - forward_log_prob
        return torch.minimum(acceptance_log_prob,torch.tensor([0.0],device=score_diff.device))
    
    def accept_reject(self,accept_probs,next_seq,curr_seq):
        
        random_draw = torch.log(torch.rand(accept_probs.shape)) 
        acceptances = accept_probs > random_draw
        print(f'P(accept): {accept_probs.exp()}') 
        if acceptances:
            return next_seq
        else:
            return curr_seq 

class GibbsWithGradientsSampler(DiscreteMCMCSampler):
    '''Implements Grathwohl et al. ICML 2021 '''
    
    def sample(self):
        # calc q(x'|x)
        q_fwd = self.forward_proposal_dist(src=self.to_onehot(self.curr_seq))
        next_state = q_fwd.sample() 
        # metropolis adjustment
        self.curr_seq = self.metropolis_adjustment(q_fwd,next_state)
        self.step += 1
        return self.curr_seq
    
    def forward_proposal_dist(self,src):
        return self.gwg_proposal_dist(src)        

    def reverse_proposal_dist(self,src):    
        return self.gwg_proposal_dist(src)

    def gwg_proposal_dist(self,src):
        # Taylor approx 
        grads = self.input_grads(src)
        grad_current_char = (grads * src).sum(dim=1)
        mutation_scores = grads - grad_current_char
        # V*(L-1) way softmax 
        entropy = lambda x : torch.sum(-x * torch.log2(x))
        cat_params = F.softmax(mutation_scores.reshape(-1),dim=0)
        return torch.distributions.Categorical(logits=mutation_scores.reshape(-1) / 2)

    def update_seq(self,next_state):
        char_idx = torch.floor_divide(next_state,self.curr_seq.shape[1])
        pos_idx = next_state % self.curr_seq.shape[1]
        # update character at position pos_idx
        next_seq = self.curr_seq.clone() 
        pos_idx = pos_idx.long()
        char_idx = char_idx.long()
        next_seq[0,pos_idx] = char_idx
        return next_seq
    
    def metropolis_adjustment(self,q_fwd,next_state):
        # calc q(x|x')
        next_seq = self.update_seq(next_state) 
        next_seq_onehot = self.to_onehot(next_seq)
        q_rev = self.reverse_proposal_dist(src=next_seq_onehot)
        # store the endogenous character in the position of the proposed next state  
        next_state_pos = torch.floor_divide(next_state,self.curr_seq.shape[1]).long()
        curr_state_char = self.curr_seq[0,next_state_pos] 
        curr_state = next_state_pos * self.curr_seq.shape[1] + curr_state_char
        # compute acceptance probabilities
        fwd_log_prob = q_fwd.log_prob(next_state)
        rev_log_prob = q_rev.log_prob(curr_state.long())
        score_diff = self.output_fn(next_seq_onehot) - self.output_fn(self.to_onehot(self.curr_seq))
        accept_probs = self.acceptance_probs(score_diff,fwd_log_prob,rev_log_prob) 
        return self.accept_reject(accept_probs,next_seq,self.curr_seq)

    def acceptance_probs(self,score_diff,forward_log_prob,reverse_log_prob):
        acceptance_log_prob = score_diff + reverse_log_prob - forward_log_prob
        return torch.minimum(acceptance_log_prob,torch.tensor([0.0],device=score_diff.device))
    
    def accept_reject(self,accept_probs,next_seq,curr_seq):
        
        random_draw = torch.log(torch.rand(accept_probs.shape)) 
        acceptances = accept_probs > random_draw
        
        if acceptances:
            return next_seq
        else:
            return curr_seq 