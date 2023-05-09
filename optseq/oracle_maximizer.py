import torch
from typing import Callable, Optional,Union, Tuple, List
import torch.nn as nn
from optseq.categorical import DifferentiableCategorical
from optseq.mcmc import LangevinSampler, PathAuxiliarySampler, GibbsWithGradientsSampler, RandomSampler, SimulatedAnnealingSampler
from optseq.ngd import NaturalGradientDescent
from tqdm import tqdm

class OracleMaximizer:

    def __init__(self,
                seed_sequence : torch.tensor, 
                num_classes : int, 
                class_dim : int,
                oracles : List[Union[Callable,nn.Module]],
                loss_items : List[Tuple[Union[Callable,None],Union[torch.tensor,None],torch.tensor]],
                onehot_fn : Callable,
                readable_fn : Callable,
                device : str = "cpu", 
                mode : str = "optimize",
                mcmc : str = "langevin",
                grad : str = "normal",
                optimizer : Union[str,None] = None,
                max_iter : Optional[int] = None):

        self.oracles = oracles
        self.loss_items = loss_items
        self.onehot_fn = onehot_fn
        self.to_nucleotide = readable_fn
        self.device = device
        self.seed_sequence = seed_sequence.to(self.device)
        self.mode = mode
        self.mcmc = mcmc
        self.grad = grad
        self.class_dim = class_dim

        if self.mode == 'optimize': 
            self.driver = DifferentiableCategorical(
                                    self.seed_sequence,
                                    self.class_dim, 
                                    self.onehot_fn, 
                                    grad_method=grad,
                                    normalize_method=None)
            self.driver.to(self.device)
        elif self.mode == 'sample':
            if mcmc == 'langevin':
                self.driver =  LangevinSampler(
                                    self.seed_sequence,
                                    num_classes=num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn)
            elif mcmc == 'path_auxiliary': 
                self.driver = PathAuxiliarySampler(
                                    self.seed_sequence,
                                    num_classes=num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn)
            elif mcmc == 'gibbs_with_gradients':
                self.driver = GibbsWithGradientsSampler(
                                    self.seed_sequence,
                                    num_classes=num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn) 
            elif mcmc == 'random':
                self.driver = RandomSampler(
                                    self.seed_sequence,
                                    num_classes=num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn)
            elif mcmc == 'annealing':
                self.driver = SimulatedAnnealingSampler(
                                    self.seed_sequence,
                                    num_classes=num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn,
                                    num_steps=max_iter)

        else:
            raise ValueError("mode must be one of optimize|sample")
        
        self.optimizer = None
        if mode == 'optimize': 
            if optimizer == 'adam': 
                self.optimizer = torch.optim.Adam(self.driver.parameters(),lr=1.0) 
            elif optimizer == 'lbfgs':  
                self.optimizer = torch.optim.LBFGS(self.driver.parameters(),lr=1e-2) 
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(self.driver.parameters(),lr=1e-3)
            elif optimizer == 'NGD':
                self.optimizer = NaturalGradientDescent(self.driver.parameters(),lr=1.0)
            else:
                raise ValueError("When `mode`==`optimize`,`optimizer` must be one of adam|lbfgs|SGD")

    def preds_and_composite_loss(self,sequence,):
        '''Evaluate all oracles on a sequence and return a weighted loss'''
        total_loss = 0.0
        for oracle,(loss,target,weight) in zip(self.oracles,self.loss_items):
            pred = oracle(sequence)
            if loss is not None and target is not None: 
                # a loss is something to minimize 
                total_loss += weight*loss(pred,target)
            else:
                # a scalar output is something to maximize, so we negate it
                total_loss += -weight*pred

        # Pytorch optimizers assume minimization, MCMC assumes maximization
        # so a final negation for MCMC
        if self.mode == 'sample':
            total_loss = -total_loss 
        return total_loss.mean()
    
    def optimize_step(self,sequence):
            
        def closure():
            self.optimizer.zero_grad()
            next_loss = self.preds_and_composite_loss(sequence)
            if self.grad == 'reinforce':
                next_loss *= self.driver.log_prob(sequence)
            next_loss.backward()
            return next_loss
        
        if isinstance(self.optimizer,torch.optim.LBFGS):
            self.optimizer.step(closure)
            loss = closure()
        elif isinstance(self.optimizer,NaturalGradientDescent):
            self.optimizer.zero_grad()
            loss = self.preds_and_composite_loss(sequence)
            # additionally accumulate gradients for the sample sequences 
            loss.backward(torch.ones_like(loss),inputs=(sequence,*self.driver.parameters()))
            per_sample_grads = sequence.grad 
            self.optimizer.step(per_sample_grads) 
        else: 
            loss = closure() 
            self.optimizer.step()
        return loss

    def sample_step(self,sequence):
        pass

    def fit(self,
            max_iter : int = 20000,
            stalled_tol : int = 10000,
            report_interval : int = 100):

        # setup the seed sequence
        onehot_seed = self.onehot_fn(self.seed_sequence).to(self.device)
        readable = self.to_nucleotide(self.seed_sequence)

        # get initial predictions and loss
        initial_loss = self.preds_and_composite_loss(onehot_seed)
        best_loss = initial_loss
        best_seq = self.onehot_fn(self.seed_sequence) 
        results = [initial_loss.detach().item()]

        print(f'Fitting OracleMaximizer using {self.driver}')
        print(f'Seed sequence {readable} loss = {initial_loss.item():.3f}')
        stalled_counter = 0
        pbar = tqdm(range(max_iter))
        for i in pbar:

            # sample and perform gradient-based updates
            next_sequence = self.driver.sample()
            if self.mode == 'optimize': 
                next_loss = self.optimize_step(next_sequence)
            else:
                next_loss = -self.preds_and_composite_loss(next_sequence)

            # report progress
            results.append(next_loss.detach().item())

            # monitor convergence
            if next_loss < best_loss:
                best_loss = next_loss
                best_seq = next_sequence
                stalled_counter = 0
            else:
                stalled_counter += 1
            if i % report_interval == 0: 
                dense = best_seq.argmax(dim=self.class_dim).squeeze()
                diff = torch.count_nonzero(dense.squeeze() != self.seed_sequence.squeeze())
                pbar.set_postfix({'best_loss': best_loss.item(),'diff': diff.item()})
            if stalled_counter > stalled_tol:
                print(f'Stopping early at iteration {i}')
                break
        
        best_seq_dense = best_seq.argmax(dim=1).squeeze()
        return best_seq_dense.detach(), best_loss.detach(), results
        
