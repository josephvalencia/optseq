import torch
from Bio.Seq import Seq
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.SeqRecord import SeqRecord
from design.base import  OneHotGradientAttribution
from bioseq2seq.bin.transforms import CodonTable, getLongestORF
import torch.nn.functional as F
import time
import numpy as np

class TargetConditionalSampler(OneHotGradientAttribution):
 
    def mask_rare_tokens(self,logits):
        '''Non-nucleotide chars may receive gradient scores, set them to a small number'''
        
        mask1 = torch.tensor([0,0,1,1,1,1,0,0],device=logits.device)
        mask2 = torch.tensor([-30,-30,0,0,0,0,-30,-30],device=logits.device)
        return logits*mask1 + mask2

    def langevin_proposal_dist(self,grads,indexes,onehot):
       
        grad_current_char =  torch.gather(grads,dim=3,index=indexes.unsqueeze(2))
        mutation_scores = grads - grad_current_char
        temperature_term = (1.0 - onehot) / (self.alpha * self.scaled_preconditioner()) 
        logits = 0.5 * mutation_scores - temperature_term 
        #logits = self.mask_rare_tokens(logits)
        proposal_dist = torch.distributions.Categorical(logits=logits.squeeze(2))
        return torch.distributions.Independent(proposal_dist,1)
  
    def metropolis_hastings(self,score_diff,forward_prob,reverse_prob):
        acceptance_log_prob = score_diff + reverse_prob - forward_prob
        return torch.minimum(acceptance_log_prob,torch.tensor([0.0],device=score_diff.device))
   
    def set_stepsize(self,alpha):
        self.alpha = alpha
    
    def get_stepsize(self):
        return self.alpha

    def update_preconditioner(self,grads,step):
       
        grads = grads.pow(2)
        diagonal = grads
        biased = self.beta*self.preconditioner + (1.0-self.beta)*diagonal 
        self.preconditioner = biased / (1.0 - self.beta**(step+1))

    def scaled_preconditioner(self):
        return self.preconditioner.sqrt() + 1e-8
        #return self.preconditioner.sqrt() + 1.0

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        start = time.time()
        mutant_records = []
        stepwise=True 
        for i,batch in enumerate(val_iterator):
            
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            tscript = transcript_names[ids[0]]
            
            tgt_len = batch.tgt.shape[0]
            # prediction and input grads given tgt_prefix
            tgt_prefix = batch.tgt[:target_pos,:,:] 
            class_token = batch.tgt[target_pos,0,0].item() if self.class_token == 'GT' else self.class_token
            
            # score true
            original = src
            original_raw_src = self.get_raw_src(original)
            original_onehot = self.onehot_embed_layer(original)
            print(original.shape,original_onehot.shape,batch.tgt.shape,src_lens)
            original_loss = self.translation_loss(original_onehot,batch.tgt,src_lens,batch)
            print(original_loss)
            quit()

            MAX_STEPS = 20000
            self.set_stepsize(0.001)
            self.beta = 0.999
            self.preconditioner = torch.zeros(size=(*src.shape,8),device=src.device,dtype=torch.float)
            max_score = original_loss
            best_seq = original 
            best_step = 0
            batch_preds = []
            stalled_count = 0
            for s in range(MAX_STEPS): 
                onehot_src = self.onehot_embed_layer(src) 
                
                # calc q(x'|x)
                loss = self.translation_loss(onehot_src,batch.tgt,src_lens,batch)
                curr_grads = self.input_grads(loss,onehot_src)
                
                self.update_preconditioner(curr_grads,s)
                proposal_dist = self.langevin_proposal_dist(curr_grads,src,onehot_src)
                resampled = proposal_dist.sample().unsqueeze(2)
                forward_prob = proposal_dist.log_prob(resampled.squeeze(2)) 
                diff = torch.count_nonzero(src != resampled)
                    
                # correct with MH step
                # calc q(x|x')
                resampled_onehot = self.onehot_embed_layer(resampled)
                resampled_loss = self.translation_loss(resampled_onehot,batch.tgt,src_lens,batch)
                resampled_grads = self.input_grads(resampled_loss,resampled_onehot)
                resampled_proposal_dist = self.langevin_proposal_dist(resampled_grads,resampled,resampled_onehot)
                reverse_prob = resampled_proposal_dist.log_prob(src.squeeze(2))

                score_diff = resampled_loss - loss
                accept_log_probs = self.metropolis_hastings(score_diff,forward_prob,reverse_prob) 
                random_draw = torch.log(torch.rand(src.shape[0],device=src.device))
                acceptances = accept_log_probs > random_draw
                #acceptances = acceptances.new_ones(*acceptances.shape)

                src = torch.where(acceptances,resampled,src)
                 
                # cache best sequence
                if resampled_loss > max_score:
                    max_score = resampled_loss 
                    best_seq = resampled
                    raw_src = ''.join(self.get_raw_src(best_seq))
                    print(f'new best seq @ step {s}, loss {-max_score} {raw_src}')
                    best_step = s
                    stalled_count = 0
                else:
                    stalled_count +=1
                    if stalled_count >= 2000:
                        break
                # log progress 
                if s % 10 == 0:
                    best_onehot = self.onehot_embed_layer(src)
                    best_loss = self.translation_loss(best_onehot,batch.tgt,src_lens,batch)
                    diff_original = torch.count_nonzero(src != original)
                    score_improvement = best_loss - original_loss
                    end = time.time()
                    verbose = True
                    if verbose: 
                        print(f'step={s}, {tscript}') 
                        print(f'# diff from original = {diff_original}/{src.shape[0]*src.shape[1]}')
                        print(f'score improvement = {score_improvement.item():.3f}')
                        print(f'Time elapsed = {end-start} s')
                    start = time.time()
            sep = '________________________________________'
            print(sep)
            print(f"The sequence has converged, found at step {best_step}")
            diff_original = torch.count_nonzero(best_seq != original)
            differences = (best_seq != original).reshape(-1).detach().cpu().numpy().tolist()
            print(f'# diff from original = {diff_original}/{best_seq.shape[0]*best_seq.shape[1]}')
            raw_src = self.get_raw_src(best_seq)
            print(''.join(raw_src))
            print('ORIGINAL',''.join(original_raw_src))
            best_onehot = self.onehot_embed_layer(best_seq)
            best_loss = self.translation_loss(best_onehot,batch.tgt,src_lens,batch)
            
            print(sep)
            description = f'step_found : {best_step}, diff:{diff_original}/{best_seq.shape[0]*best_seq.shape[1]} best_loss : {-best_loss.item():.3f}, original_loss : {-original_loss.item():.3f}'
            print(description)
            rec = SeqRecord(Seq(''.join(raw_src)),id=f'{tscript}-MUTANT',description=description)
            mutant_records.append(rec)
        
        with open(f'{savefile}.fasta','w') as out_file:
            SeqIO.write(mutant_records,out_file,'fasta')
