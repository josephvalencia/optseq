import torch
from models.optimus_5prime.model import MeanRibosomeLoadModule
from optseq.oracle_maximizer import OracleMaximizer
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np

def encode_seq(seq):
    utr = torch.nn.functional.one_hot(seq,num_classes=4).permute(0,2,1).float().requires_grad_(True)
    return utr

def to_nucleotide(dense):                                                                                                                                        
    as_list = dense.cpu().numpy().ravel().tolist()
    mapping = 'ACGT'
    nucs = [mapping[x] for x in as_list] 
    return ''.join(nucs)

if __name__ == "__main__":

    device = "cuda:0"
    module = MeanRibosomeLoadModule.load_from_checkpoint(
                "models/optimus_5prime/checkpoints/epoch=2-val_loss=0.1396.ckpt",map_location=device)
    model = module.model
    model.eval()
    model.to(device) 
    print('MRL model loaded')
    seed_sequence = torch.randint(0,4,(1,50))
    
    maximizer = OracleMaximizer(seed_sequence,num_classes=4,
                                class_dim=1,
                                oracles = [model],
                                loss_items=[(None,None,1.0)],
                                onehot_fn=encode_seq,
                                readable_fn=to_nucleotide,
                                device=device, 
                                mode='optimize',optimizer='adam',grad='normal')
                                #mode='sample',mcmc='gibbs_with_gradients',
                                #max_iter=10000)
    
    best_seq,best_loss,results = maximizer.fit(max_iter=10000,stalled_tol=5000)
    readable = to_nucleotide(best_seq)
    score = model(encode_seq(best_seq.unsqueeze(0)).to(device)) 
    print(f'Best sequence {readable}, score ={score.item():.3f}') 
   
    smooth_window = 100
    smoothed_results = convolve(results,np.ones(smooth_window),mode="valid") / smooth_window
    smoothed_domain = [smooth_window+x for x in range(smoothed_results.size)] 
    plt.plot(smoothed_domain,smoothed_results,c='blue')
    plt.plot(results,alpha=0.2,c='grey') 
    plt.savefig('results.png')
    plt.close()



