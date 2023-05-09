import torch
from functools import reduce
from typing import Union
from bioseq2seq.utils.loss import NMTLossCompute
from optseq.onehot import TensorToOneHot
from optseq.onehot import OneHotToEmbedding

def get_module_by_name(parent: Union[torch.Tensor, torch.nn.Module],
                               access_string: str):
    names = access_string.split(sep='.')
    return reduce(getattr, names, parent)

def setup_onehot_embedding(model : torch.nn.Module):
    # augment Embedding with one hot utilities
    embedding_modulelist = get_module_by_name(model,'encoder.embeddings.make_embedding.emb_luts')
    old_embedding = embedding_modulelist[0] 
    onehot_embed_layer = TensorToOneHot(old_embedding)
    dense_embed_layer = OneHotToEmbedding(old_embedding)
    embedding_modulelist[0] = dense_embed_layer
    return model, onehot_embed_layer

def readable_fn(src_vocab,src):
    
    saved_src = src.detach().cpu().numpy()
    if saved_src.shape[1] > 1:
        storage = []
        for b in range(saved_src.shape[0]):
            raw = [src_vocab.itos[c] for c in saved_src[b,:].ravel()]
            storage.append(raw)
        return storage
    else:
        saved_src = saved_src.ravel()
        return ''.join([src_vocab.itos[c] for c in saved_src])

class TranslationWrapper(torch.nn.Module):

    def __init__(self,model,iterator):

        super(TranslationWrapper,self).__init__() 
        self.model = model
        self.setup_dummybatch(iterator) 
        criterion = torch.nn.NLLLoss(ignore_index=1,reduction='none')
        self.train_loss = NMTLossCompute(criterion,generator=model.generator)

    def setup_dummybatch(self,iterator):
        
        iterator = iter(iterator)
        first_batch = next(iterator)
        src,src_lens = first_batch.src 
        self.tgt = first_batch.tgt
        self.src_lengths = src_lens
        self.batch = first_batch 

    def forward(self,src):
    
        #src = src.transpose(0,1)
        
        outputs, enc_attns, attns = self.model(
            src, self.tgt, self.src_lengths,
            bptt=False,with_align=False)
        loss, batch_stats = self.train_loss(
            self.batch,
            outputs,
            attns,
            normalization=40,
            shard_size=0,
            trunc_start=0,
            trunc_size=self.tgt.shape[0])
     
        start_weight = 1
        lambd = 0.01
        weights = start_weight * torch.exp(-lambd * torch.arange(loss.shape[0])).to(loss.device) 
        weighted = -torch.dot(weights,loss)
        return weighted 
    
