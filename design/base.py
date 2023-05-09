import torch
import torch.nn.functional as F
from functools import reduce
from typing import Union
from bioseq2seq.utils.loss import NMTLossCompute
from optseq.onehot import TensorToOneHot
from optseq.onehot import OneHotToEmbedding

class PredictionWrapper(torch.nn.Module):
    
    def __init__(self,model,softmax):
        
        super(PredictionWrapper,self).__init__()
        self.model = model
        self.softmax = softmax 

    def forward(self,src,memory_lens,decoder_inputs,batch_size):
        
        src = src.transpose(0,1)
        src, enc_states, memory_bank, src_lengths, enc_cache = self.run_encoder(src,memory_lens,batch_size)

        self.model.decoder.init_state(src,memory_bank,enc_states)
        
        for i,dec_input in enumerate(decoder_inputs):
            scores, attn = self.decode_and_generate(
                dec_input,
                memory_bank,
                memory_lengths=memory_lens,
                step=i)
        
        return scores

    def run_encoder(self,src,src_lengths,batch_size):

        enc_states, memory_bank, src_lengths, enc_cache = self.model.encoder(src,src_lengths,grad_mode=False)
        
        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank).long().fill_(memory_bank.size(0))

        return src, enc_states, memory_bank, src_lengths,enc_cache

    def decode_and_generate(self,decoder_in, memory_bank, memory_lengths, step=None):
        
        dec_out, dec_attn = self.model.decoder(decoder_in,
                                            memory_bank,
                                            memory_lengths=memory_lengths,
                                            step=step,grad_mode=True)

        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        
        #print(f'dec_out before squeeze = {dec_out.shape}')    
        #scores = self.model.generator(dec_out.squeeze(0),softmax=self.softmax)
        scores = self.model.generator(dec_out,softmax=self.softmax)
        return scores, attn

class Attribution:
    '''Base class for attribution experiments '''
    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,minibatch_size=None):
        
        self.device = device
        self.model = model
        self.sample_size = sample_size
        self.minibatch_size = minibatch_size
        self.softmax = softmax
        self.tgt_vocab = vocab['tgt'].base_field.vocab
        self.class_token = self.tgt_vocab[tgt_class] if tgt_class != 'GT' else 'GT'
        self.sos_token = self.tgt_vocab['<s>']
        self.pc_token = self.tgt_vocab['<PC>']
        self.nc_token = self.tgt_vocab['<NC>']
        self.src_vocab = vocab["src"].base_field.vocab
        self.predictor = PredictionWrapper(self.model,self.softmax)
    
    def decoder_input(self,batch_size,prefix=None):
        
        if prefix is None:
            ones =  torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
            return (self.sos_token*ones,)
        else:
            chunked = [x.repeat(batch_size,1,1) for x in torch.tensor_split(prefix,prefix.shape[0],dim=0)]
            return tuple(chunked)
            
    def get_raw_src(self,src):
        saved_src = src.detach().cpu().numpy()
        if saved_src.shape[0] > 1:
            storage = []
            for b in range(saved_src.shape[0]):
                raw = [self.src_vocab.itos[c] for c in saved_src[b,:].ravel()]
                storage.append(raw)
            return storage
        else:
            saved_src = saved_src.ravel()
            return [self.src_vocab.itos[c] for c in saved_src]

    def get_src_token(self,char):
        return self.src_vocab.stoi[char]
    
    def get_tgt_string(self,char):
        return self.tgt_vocab.itos[char]
    
    def input_grads(self,outputs,inputs):
        ''' per-sample gradient of outputs wrt. inputs '''
        
        total_pred = outputs.sum()
        total_pred.backward(torch.ones_like(total_pred),inputs=inputs)
        grads = inputs.grad
        return grads
            
    def class_logit_ratio(self,pred_classes,class_token):
        ''' logit of class_token minus all the rest'''
       
        counterfactual = [x for x in range(pred_classes.shape[2]) if x != class_token]
        counter_idx = torch.tensor(counterfactual,device=pred_classes.device)
        counterfactual_score = pred_classes.index_select(2,counter_idx)
        class_score = pred_classes[:,:,class_token] 
        
        # for PC and NC, contrast with only the counterfactual class, the others grads are noisy
        if class_token == self.nc_token:
            return pred_classes[:,:,self.nc_token] - pred_classes[:,:,self.pc_token]
        elif class_token == self.pc_token:
            return pred_classes[:,:,self.pc_token] - pred_classes[:,:,self.nc_token]
        else:
            return pred_classes[:,:,class_token] - counterfactual_score.sum(dim=2)
   
    def predict_logits(self,src,src_lens,decoder_input,batch_size,class_token,ratio=True):

        pred_classes = self.predictor(src,src_lens,decoder_input,batch_size)
        probs = F.softmax(pred_classes,dim=-1)
        
        if ratio:
            outputs = self.class_logit_ratio(pred_classes,class_token)
        else:
            outputs = pred_classes[:,:,class_token]
       
        return outputs, probs

    def translation_loss(self,src,tgt,src_lengths,batch):
    
        src = src.transpose(0,1)
        print(src.shape,tgt.shape,src_lengths) 
        outputs, enc_attns, attns = self.model(
            src, tgt, src_lengths, bptt=False,
            with_align=False)
        
        loss, batch_stats = self.train_loss(
            batch,
            outputs,
            attns,
            normalization=40,
            shard_size=0,
            trunc_start=0,
            trunc_size=tgt.shape[0])
     
        start_weight = 1
        lambd = 0.01
        weights = start_weight * torch.exp(-lambd * torch.arange(loss.shape[0])).to(loss.device) 
        weighted = -torch.dot(weights,loss)
        return weighted 

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        self.run_fn(savefile,val_iterator,target_pos,baseline,transcript_names)
    
def get_module_by_name(parent: Union[torch.Tensor, torch.nn.Module],
                               access_string: str):
    names = access_string.split(sep='.')
    return reduce(getattr, names, parent)

class OneHotGradientAttribution(Attribution):

    def __init__(self,model,device,vocab,tgt_class,softmax=False,sample_size=None,minibatch_size=None):
        
        # augment Embedding with one hot utilities
        embedding_modulelist = get_module_by_name(model,'encoder.embeddings.make_embedding.emb_luts')
        old_embedding = embedding_modulelist[0] 
        self.onehot_embed_layer = TensorToOneHot(old_embedding)
        dense_embed_layer = OneHotToEmbedding(old_embedding)
        embedding_modulelist[0] = dense_embed_layer
        self.predictor = PredictionWrapper(model,softmax)
        
        # computes position-wise NLLoss
        criterion = torch.nn.NLLLoss(ignore_index=1,reduction='none')
        self.train_loss = NMTLossCompute(criterion,generator=model.generator)
        
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,minibatch_size=minibatch_size)
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        raise NotImplementedError
