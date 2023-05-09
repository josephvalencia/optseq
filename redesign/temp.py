class PredictionWrapper(torch.nn.Module):
    
    def __init__(self,model,device,vocab,tgt_class,
                 softmax=True,sample_size=None,
                 minibatch_size=None):
        
        super(PredictionWrapper,self).__init__()
        self.model = model
        self.softmax = softmax 
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

    def forward(self,src,memory_lens,decoder_inputs,batch_size):
       raise NotImplementedError 

class ClassificationWrapper(PredictionWrapper):

    def class_logit_ratio(self,pred_classes,class_token):
        ''' logit of class_token minus all the rest'''
       
        counterfactual = [x for x in range(pred_classes.shape[2]) if x != class_token]
        counter_idx = torch.tensor(counterfactual,device=pred_classes.device)
        counterfactual_score = pred_classes.index_select(2,counter_idx)
        
        # for PC and NC, contrast with only the counterfactual class, the others grads are noisy
        if class_token == self.nc_token:
            return pred_classes[:,:,self.nc_token] - pred_classes[:,:,self.pc_token]
        elif class_token == self.pc_token:
            return pred_classes[:,:,self.pc_token] - pred_classes[:,:,self.nc_token]
        else:
            return pred_classes[:,:,class_token] - counterfactual_score.sum(dim=2)

    def decoder_input(self,batch_size,prefix=None):
        
        if prefix is None:
            ones =  torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
            return (self.sos_token*ones,)
        else:
            chunked = [x.repeat(batch_size,1,1) for x in torch.tensor_split(prefix,prefix.shape[0],dim=0)]
            return tuple(chunked)
    
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
        
        scores = self.model.generator(dec_out,softmax=self.softmax)
        return scores, attn
    
    def forward(self,src,memory_lens,decoder_input,batch_size,class_token,ratio=True):

        src = src.transpose(0,1)
        src, enc_states, memory_bank, src_lengths, enc_cache = self.run_encoder(src,memory_lens,batch_size)

        self.model.decoder.init_state(src,memory_bank,enc_states)
        
        for i,dec_input in enumerate(decoder_inputs):
            scores, attn = self.decode_and_generate(
                dec_input,
                memory_bank,
                memory_lengths=memory_lens,
                step=i)
        
        if ratio:
            outputs = self.class_logit_ratio(scores,class_token)
        else:
            outputs = scores[:,:,class_token]
       
        return outputs