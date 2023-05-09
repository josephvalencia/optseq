#!/usr/bin/env python
import torch
import argparse
import random
import numpy as np
import os
import warnings

from bioseq2seq.bin.models import restore_seq2seq_model
from bioseq2seq.bin.data_utils import iterator_from_fasta, build_standard_vocab, IterOnDevice
from bioseq2seq.inputters.corpus import maybe_fastafile_open

from design.sampler_tgt_conditional import TargetConditionalSampler
from design.sampler_src_conditional import SourceConditionalSampler

def parse_args():

    """ Parse required and optional configuration arguments"""
    parser = argparse.ArgumentParser()
    
    # optional flags
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--negative_class",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--tgt_input",default=None,help="File for translation")
    parser.add_argument("--checkpoint", "--c",help ="ONMT checkpoint (.pt)")
    parser.add_argument("--inference_mode",default ="combined")
    parser.add_argument("--attribution_mode",default="ig")
    parser.add_argument("--baseline",default="zero", help="zero|avg|A|C|G|T")
    parser.add_argument("--tgt_class",default="<PC>", help="<PC>|<NC>")
    parser.add_argument("--tgt_pos",type=int,default=1, help="token position in target")
    parser.add_argument("--max_tokens",type=int,default = 1200, help = "Max number of tokens in training batch")
    parser.add_argument("--sample_size",type=int,default = 32, help = "Max number of tokens in training batch")
    parser.add_argument("--minibatch_size",type=int,default = 16, help = "Max number of tokens in training batch")
    parser.add_argument("--name",default = "temp")
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    
    return parser.parse_args()

def run_helper(rank,args,model,vocab,use_splits=False):
    
    random_seed = 65
    random.seed(random_seed)
    random_state = random.getstate()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    target_pos = args.tgt_pos
    vocab_fields = build_standard_vocab()
    device = "cpu"
    
    # GPU training
    if args.num_gpus > 0:
        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        model.cuda()
    
    # multi-GPU training
    if args.num_gpus > 1:
        # configure distributed training with environmental variables
        os.environ['MASTER_ADDR'] = args.address
        os.environ['MASTER_PORT'] = args.port
        torch.distributed.init_process_group(
            backend="nccl",
            init_method= "env://",
            world_size=args.num_gpus,
            rank=rank)
   
    gpu = rank if args.num_gpus > 0 else -1
    world_size = args.num_gpus if args.num_gpus > 0 else 1
    offset = rank if world_size > 1 else 0
    
    class_name = args.tgt_class
    tgt_class = args.tgt_class
    if tgt_class == "PC" or tgt_class == "NC":
        tgt_class = f'<{tgt_class}>'
    if tgt_class == "STOP":
        tgt_class = "</s>"
    
    print(args.name)
    pathname = os.path.split(args.name)
    
    if pathname[0] == '':
        pathname = pathname[-1]
    else:
        pathname = pathname[0]
    if not os.path.isdir(pathname):
        os.mkdir(pathname)
    path = args.checkpoint.split('/')
    input_name = os.path.split(args.input)[-1].replace('.fa','')
    
    savefile = "{}/{}.{}.{}.{}".format(args.name,input_name,class_name,target_pos,args.attribution_mode)
    if world_size > 1 :
        savefile += f'.rank-{gpu}'
    print(savefile) 
    tscripts = []
    with maybe_fastafile_open(args.input) as fa:
        for i,record in enumerate(fa):
            #if (i % world_size) == offset:
            tscripts.append(record.id)
    
    
    print(f'INFERENCE MODE {args.inference_mode}')
    valid_iter = iterator_from_fasta(src=args.input,
                                    tgt=args.tgt_input,
                                    vocab_fields=vocab_fields,
                                    mode=args.inference_mode,
                                    is_train=False,
                                    max_tokens=args.max_tokens,
                                    external_transforms={}, 
                                    rank=rank,
                                    world_size=world_size) 
    valid_iter = IterOnDevice(valid_iter,gpu)
   
    apply_softmax = False
    model.eval()
    sep = '___________________________________' 
    kwargs = dict()
    
    print(f'Running Discrete Langevin wrt. {tgt_class} conditioned on {args.attribution_mode}')
    if args.attribution_mode == 'src': 
        attributor = SourceConditionalSampler(model,device,vocab,tgt_class,
                                            softmax=apply_softmax,sample_size=args.sample_size,
                                            minibatch_size=args.minibatch_size)
    elif args.attribution_mode == 'tgt':
        attributor = TargetConditionalSampler(model,device,vocab,tgt_class,
                                            softmax=apply_softmax,sample_size=args.sample_size,
                                            minibatch_size=args.minibatch_size)

    attributor.run(savefile,valid_iter,target_pos,args.baseline,tscripts,**kwargs)
  
def run_attribution(args,device):
    
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    vocab = checkpoint['vocab']
    model = restore_seq2seq_model(checkpoint,device,options)

    if not options is None:
        model_name = ""
        print("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
        for k,v in vars(options).items():
            print(k,v)
 
    if args.num_gpus > 1:
        torch.multiprocessing.spawn(run_helper, nprocs=args.num_gpus, args=(args,model,vocab))
    elif args.num_gpus > 0:
        run_helper(args.rank,args,model,vocab)
    else:
        run_helper(0,args,model,vocab)

if __name__ == "__main__": 

    warnings.filterwarnings("ignore")
    args = parse_args()
    device = "cpu"
    run_attribution(args,device)
