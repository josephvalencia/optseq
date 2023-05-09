import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torchtext.transforms as T
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from collections import OrderedDict, Counter, defaultdict
import torchdata.datapipes as dp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PolysomeEGFP(dp.iter.IterDataPipe):

    def __init__(self,df):
        self.df = df

    def __iter__(self):
        for sample in self.df.to_dict('records'):
            counts = np.asarray([sample[f'{x}'] for x in range(14)])
            ribosomes = np.asarray([sample[f'r{x}'] for x in range(14)])
            reads = sample['total_reads'] 
            example = {'utr' : list(sample['utr']), 'mrl' : sample['mrl']}     
            yield example 

def add_mrl(df):
    '''Add the mean ribosome load to the dataframe'''

    df['mrl'] = StandardScaler().fit_transform(df['rl'].to_numpy().reshape(-1,1)).squeeze().tolist()
    return df

def make_dataset_splits(df_test,df_train,random_seed):

    splits = train_test_split(df_train,test_size=0.1,random_state=random_seed)
    df_train,df_valid = splits[0],splits[1] 
    return PolysomeEGFP(add_mrl(df_test)),PolysomeEGFP(add_mrl(df_valid)),PolysomeEGFP(add_mrl(df_train))

def apply_transform(x):
    '''Transforms a string of nucleotides into a list of integers'''

    mapping = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}
    utr = [mapping[x] for x in x['utr']] 
    return torch.tensor(utr,dtype=torch.int64),torch.tensor(x['mrl'],dtype=torch.float32)

def dataloader_from_dataset(dataset,batch_size):
    '''Use datapipes to make dataloaders and numericalize utr seq'''

    dataset = dataset.map(apply_transform)
    dataset = dataset.batch(batch_size)
    dataset = dataset.rows2columnar(['utr','mrl'])
    dataloader = DataLoader(dataset,batch_size=None)
    return dataloader

def parse_egfp_polysome(file1,file2):

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    new_cols = { x : y for x,y in zip(df2.columns,df1.columns) }
    df2 = df2.rename(columns=new_cols)
    combined = pd.concat([df1,df2])

    test_n = 20000
    total_n = 280000
    combined.sort_values(by='total_reads',ascending=False)
    combined = combined[combined['total_reads'] > 0.0] 
    df_test = combined[:test_n]
    df_train = combined[test_n:total_n]
    return df_test, df_train

