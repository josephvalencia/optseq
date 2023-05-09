import sys,re
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import pandas as pd
import numpy as np
from bioseq2seq.bin.evaluate import ficketTestcode
from utils import getLongestORF
from cai import CAI
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def longest_orf_len(seq):
    s,e = getLongestORF(seq)
    return e-s

def gc_content(rna):
    count = 0
    for c in rna:
        if c == 'G' or c == 'C':
            count +=1
    return count / len(rna)

def parse(filename,usage_file,test_file):
    
    df = pd.read_csv(test_file,sep='\t')
    df = df.set_index('ID')
    big = 0 
    total = 0 
    storage = []  
    threshold = 0.9
    big_storage = []
    cai = CAI()
    cai.build_codon_table(usage_file)
    for record in SeqIO.parse(filename,'fasta'):
        description = record.description
        score_change = r'(P\(<PC>\)|P\(<NC>\)):(.*)->(.*)' 
        match = re.search(score_change,description)
        original_id = record.id.split('-')[0] 
        wildtype = df.loc[original_id]['RNA']
        designed = str(record.seq)
        
        s1,e1 = getLongestORF(wildtype)
        s2,e2 = getLongestORF(designed)
        if not(any([c == -1 for c in [s1,e1,s2,e2]])):
            cds_wild = wildtype[s1:e1]
            cds_designed = designed[s2:e2]
            cai_wild = cai.calculate(cds_wild)
            cai_designed = cai.calculate(cds_designed)
        else:
            cai_wild = 0.0
            cai_designed = 0.0

        mismatches = 0 
        running_shared = ''
        print(original_id)
        interesting = ['NM_001110187.2','NM_001261591.1','NM_002824.6','NM_174328.2']
        counter = Counter()
        for i,(c1,c2) in enumerate(zip(wildtype,designed)):
            if c1 != c2:
                mismatches+=1
                running_shared+=c1
            else:
                mismatch_len = len(running_shared)
                if mismatch_len > 0 and original_id in interesting: 
                    print(f'# consecutive mismatches = {mismatch_len}, modifed substring = {running_shared}, loc = {i-mismatch_len}:{i}')
                    counter.update([mismatch_len])
                running_shared = ''
        
        matches = len(wildtype) - mismatches
        print(f'matches = {matches} / {len(wildtype)} = {matches/len(wildtype):.3f}') 
        print(counter.most_common())
        gc_wild = gc_content(wildtype)
        gc_designed = gc_content(designed)
        ficket_wild = ficketTestcode(wildtype)
        ficket_designed = ficketTestcode(designed)
        orf_len_wild = longest_orf_len(wildtype)
        orf_len_designed = longest_orf_len(designed)
        
        pct_change = lambda x,y : (y-x) / x if x > 0 else 0.0 
        if match:
            total+=1
            tgt_class = match.group(1)
            start_prob = float(match.group(2))
            end_prob = float(match.group(3))
            if abs(end_prob-start_prob) >= threshold:
                big +=1
                big_storage.append(record) 
            entry = {'tscript' : original_id, 'pct_id' : matches / len(wildtype), 'delta_class_prob' : end_prob - start_prob, 'GC_diff' : pct_change(gc_wild,gc_designed),\
                    'Fickett_diff' : pct_change(ficket_wild,ficket_designed), 'ORF_len_diff' : pct_change(orf_len_wild,orf_len_designed), 'CAI_diff' :  pct_change(cai_wild,cai_designed)}
        storage.append(entry)
    
    df = pd.DataFrame(storage)
    big_df = df[df['delta_class_prob'] > threshold] 
    print(big_df)
    sns.set_style('whitegrid')
    g = sns.swarmplot(data=big_df[['GC_diff','Fickett_diff','CAI_diff']])
    feature_file = filename.split('.fa')[0]+f'_feature_plot.svg'
    plt.savefig(feature_file)
    plt.close()
    print('ALL') 
    print(df.describe())
    print('BIG') 
    print(big_df.describe())
    print(f'# delta({tgt_class}) >= {threshold} : {big}/{total} = {big/total:.3f}')
    
    significant_file = filename.split('.fa')[0]+f'_thresh_{threshold}.fa'
    print(f'significant redesigns saved at {significant_file}') 
    with open(significant_file,'w') as outFile:
        SeqIO.write(big_storage,outFile,'fasta')

if __name__ == "__main__":
    
    test_file = '/home/bb/valejose/home/bioseq2seq/data/mammalian_200-1200_test_nonredundant_80.csv'
    parse(sys.argv[1],sys.argv[2],test_file)
