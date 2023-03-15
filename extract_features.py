from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
import sys, os, re
import multiprocessing
from tqdm import tqdm

def convert_seq_to_bicoding(seq):
    seq = seq.replace('U', 'T')
    features = []
    feature1 = []
    feature2 = []
    feature = []
    bicoding_dict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0],'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],'n':[0,0,0,0]}
    if len(seq) < 65:
        seq = seq + 'N'*(65-len(seq))
    A_count = seq.count('A')
    C_count = seq.count('C')
    G_count = seq.count('G')
    T_count = seq.count('T')
    A_frequency = A_count / 65
    C_frequency = C_count / 65
    G_frequency = G_count / 65
    T_frequency = T_count / 65

    feature2.append(A_frequency)
    feature2.append(C_frequency)
    feature2.append(G_frequency)
    feature2.append(T_frequency)

    base_2 = ['GG', 'GA', 'GC', 'GT', 'AG', 'AA', 'AC', 'AT', 'CG', 'CA', 'CC', 'CT', 'TG', 'TA', 'TC', 'TT']
    for each_2nt in base_2:
        dou_base_count = seq.count(each_2nt)
        dou_base_frequency = dou_base_count / 64
        feature1.append(dou_base_frequency)

    for base in seq:
        feature += bicoding_dict[base]
    features = feature + feature2 + feature1
    return features

def load_data_bicoding(in_fa):
    data=[]
    for record in SeqIO.parse(in_fa, "fasta"):
        seq=str(record.seq)
        bicoding=convert_seq_to_bicoding(seq)
        data.append(bicoding)
    #print(len(data))

    return data

def load_data_bicoding_with_header(in_fa):
    data=[]
    fa_header=[]
    for record in SeqIO.parse(in_fa, "fasta"):
        seq=str(record.seq)
        bicoding=convert_seq_to_bicoding(seq)
        data.append(bicoding)
        fa_header.append(str(record.description))
    #print(len(data))

    return data, fa_header

def load_train_val_bicoding(pos_train_fa,neg_train_fa):
    data_pos_train = []
    data_neg_train = []

    data_pos_train = load_data_bicoding(pos_train_fa)
    data_neg_train = load_data_bicoding(neg_train_fa)


    data_train = np.array([_ + [1] for _ in data_pos_train] + [_ + [0] for _ in data_neg_train])
    np.random.seed(42)
    np.random.shuffle(data_train)

    X = np.array([_[:-1] for _ in data_train])
    # y = np.array([_[-1] for _ in data_train])
    y = np.array([_[-1] for _ in data_train])

    return X, y