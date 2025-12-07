import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs

vocab_path = './ESPF/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 50
max_p = 545


# 用于将输入的蛋白质序列转换为固定长度的嵌入向量（embedding）和相应的输入掩码（input mask）；这个函数的主要目的是将可变长度的蛋白质序列转换为固定长度的数值表示，因为机器学习模型要求输入具有固定的维度
def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split——pbpe.process_line将氨基酸转换为对应的数字索引；然后split()方法将处理后的字符串分割为单词列表
    try:
        i1 = np.asarray(
            [words2idx_p[i] for i in t1])  # index——将t1中的每个单词（或标记）转换为相应的索引，通过words2idx_p字典实现，该字典将每个单词映射到一个唯一的整数索引
    except:
        i1 = np.array([0])  # 如果words2idx_p字典中没有某个单词，则将所有单词的索引设置为0
        # print(x)

    l = len(i1)

    if l < max_p:  # 如果序列长度小于max_p，则用0将i1填充到max_p；创建一个input_mask列表用于指示哪些部分是原始序列，哪些部分是填充的
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:  # 如果序列长度大于或等于max_p，则简单地截取i1的前max_p个元素
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask)


# 同上
def drug2emb_encoder(x):
    max_d = 50
    # max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization——初始化'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

    def __len__(self):
        'Denotes the total number of samples——表示样本总数'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data——生成一个数据样本'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]  # 从self.list_IDs列表中获取实际的索引，通常是为了确保从DataFrame中正确地提取样本
        # d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']  # 从DataFrame self.df中根据索引index，提取'SMILES'列的值
        p = self.df.iloc[index]['Target Sequence']  # 同上从DataFrame中提取'Target Sequence'列的值

        # d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)  # 使用drug2emb_encoder函数将'SMILES'字符串d编码为嵌入向量d_v和相应的输入掩码input_mask_d
        p_v, input_mask_p = protein2emb_encoder(p)  # 使用protein2emb_encoder函数将'Target Sequence'字符串p编码为嵌入向量p_v和相应的输入掩码input_mask_p#################################################

        # print(d_v.shape)
        # print(input_mask_d.shape)
        # print(p_v.shape)
        # print(input_mask_p.shape)
        y = self.labels[index]  # 从self.labels列表中，根据索引index，获取样本的标签y
        return d_v, p_v, input_mask_d, input_mask_p, y  # 返回药物嵌入向量、蛋白质嵌入向量、药物输入掩码、蛋白质输入掩码和样本标签