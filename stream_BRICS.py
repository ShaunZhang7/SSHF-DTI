import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Recap

vocab_path = './ESPF/protein_codes_uniprot.txt'  # 加载词汇表‌：‌使用codecs.open打开词汇表文件protein_codes_uniprot.txt
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1,
           separator='')  # 初始化BPE编码器‌：‌使用加载的词汇表初始化BPE编码器，‌merges参数设为-1表示使用所有合并规则，‌separator为空字符串
sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv', keep_default_na=False)

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0,
                                         len(idx2word_p))))  # 构建映射‌：‌从CSV文件中提取index列的值，‌构建索引到单词（‌idx2word_p）‌和单词到索引（‌words2idx_p）‌的映射
# print(words2idx_p)

frequency_dict = dict(
    zip(idx2word_p, sub_csv['frequency'].values))  ####################################################增加的

# ------------------------------------------------------------
vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
# sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

# ------------------------------------------------------------
# sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
sub_csv1 = pd.read_csv('./ESPF_BRICS/subword_units_map_chembl.csv')

idx2word_d1 = sub_csv1['index'].values
words2idx_d1 = dict(zip(idx2word_d1, range(0, len(idx2word_d1))))

max_d = 30  #########################################################################################################205
max_p = 1200  #########################################################################################################545


def protein2emb_encoder(x):
    max_p = 1200  #####################################################################################################545
    # print(x) #MVVMNSLRVILQASPGKLLWRKFQIPRFMPARPCSLYTCTY...
    # 使用BPE分词
    t1 = pbpe.process_line(x).split()  # split(BPE)

    # 映射到索引
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:  # # 如果发生任何异常(t1中不在字典内)，将 i1 设置为只包含一个元素0的NumPy数组
        #i1 = np.array([0])
        i1 = np.asarray([words2idx_p.get(i, 0) for i in t1])
        print("存在BPE靶标子结构无映射！")
        print(x)
    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    # 根据数组i的内容修改掩码input_mask ####################################################################################增加的掩码修改
    for idx, value in enumerate(i):
        if value == 0:
            input_mask[idx] = 0

    return i, np.asarray(input_mask)


def fragment_recursive(mol, frags):
    try:
        # 找到所有 BRICS 键
        bonds = list(BRICS.FindBRICSBonds(mol))
        if len(bonds) == 0:
            # 如果没有 BRICS 键，将整个分子作为片段
            frags.append(Chem.MolToSmiles(mol))
            return frags
        # 获取 BRICS 键的原子索引和键索引
        idxs, labs = list(zip(*bonds))
        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())
        # 按键索引排序
        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]
        # 断裂第一个 BRICS 键
        broken = Chem.FragmentOnBonds(mol, bondIndices=[bond_idxs[0]], dummyLabels=[(0, 0)])
        # 获取所有片段（可能多于 2 个）
        fragments = Chem.GetMolFrags(broken, asMols=True)
        # 将第一个片段加入结果
        frags.append(Chem.MolToSmiles(fragments[0]))
        # 递归处理剩余的片段
        for frag in fragments[1:]:
            fragment_recursive(frag, frags)
        return frags
    except Exception as e:
        print(f"Error: {e}")
        return frags
# --------------------- 将*号去掉 ---------------------
def remove_dummy(smiles):
    try:
        stripped_smi = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(stripped_smi)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(e)
        return None
def remove_star_brackets(patterns):
    result = []
    for pattern in patterns:
        # 找到所有类似 [*] 的结构
        stack = []
        indices = []
        for i, char in enumerate(pattern):
            if char == '[':
                stack.append(i)
            elif char == ']':
                if stack:
                    start = stack.pop()
                    if pattern[start + 1] == '*' or pattern[start + 1].isdigit():
                        indices.append((start, i))
        # 根据找到的索引分割字符串
        parts = []
        prev = 0
        for start, end in indices:
            if prev < start:
                parts.append(pattern[prev:start])
            prev = end + 1
        if prev < len(pattern):
            parts.append(pattern[prev:])
        # 去掉空字符串，并将中间部分进一步分开
        final_parts = []
        for part in parts:
            if part:
                # 如果 part 包含类似 [*] 的结构，进一步分割
                if '[' in part and ']' in part:
                    # 找到类似 [*] 的结构
                    sub_indices = []
                    sub_stack = []
                    for i, char in enumerate(part):
                        if char == '[':
                            sub_stack.append(i)
                        elif char == ']':
                            if sub_stack:
                                sub_start = sub_stack.pop()
                                if part[sub_start + 1] == '*' or part[sub_start + 1].isdigit():
                                    sub_indices.append((sub_start, i))
                    # 根据 sub_indices 分割 part
                    sub_parts = []
                    sub_prev = 0
                    for sub_start, sub_end in sub_indices:
                        if sub_prev < sub_start:
                            sub_parts.append(part[sub_prev:sub_start])
                        sub_prev = sub_end + 1
                    if sub_prev < len(part):
                        sub_parts.append(part[sub_prev:])
                    # 去掉空字符串并添加到 final_parts
                    sub_parts = [p for p in sub_parts if p]
                    final_parts.extend(sub_parts)
                else:
                    final_parts.append(part)
        # 将结果添加到最终列表
        result.extend(final_parts)
    return result

# 定义一个函数来加载CSV文件并返回字典
def load_words2idx():
    sub_csv1 = pd.read_csv('./ESPF_BRICS/subword_units_map_chembl.csv')
    idx2word_d1 = sub_csv1['index'].values
    words2idx_d1 = dict(zip(idx2word_d1, range(0, len(idx2word_d1))))
    return words2idx_d1
# 定义函数来更新CSV文件
def update_csv(new_words):
    sub_csv1 = pd.read_csv('./ESPF_BRICS/subword_units_map_chembl.csv')  ########
    # 获取当前CSV文件中的最大索引
    max_idx = sub_csv1['None'].max()  # 假设'序号'是第一列的列名
    # print(max_idx)
    new_rows = []
    for word in new_words:
        max_idx += 1
        new_rows.append({
            'None': max_idx,  # 第一列
            'level_0': max_idx,  # 第二列（这里假设和序号相同）
            'index': word,  # 第三列
            'frequency': 1  # 第四列
        })
    # 将新行添加到DataFrame
    new_df = pd.DataFrame(new_rows)
    # 将新DataFrame追加到原始CSV
    updated_csv = pd.concat([sub_csv, new_df], ignore_index=True)
    # 保存到CSV文件
    updated_csv.to_csv('./ESPF_BRICS/subword_units_map_chembl.csv', index=False)


def drug2emb_encoder(x):
    max_d = 30  #######################################################################################################50
    # max_d = 100
    t1 = dbpe.process_line(x).split()  # 使用dbpe处理输入x并分割
    #words2idx_d = load_words2idx()  ##################################################################################增加
    words2idx_d1 = load_words2idx()

    try:
        i1 = np.asarray([words2idx_d1[i] for i in t1])  # 尝试将t1中的每个词转换为索引，‌若失败则索引为0
        ###i1 = np.asarray([words2idx_d.get(i, 0) for i in clean_fragments])#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!只将不存在的子结构置0（原码是全置0）
        #print(i1)
    except:
        # i1 = np.array([0])
        print("存在BPE药物子结构无映射!")

        ###BRICS
        t1 = Chem.MolFromSmiles(x)  # ouput example: <rdkit.Chem.rdchem.Mol object at 0x000002D306105030>
        fragments = fragment_recursive(t1, [])  # ouput example: ['*c1ncc2c(n1)-c1ccc(Cl)cc1C(c1c(F)cccc1F)=NC2', '*N*', '*c1ccc(*)cc1', '*C(=O)O']
        t1 = [remove_dummy(smi) for smi in fragments]  # ouput example: ['Fc1cccc(F)c1C1=NCc2cncnc2-c2ccc(Cl)cc21', 'N', 'c1ccccc1', 'O=CO']
        # fragments = BRICS.BRICSDecompose(t1, minFragmentSize=2)
        # t1 = remove_star_brackets(fragments)

        #i1 = np.asarray([words2idx_d1.get(i, 0) for i in t1])######

        # 找出t1中无法映射到索引的词
        #'''
        #unknown_words = [word for word in t1 if word not in words2idx_d1]
        # 使用集合推导式去重，然后转回列表
        unknown_words = list({word for word in t1 if word not in words2idx_d1})
        # 如果有未知词，更新CSV和字典
        if unknown_words:
            print("存在未知子结构，增加生成药物词典内容...")
            update_csv(unknown_words)
            # 重新加载字典
            words2idx_d1 = load_words2idx()
        i1 = np.asarray([words2idx_d1.get(i, 0) for i in t1])
        #print(i1)
        #'''

    # 序列填充或截断
    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    # 返回处理后的嵌入向量及其输入掩码
    # print(i)
    return i, np.asarray(input_mask)  ###################################################################################


class BIN_Data_Encoder(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti):  # (df_train.index.values, df_train.Label.values, df_train)
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.df = df_dti

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        # d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']

        # d_v = drug2single_vector(d)
        p_v, input_mask_p = protein2emb_encoder(p)
        d_v, input_mask_d = drug2emb_encoder(d)  ########################################################################

        # print(d_v.shape)
        # print(input_mask_d.shape)
        # print(p_v.shape)
        # print(input_mask_p.shape)
        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, y

