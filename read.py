import pandas as pd
import random
import csv
import pickle
import numpy as np

from rdkit import Chem
from rdkit.Chem import BRICS

from rdkit.Chem import Recap

import torch
import torch.nn as nn
import torch.nn.functional as F



# 保存完全相同的，其它的也保存
'''
file1 = pd.read_csv('D:\\Shaun\\SSHF-DTI\\dataset\\DAVIS\\train&val.csv')  # 替换为文件1的路径
file2 = pd.read_csv('D:\\Shaun\\SSHF-DTI\\dataset\\DAVIS\\test.csv')  # 替换为文件2的路径

# 使用 merge 函数筛选 SMILES 和 Target Sequence 列的值都相同的行
same_rows = pd.merge(file1, file2[['SMILES', 'Target Sequence']],
                     on=['SMILES', 'Target Sequence'],
                     how='inner')
# 筛选不同的行
different_rows = file1[~file1.set_index(['SMILES', 'Target Sequence']).index.isin(file2.set_index(['SMILES', 'Target Sequence']).index)]
# 保存相同的行到新的 CSV 文件
same_rows.to_csv('D:\\Shaun\\SSHF-DTI\\dataset\\DAVIS\\repeat.csv', index=False)
# 保存不同的行到新的 CSV 文件
different_rows.to_csv('D:\\Shaun\\SSHF-DTI\\dataset\\DAVIS\\norepeat.csv', index=False)

# 保存任一项相同的，其它的也保存
file1 = pd.read_csv('D:\\Shaun\\SSHF-DTI\\dataset\\DAVIS\\train&val.csv')  # 替换为文件1的路径
file2 = pd.read_csv('D:\\Shaun\\SSHF-DTI\\dataset\\DAVIS\\test.csv')  # 替换为文件2的路径
# 提取文件2中的 SMILES 和 Target Sequence 列的值
file2_smiles = set(file2['SMILES'])
file2_target_sequence = set(file2['Target Sequence'])
# 筛选文件1中 SMILES 或 Target Sequence 在文件2中出现过的行
#filtered_rows = file1[file1['SMILES'].isin(file2_smiles) ]##############################################################
filtered_rows = file1[file1['SMILES'].isin(file2_smiles) | file1['Target Sequence'].isin(file2_target_sequence)]
# 筛选文件1中 SMILES 和 Target Sequence 在文件2中未出现过的行
#unfiltered_rows = file1[~file1['SMILES'].isin(file2_smiles)]
unfiltered_rows = file1[~(file1['SMILES'].isin(file2_smiles) | file1['Target Sequence'].isin(file2_target_sequence))]
# 保存出现的行到新的 CSV 文件
filtered_rows.to_csv('D:\\Shaun\\SSHF-DTI\\dataset\\DAVIS\\exit_true.csv', index=False)
# 保存未出现的行到新的 CSV 文件
unfiltered_rows.to_csv('D:\\Shaun\\SSHF-DTI\\dataset\\DAVIS\\exit_false.csv', index=False)
'''


#清洗CSV文件中index列的重复内容，提供详细统计信息
"""
def clean_duplicate_index_detailed(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    print(f"原始数据统计:")
    print(f"- 总行数: {len(df)}")
    print(f"- 唯一index数量: {df['index'].nunique()}")
    print(f"- 重复index数量: {len(df) - df['index'].nunique()}")

    # 找出所有重复的index及其出现次数
    index_counts = df['index'].value_counts()
    duplicate_indices = index_counts[index_counts > 1]

    if len(duplicate_indices) > 0:
        print(f"\n发现 {len(duplicate_indices)} 个重复的index:")
        for idx, count in duplicate_indices.items():
            print(f"  '{idx}': 出现 {count} 次")

            # 显示重复行的详细信息
            duplicate_rows = df[df['index'] == idx]
            print(f"    所在行: {list(duplicate_rows['None'].values)}")

        # 保留每个index第一次出现的行
        df_cleaned = df.drop_duplicates(subset=['index'], keep='first')

        print(f"\n清洗结果:")
        print(f"- 删除行数: {len(df) - len(df_cleaned)}")
        print(f"- 保留行数: {len(df_cleaned)}")
        print(f"- 唯一index数量: {df_cleaned['index'].nunique()}")

        # 保存清洗后的数据
        df_cleaned.to_csv(csv_file_path, index=False)
        print(f"\n已保存清洗后的数据到: {csv_file_path}")

        return df_cleaned
    else:
        print("\n没有发现重复的index，数据无需清洗")
        return df

# 使用示例
if __name__ == "__main__":
    csv_file = '.\SSHF-DTI\ESPF_BRICS\subword_units_map_chembl.csv'
    cleaned_df = clean_duplicate_index_detailed(csv_file)
"""

# 按'name'列去重，保留第一个出现的行
'''file_path = r"D:\Shaun\SSHF-DTI\ESPF_DDI\subword_units_map_chembl.csv"
df = pd.read_csv(file_path)

# 按'name'列去重，保留第一个出现的行
df_deduplicated = df.drop_duplicates(subset=['index'], keep='first')

# 保存处理后的文件（覆盖原文件或另存为新文件）
output_path = r"D:\Shaun\SSHF-DTI\ESPF_DDI\subword_units_map_chembl_deduplicated.csv"
df_deduplicated.to_csv(output_path, index=False)

print("去重完成！已保存到:", output_path)'''

# DTA预测散点图
'''import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
data = pd.read_csv('D:\\Shaun\\SSHF-DTA\\dataset\\Test\\Kang\\predict1_with_results_MolTrans.csv')

# 提取实际值和预测值
actual_values = data['Label']
predicted_values = data['predict']

# 创建散点图
plt.figure(figsize=(10, 8)) #数据点密集在左下角，表明大部分预测值较低，实际值也较低，
# 但未提及具体颜色和大小要求，这里使用默认设置，即alpha=0.3来控制点的透明度
#plt.scatter(predicted_values, actual_values, alpha=1)
plt.scatter(
    predicted_values,
    actual_values,
    alpha=1,          # 透明度保持1（不透明）
    c='darkblue',         # 设置颜色为海军蓝
    s=10              # 设置点的大小（默认20，这里缩小为10）
)
# 设置坐标轴上下限，由于未给出具体范围，这里使用默认值，
# Matplotlib会根据数据自动调整显示范围，但可以通过plt.xlim()和plt.ylim()手动设置
plt.xlim(4, 11)  # 设置x轴范围为0到12
plt.ylim(4, 11)  # 设置y轴范围为4到11

# 添加y=x参考线（红色虚线）
plt.plot([4, 11], [4, 11],
         color='red',
         linestyle='--',
         linewidth=2,
         label='y = x')  # 添加图例标签

# 添加标题和标签
plt.title('Scatter Plot of Actual vs Predicted Values')
plt.xlabel('Prediction')
plt.ylabel('Actual Value')

# 显示网格
plt.grid(True, alpha=0.5)

# 显示图像
plt.show()'''

#SMILES检查
'''
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from tqdm import tqdm  # 可选：用于显示进度条

def check_smiles_validity(smiles):
    """
    检查SMILES的价键有效性
    返回值：
    - 'valid'：价键正确且可解析
    - 'valence_error'：价键错误
    - 'invalid'：无法解析的SMILES格式
    """
    # 尝试解析SMILES（不进行自动价键检查）
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    if mol is None:
        return 'invalid'  # 无法解析的基础格式错误

    try:
        # 显式执行价键检查（使用 SANITIZE_PROPERTIES）
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_PROPERTIES)
        return 'valid'
    except ValueError as e:
        if 'Valence' in str(e):
            return 'valence_error'
        else:
            return 'invalid'  # 其他化学规则错误

# 读取CSV文件
input_file = 'D:\\Shaun\\SSHF-DTA\\dataset\\Kd\\test.csv'
df = pd.read_csv(input_file)

# 检查每个SMILES（使用tqdm显示进度条）
tqdm.pandas(desc="Processing SMILES")
df['validation_status'] = df['SMILES'].progress_apply(check_smiles_validity)

# 分割数据
valid_df = df[df['validation_status'] == 'valid'].drop(columns=['validation_status'])
invalid_df = df[df['validation_status'] != 'valid'].drop(columns=['validation_status'])

# 保存结果
valid_file = 'D:\\Shaun\\SSHF-DTA\\dataset\\Kd\\test_valid.csv'
invalid_file = 'D:\\Shaun\\SSHF-DTA\\dataset\\Kd\\test_invalid.csv'

valid_df.to_csv(valid_file, index=False)
invalid_df.to_csv(invalid_file, index=False)

print(f"""
处理完成！
有效SMILES已保存至：{valid_file}（共{len(valid_df)}条）
无效SMILES已保存至：{invalid_file}（共{len(invalid_df)}条，包含：\n
 - 价键错误：{len(df[df['validation_status'] == 'valence_error'])}条
 - 格式错误：{len(df[df['validation_status'] == 'invalid'])}条
""")

mols = [Chem.MolFromSmiles(s) for s in invalid_df['SMILES']] #  if pd.notnull(s)
mols = [m for m in mols if m is not None]
# 批量分解分子
hierarchies = [Recap.RecapDecompose(mol) for mol in mols]

for i in new:
    new.UpdatePropertyCache(strict=False)
#纠正不正确的化学键
Chem.MolToSmiles(new[1],True)
'''

#Scare Data折线图
'''import matplotlib.pyplot as plt
import numpy as np

# 数据配置
models = {
    "TransConv":    [0.861, 0.847, 0.815, 0.771],
    "MolTrans":     [0.855, 0.835, 0.803, 0.769],
    "DeepDTI":      [0.852, 0.830, 0.769, 0.660],
    "DeepConv-DTI": [0.845, 0.824, 0.792, 0.725],
    "DeepDTA":      [0.839, 0.819, 0.788, 0.762]
}
x_labels = ['70%', '80%', '90%', '95%']
x = np.arange(len(x_labels))

# 创建画布
plt.figure(figsize=(10, 6), dpi=100)

# 绘制折线
for model, values in models.items():
    plt.plot(x, values, linestyle='-', linewidth=2, label=model)

# 图表装饰
plt.title("AUC-ROC Performance with Different Missing Data Ratios", fontsize=14, pad=20)
plt.xlabel("Percentage of Missing Dataset", fontsize=12)
plt.ylabel("AUC-ROC", fontsize=12)
plt.xticks(x, x_labels)
plt.ylim(0.73, 0.87)

# 移除网格线和调整图例位置
plt.grid(False)  # 关闭网格线
plt.legend(
    loc='lower left',          # 定位在左下角
    bbox_to_anchor=(0.02, 0.02),  # 微调位置（距离左/下边距2%）
    frameon=True,              # 显示图例外框
    fontsize=10                # 调小字体
)

plt.tight_layout()
plt.show()
# 保存图片（取消注释使用）
# plt.savefig('dti_performance.png', bbox_inches='tight')'''


# 更改列名(针对大型数据集)
'''df = pd.read_csv('D:\\Desktop\\DTI药物靶标亲和力预测\\TEFDTA-master\\data\\BindingDB\\BindingDB_train.csv')
df.rename(columns={
    'iso_smiles': 'SMILES',
    'target_sequence': 'Target Sequence',
    'affinity': 'Label'
}, inplace=True)
# 将修改后的DataFrame写回CSV文件
df.to_csv('D:\\Desktop\\DTI药物靶标亲和力预测\\TEFDTA-master\\data\\BindingDB\\train.csv', index=False)
'''

# 每隔10行抽取一行
'''
input_file = 'D:\\Shaun\\SSHF-DTI\\dataset\\DDI\\add_true.csv'
df = pd.read_csv(input_file)
# 创建两个空的DataFrame来分别存储抽取的行和未抽取的行
sampled_df = pd.DataFrame(columns=df.columns)
non_sampled_df = pd.DataFrame(columns=df.columns)


step = 10 # 每隔10行抽取一行
for i in range(len(df)):
    if i % step == 0:
        # 如果是每隔10行的那一行，则添加到sampled_df
        sampled_df = pd.concat([sampled_df, df.iloc[[i]]], ignore_index=True)
    #else:
        # 否则，添加到non_sampled_df
        #non_sampled_df = pd.concat([non_sampled_df, df.iloc[[i]]], ignore_index=True)

# 将抽取的行保存到新的CSV文件
sampled_output_file = 'D:\\Shaun\\SSHF-DTI\\dataset\\DDI\\add.csv'
sampled_df.to_csv(sampled_output_file, index=False)
# 将未抽取的行保存到另一个新的CSV文件
#non_sampled_output_file = 'D:\\Shaun\\SSHF-DTA\\dataset\\Kd1\\train.csv'
#non_sampled_df.to_csv(non_sampled_output_file, index=False)
print(f"抽取的行已保存至文件：{sampled_output_file}")
#print(f"未抽取的行已保存至文件：{non_sampled_output_file}")
'''

# 抽取拼接（也可直接做拼接: num_rows_to_sample = len(df)）
'''
# 读取CSV文件
file_path = 'D:\\Shaun\\SSHF-DTI\\dataset\\BIOSNAP\\all.csv' #Add
df = pd.read_csv(file_path)
# 指定要随机抽取的行数
num_rows_to_sample = len(df) # 这里以10行为例
# 如果数据集小于要抽取的行数，则进行警告或处理
if num_rows_to_sample > len(df):
    print("警告：数据集小于要抽取的行数，将返回整个数据集。")
    sampled_df = df
else:
    # 随机抽取指定行数的数据（不放回）
    #sampled_indices = np.random.choice(df.index, size=num_rows_to_sample, replace=False)
    sampled_indices = range(num_rows_to_sample)
    sampled_df = df.loc[sampled_indices]
# 重置索引（如果需要）
sampled_df.reset_index(drop=True, inplace=True)
# 保存抽取的行到新的CSV文件
#sampled_df.to_csv('E:/Shaun/SSHF-DTI/dataset/Select/train_correct.csv', index=False)
# 读取原有的train.csv文件
train_file_path = 'D:\\Shaun\\SSHF-DTI\\dataset\\all1+2.csv' #origin
train_df = pd.read_csv(train_file_path)
# 将抽取的数据追加到train_df的末尾
combined_df = pd.concat([train_df, sampled_df], ignore_index=True)
# 保存组合后的数据到新的CSV文件
combined_df.to_csv('D:\\Shaun\\SSHF-DTI\\dataset\\all1+2+3.csv', index=False)
'''

# 读取CSV文件按9:1分割
'''
file_path = 'D:\\Shaun\\SSHF-DTA\\dataset\\Kd\\train_origin.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 按照9:1的比例划分数据
train_data = data.sample(frac=0.9, random_state=1)  # 90%的数据作为训练集
val_data = data.drop(train_data.index)             # 剩余10%的数据作为测试集

# 保存划分后的数据到新的CSV文件
train_file_path = 'D:\\Shaun\\SSHF-DTA\\dataset\\Kd\\train.csv'  # 训练集文件路径
val_file_path = 'D:\\Shaun\\SSHF-DTA\\dataset\\Kd\\val.csv'    # 验证集文件路径

train_data.to_csv(train_file_path, index=False)  # 保存训练集，不包含索引
val_data.to_csv(val_file_path, index=False)    # 保存验证集，不包含索引
'''

'''
# 定义输入和输出文件路径
input_file_path = 'D:\\Shaun\\BACPI-master\\data\\affinity\\IC50\\test.txt'
output_file_path = 'D:\\Shaun\\SSHF-DTA\\dataset\\IC50\\test.csv'

# 打开输入文件读取内容，并打开输出文件准备写入
with open(input_file_path, 'r', encoding='utf-8') as infile, \
        open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
    # 创建一个csv写入器，并写入列名
    writer = csv.writer(outfile)
    writer.writerow(['Index', 'SMILES', 'Target Sequence', 'Label'])

    # 读取输入文件的每一行，并处理成csv格式写入输出文件
    index = 0  # 初始化序号
    for line in infile:
        # 去除每行末尾的换行符，并按逗号分割成列表
        parts = line.strip().split(',')

        # 确保有三部分（两个序列和一个指标）
        if len(parts) == 3:
            smiles, target_sequence, label = parts

            # 将序号、SMILES、Target Sequence和Label写入csv文件
            writer.writerow([index, smiles, target_sequence, label])

            # 序号加1
            index += 1
        else:
            print(f"Warning: Line {index + 1} does not have exactly 3 parts and will be skipped.")
'''

#modif词典筛选
'''
# 读取CSV文件
df = pd.read_csv('D:/Shaun/SSHF-DTI/ESPF_A/subword_units_map_uniprot1.csv')
# 根据“index”列中字符串的长度过滤行，只保留长度为1, 3, 4, 5的行
filtered_df = df[df['index'].str.len().isin([1, 2, 3, 4, 5])]

# 处理长度大于5的字符串，拆分为长度为5的子序列
def split_into_chunks(seq, chunk_size):
    return [seq[i:i + chunk_size] for i in range(0, len(seq) - chunk_size + 1, chunk_size)]

# 创建一个新的DataFrame来存储拆分后的数据
extra_rows = []

# 遍历原始数据，处理长度大于3的index
for _, row in df[df['index'].str.len() > 3].iterrows():
    chunks = split_into_chunks(row['index'], 3)
    frequency = row['frequency']  # 假设CSV中有一个名为'frequency'的列
    for chunk in chunks:
        # 只添加长度为3的子序列
        if len(chunk) == 3:
            extra_rows.append({'index': chunk, 'frequency': frequency})
# 遍历原始数据，处理长度大于5的index
for _, row in df[df['index'].str.len() > 4].iterrows():
    chunks = split_into_chunks(row['index'], 4)
    frequency = row['frequency']  # 假设CSV中有一个名为'frequency'的列
    for chunk in chunks:
        # 只添加长度为4的子序列
        if len(chunk) == 4:
            extra_rows.append({'index': chunk, 'frequency': frequency})
# 遍历原始数据，处理长度大于5的index
for _, row in df[df['index'].str.len() > 5].iterrows():
    chunks = split_into_chunks(row['index'], 5)
    frequency = row['frequency']  # 假设CSV中有一个名为'frequency'的列
    for chunk in chunks:
        # 只添加长度为5的子序列
        if len(chunk) == 5:
            extra_rows.append({'index': chunk, 'frequency': frequency})

# 将拆分后的数据转换为DataFrame
extra_df = pd.DataFrame(extra_rows)
# 将原始过滤后的数据和拆分后的数据合并
final_df = pd.concat([filtered_df, extra_df], ignore_index=True)

# 标记重复的index行（不是第一个出现的）
final_df['duplicated'] = final_df.duplicated(subset='index', keep='first')
# 计算每个index对应的frequency总和，并将这个总和广播回原始DataFrame的相应行
final_df['total_frequency'] = final_df.groupby('index')['frequency'].transform('sum')
# 创建一个新的DataFrame来保存处理后的结果
# 只保留不是重复的行，或者保留所有行但是frequency使用总和
result_df = final_df[~final_df['duplicated']].drop(columns=['duplicated', 'frequency'])
result_df = result_df.rename(columns={'total_frequency': 'frequency'})

# 将处理后的结果保存到新的CSV文件中
result_df.to_csv('D:/Shaun/SSHF-DTI/ESPF_A/subword_units_map_uniprot2.csv', index=False)
'''

# 筛选出SMILES列中不包含'.'的行
'''
# 读取CSV文件
df = pd.read_csv('D:\\Shaun\\SSHF-DTA\\dataset\\Kd1\\test_o.csv')
# 筛选出SMILES列中不包含'.'的行
filtered_df = df[~df['SMILES'].str.contains("\\.")]
# 将筛选后的数据保存到新的CSV文件
filtered_df.to_csv('D:\\Shaun\\SSHF-DTA\\dataset\\Kd1\\test.csv', index=False)
'''

# txt转csv
'''# 输入和输出文件名
input_filename = 'E:/Shaun/MCL-DTI-main/data/Human/Human_val.txt'
output_filename = 'E:/Shaun/SSHF-DTI/dataset/Human/val.csv'
# 打开输入文件和输出文件
with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
    # 创建CSV写入器，‌指定列名
    writer = csv.writer(outfile)
    writer.writerow(['index', 'SMILES', 'Target Sequence', 'Label'])
    # 读取每一行，‌分割并写入CSV
    index = 0
    for line in infile:
        smiles, target_seq, label = line.strip().split()
        writer.writerow([index, smiles, target_seq, label])
        index += 1
print(f'转换完成，‌已生成{output_filename}')'''

# csv转txt
'''csv_file_path = 'D:/SSHF-DTI/dataset/BELKA/train1000to1_withoutDy.csv'
txt_file_path = 'D:/SSHF-DTI/dataset/BELKA/train1000to1_withoutDy.txt'
print("Load Done!")
# 打开CSV文件以读取内容
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    # 创建一个CSV读取器
    reader = csv.reader(csvfile)
    # 跳过第一行（列名）
    next(reader)
    # 打开TXT文件以写入内容
    with open(txt_file_path, 'w', encoding='utf-8') as txtfile:
        # 遍历CSV文件的每一行
        for row in reader:
            # 将每行的内容（即每一列）中的[Dy]替换为空或空格
            modified_row = [col.replace('[Dy]', '') for col in row]
            # 将替换后的每行内容（即每一列）用逗号连接，并写入TXT文件
            txtfile.write(','.join(modified_row) + '\n')
print('CSV文件内容已写入TXT文件，并跳过了第一行且替换了[Dy]。')'''
