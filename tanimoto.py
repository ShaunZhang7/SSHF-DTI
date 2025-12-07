import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing
import warnings

warnings.filterwarnings('ignore')  # 常规警告
# 在 import rdkit 之后立刻添加：
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # 禁用所有RDKit日志


def smiles_to_fingerprint_wrapper(args):
    """包装函数，用于并行处理"""
    smiles, radius, n_bits = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def batch_find_similar(target_smiles_list, source_smiles_list, source_df, threshold=1.0, n_workers=None):
    """批量查找相似SMILES"""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    # 为所有源SMILES预计算指纹
    print("预计算源数据指纹...")
    source_args = [(smiles, 2, 2048) for smiles in source_smiles_list]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        source_fps = list(tqdm(executor.map(smiles_to_fingerprint_wrapper, source_args),
                               total=len(source_args), desc="计算指纹"))

    # 构建有效索引映射
    valid_source_indices = [i for i, fp in enumerate(source_fps) if fp is not None]
    valid_source_fps = [source_fps[i] for i in valid_source_indices]

    # 批量处理目标SMILES
    print("批量查找相似序列...")
    all_similar_indices = []

    for target_smiles in tqdm(target_smiles_list, desc="处理目标SMILES"):
        target_fp = smiles_to_fingerprint_wrapper((target_smiles, 2, 2048))
        if target_fp is None:
            continue

        # 计算与所有有效源指纹的相似度
        similarities = []
        for source_fp in valid_source_fps:
            sim = DataStructs.TanimotoSimilarity(target_fp, source_fp)
            similarities.append(sim)

        # 找出相似度达到阈值的
        similar_indices = []
        for i, sim in enumerate(similarities):
            if sim >= threshold - 1e-6:
                original_idx = valid_source_indices[i]
                similar_indices.append(original_idx)

        all_similar_indices.extend(similar_indices)

    # 去重并获取对应行
    unique_indices = list(set(all_similar_indices))
    if unique_indices:
        return source_df.iloc[unique_indices].copy()
    else:
        return pd.DataFrame()


def main_optimized():
    # 1. 定义文件路径
    csv1_path = r"D:\Shaun\SSHF-DTI\dataset\BindingDB\test.csv"  # 基准文件
    csv2_path = r"D:\Shaun\SSHF-DTI\dataset\BindingDB\train.csv"  # 需要添加数据的文件
    csv3_path = r"D:\Shaun\SSHF-DTI\dataset\BindingDB\val.csv"  # 来源文件1
    csv4_path = r"D:\Shaun\SSHF-DTI\dataset\DAVIS\all.csv"  # 来源文件2
    output_path = r"D:\Shaun\SSHF-DTI\dataset\Test\BindingDB_Add.csv"  # 最终输出文件

    # 2. 读取CSV文件
    print("正在读取CSV文件...")
    try:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        df3 = pd.read_csv(csv3_path)
        df4 = pd.read_csv(csv4_path)
    except Exception as e:
        print(f"文件读取失败: {e}")
        return

    # 3. 检查列
    required_cols = ['SMILES', 'Target Sequence']
    dataframes = [df1, df2, df3, df4]
    names = ['test.csv', 'train.csv', 'val.csv', 'BindingDB/all.csv', 'DAVIS/all.csv']

    for df, name in zip(dataframes, names):
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"{name} 缺少列: {missing_cols}")
            return

    print(f"test.csv (基准): {len(df1)} 行")
    print(f"train.csv (待扩充): {len(df2)} 行")
    print(f"val.csv (来源1): {len(df3)} 行")
    print(f"BindingDB/all.csv (来源2): {len(df4)} 行")


    # 4. 获取基准SMILES列表
    target_smiles_list = df1['SMILES'].tolist()
    print(f"\n基准SMILES数量: {len(target_smiles_list)}")

    # 5. 并行处理三个来源文件
    all_matched_dfs = []  # 存储所有匹配到的行

    # 处理 val.csv
    print("\n=== 在 val.csv 中查找匹配项 ===")
    matched_df3 = batch_find_similar(
        target_smiles_list,
        df3['SMILES'].tolist(),
        df3,
        threshold=1.0
    )
    if not matched_df3.empty:
        print(f"  找到 {len(matched_df3)} 条匹配记录")
        all_matched_dfs.append(matched_df3)
    else:
        print("  未找到匹配记录")

    # 处理 BindingDB/all.csv
    print("\n=== 在 BindingDB/all.csv 中查找匹配项 ===")
    matched_df4 = batch_find_similar(
        target_smiles_list,
        df4['SMILES'].tolist(),
        df4,
        threshold=1.0
    )
    if not matched_df4.empty:
        print(f"  找到 {len(matched_df4)} 条匹配记录")
        all_matched_dfs.append(matched_df4)
    else:
        print("  未找到匹配记录")


    # 6. 合并结果到 train.csv
    if all_matched_dfs:
        # 合并所有匹配到的行
        all_matched = pd.concat(all_matched_dfs, ignore_index=True)
        print(f"\n总计匹配到 {len(all_matched)} 条记录")

        # 合并到原始 train.csv
        result_df = pd.concat([df2, all_matched], ignore_index=True)
        initial_count = len(result_df)


        # 去重（确保 SMILES 和 Target Sequence 都相同时才去重）
        #result_df = result_df.drop_duplicates(subset=['SMILES', 'Target Sequence'], keep='first')
        #final_count = len(result_df)

        print(f"\n=== 最终结果统计 ===")
        print(f"原始 train.csv 行数: {len(df2)}")
        print(f"新增匹配记录数: {len(all_matched)}")
        print(f"合并后总行数 (去重前): {initial_count}")
        #print(f"去重后总行数: {final_count}")
        #print(f"实际净增加行数: {final_count - len(df2)}")
        #print(f"重复记录数: {initial_count - final_count}")
    else:
        print("\n未在任何来源文件中找到匹配记录，train.csv 保持不变")
        result_df = df2.copy()

    # 7. 保存结果
    result_df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main_optimized()