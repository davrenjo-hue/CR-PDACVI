import os
import pandas as pd
import shutil

# ==================== 配置参数 ====================
# 输入特征文件路径（每个标签/器官一个文件）
feature_files = {
    2: r"2\reduced_features.csv",
    3: r"3\reduced_features.csv",
    4: r"4\reduced_features.csv",
    5: r"5\reduced_features.csv",
    6: r"6\reduced_features.csv"
}

# 输出文件夹（保存对齐后的特征文件）
output_dir = r""
os.makedirs(output_dir, exist_ok=True)

# 文件名所在列（用于匹配样本）
filename_col = "filename"


# ==================== 核心功能 ====================
def load_organ_samples(organ_id, file_path):
    """加载单个器官的样本数据"""
    df = pd.read_csv(file_path)
    if filename_col not in df.columns:
        raise ValueError(f"器官 {organ_id} 的文件中未找到文件名列 '{filename_col}'")
    # 返回数据框和样本文件名集合
    return df, set(df[filename_col].unique())


def find_common_samples(organ_data):
    """找到所有器官共有的样本文件名"""
    # 以第一个器官的样本为基准
    common_samples = next(iter(organ_data.values()))["samples"]

    # 逐步与其他器官的样本取交集
    for organ_id, data in organ_data.items():
        common_samples &= data["samples"]
        print(f"与器官 {organ_id} 取交集后，剩余样本数: {len(common_samples)}")

    return sorted(common_samples)


def align_and_save_samples(organ_data, common_samples, output_dir):
    """对齐所有器官的样本并保存到输出文件夹"""
    for organ_id, data in organ_data.items():
        # 筛选出共同样本
        aligned_df = data["df"][data["df"][filename_col].isin(common_samples)].copy()

        # 按文件名排序，确保所有器官的样本顺序一致
        aligned_df = aligned_df.sort_values(by=filename_col).reset_index(drop=True)

        # 保存对齐后的文件
        output_path = os.path.join(output_dir, f"aligned_features_label{organ_id}.csv")
        aligned_df.to_csv(output_path, index=False)

        print(f"器官 {organ_id}: 对齐后样本数 = {len(aligned_df)}，已保存至 {output_path}")

    return


# ==================== 主流程 ====================
if __name__ == "__main__":
    print("===== 开始对齐所有器官的样本 =====")

    # 1. 加载所有器官的数据和样本名
    organ_data = {}
    for organ_id, file_path in feature_files.items():
        print(f"加载器官 {organ_id} 的数据...")
        df, samples = load_organ_samples(organ_id, file_path)
        organ_data[organ_id] = {
            "df": df,
            "samples": samples,
            "original_count": len(samples)
        }
        print(f"器官 {organ_id} 原始样本数: {len(samples)}")

    # 2. 找到所有器官共有的样本
    print("\n===== 计算所有器官的共同样本 =====")
    common_samples = find_common_samples(organ_data)
    print(f"\n所有器官共有的样本数: {len(common_samples)}")

    # 3. 对齐并保存样本
    print("\n===== 保存对齐后的样本 =====")
    align_and_save_samples(organ_data, common_samples, output_dir)

    print("\n===== 所有器官样本对齐完成 =====")
    print(f"对齐后的样本保存在: {output_dir}")
