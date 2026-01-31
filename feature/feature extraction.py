import os
import yaml
import logging
import pandas as pd
from tqdm import tqdm
from radiomics import featureextractor
import SimpleITK as sitk
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ================== 路径配置 ==================
normal_image_dir = r""
normal_mask_dir = r""
disease_no_transfer_image_dir = r""
disease_no_transfer_mask_dir = r""
disease_transfer_image_dir = r""
disease_transfer_mask_dir = r""

param_file = r"features.yaml"
output_dir = r""
os.makedirs(output_dir, exist_ok=True)

# ================== 读取配置（全局共享） ==================
extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
logging.getLogger("radiomics").setLevel(logging.ERROR)
print(extractor.settings)
print(extractor.enabledImagetypes)

# ================== 辅助函数 ==================
label_results = {2: [], 3: [], 4: [], 5: [], 6: []}
label_counter = Counter()
lock = threading.Lock()


def extract_features_task(args):
    """单个病例 + 单个标签的特征提取任务（线程安全）"""
    img_path, mask_path, label, disease_label = args
    patient_id = os.path.basename(img_path).split(".")[0]

    try:
        result = extractor.execute(img_path, mask_path, label=label)
        features = {k: v for k, v in result.items() if not k.startswith("diagnostics_")}
        # features = {k: v for k, v in result.items() if k.startswith("original")}
        if not features:
            return None, patient_id, label, f"⚠️ {patient_id} Label {label}: 无特征提取结果"

        features["filename"] = patient_id
        features["group"] = label
        features["label"] = disease_label

        num_features = len(features) - 3
        msg = f"✅ {patient_id} Label {label}: 特征提取成功 ({num_features} 个特征)"

        with lock:
            label_counter[label] += 1
            label_results[label].append(features)

        return features, patient_id, label, msg

    except Exception as e:
        msg = f"❌ {patient_id} Label {label}: 提取失败 ({e})"
        return None, patient_id, label, msg


def get_image_mask_pairs(image_folder, mask_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".nii", ".nii.gz"))]
    pairs = []
    for img in image_files:
        mask_fname = img.replace("_0000.nii.gz", ".nii.gz") if "_0000.nii.gz" in img else img
        mask_path = os.path.join(mask_folder, mask_fname)
        if os.path.exists(mask_path):
            pairs.append((os.path.join(image_folder, img), mask_path))
        else:
            print(f"⚠️ 未找到 mask 文件: {mask_fname}")
    return pairs


def process_group_parallel(image_dir, mask_dir, disease_label, group_name, max_workers=None):
    """并行处理一个组（多标签）"""
    pairs = get_image_mask_pairs(image_dir, mask_dir)
    labels = [2, 3, 4, 5, 6]
    tasks = [(img_path, mask_path, lbl, disease_label) for img_path, mask_path in pairs for lbl in labels]

    if not tasks:
        print(f"⚠️ {group_name} 无有效配对数据")
        return

    if max_workers is None:
        max_workers = min(32, os.cpu_count() + 4)  # 推荐：CPU核心数 + 4

    print(f"🚀 {group_name} 开始并行提取，使用 {max_workers} 线程，共 {len(tasks)} 任务")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(extract_features_task, task): task for task in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=group_name):
            _, patient_id, label, msg = future.result()
            print(msg)

    print(f"🟢 {group_name} 完成，成功提取: {dict(label_counter)}\n")


# ================== 主程序（多线程） ==================
if __name__ == "__main__":
    MAX_WORKERS = min(2, os.cpu_count() or 1)  # 推荐 8 线程，防内存溢出

    # process_group_parallel(normal_image_dir, normal_mask_dir, 0, "无病组", MAX_WORKERS)
    process_group_parallel(disease_transfer_image_dir, disease_transfer_mask_dir, 1, "有病-转移", MAX_WORKERS)
    process_group_parallel(disease_no_transfer_image_dir, disease_no_transfer_mask_dir, 0, "有病-未转移", MAX_WORKERS)

    # ================== 保存结果 ==================
    for lbl, feats in label_results.items():
        if not feats:
            print(f"⚠️ Label {lbl} 无数据，跳过保存")
            continue
        df = pd.DataFrame(feats)
        out_path = os.path.join(output_dir, f"Radiomics_label{lbl}.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Label {lbl} 保存: {len(df)} 行 × {len(df.columns) - 3} 特征 → {out_path}")
        print(f"包含 distance-2:", any("distance-2" in col for col in df.columns))
        print(f"包含 wavelet:", any("wavelet" in col for col in df.columns))

    print("\n🎉 全部提取完成！所有特征已并行保存。")