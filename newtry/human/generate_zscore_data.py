#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成z-score归一化的ROI RDM数据
从原始的all_subjects_roi_rdms.mat文件生成z-score版本
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform

def zscore_normalize_fmri_data(roi_data):
    """
    对fMRI数据进行z-score归一化
    roi_data: [n_roi_voxels, n_images] - ROI内体素×图像
    返回: 归一化后的数据 [n_roi_voxels, n_images]
    """
    # 对每个体素在1000张图片上进行z-score归一化
    # 即对每一行（每个体素）进行归一化
    normalized_data = np.zeros_like(roi_data)
    
    for voxel_idx in range(roi_data.shape[0]):
        voxel_responses = roi_data[voxel_idx, :]  # 该体素对1000张图片的响应
        
        # 计算该体素的均值和标准差
        mean_response = np.mean(voxel_responses)
        std_response = np.std(voxel_responses)
        
        # 避免除零错误
        if std_response > 0:
            normalized_data[voxel_idx, :] = (voxel_responses - mean_response) / std_response
        else:
            normalized_data[voxel_idx, :] = voxel_responses - mean_response
    
    return normalized_data

def generate_zscore_rdms():
    """生成z-score归一化的RDM数据"""
    print("=" * 60)
    print("生成z-score归一化的ROI RDM数据")
    print("=" * 60)
    
    # 加载原始数据
    print("📁 加载原始ROI数据...")
    roi_file = 'roi_rdm_results/all_subjects_roi_rdms.mat'
    
    if not os.path.exists(roi_file):
        print(f"❌ 原始数据文件不存在: {roi_file}")
        return
    
    roi_data = loadmat(roi_file)
    print(f"✅ 原始数据加载成功")
    
    # 提取被试数据
    subjects = ['s1', 's2', 's5', 's7']
    
    # 存储z-score版本的数据
    zscore_data = {}
    roi_labels = []
    subject_ids = []
    
    print("\n🔄 开始处理z-score归一化...")
    
    for subject in subjects:
        if subject not in roi_data:
            print(f"  ⚠️ 被试 {subject} 数据不存在")
            continue
        
        print(f"\n处理被试: {subject}")
        subject_data = roi_data[subject]
        
        # 存储该被试的z-score RDM
        subject_zscore_rdms = []
        
        # 遍历所有ROI
        for roi_idx in range(subject_data.shape[0]):
            roi_info = subject_data[roi_idx]
            
            # 提取ROI信息
            roi_label = roi_info['roi_label'][0][0]
            roi_name = roi_info['roi_name'][0]
            hemisphere = roi_info['hemisphere'][0]
            n_voxels = roi_info['n_voxels'][0][0]
            n_images = roi_info['n_images'][0][0]
            roi_data_raw = roi_info['roi_data']  # [n_voxels, n_images]
            
            print(f"  ROI {roi_label} ({roi_name}): {n_voxels} 体素, {n_images} 图像")
            
            # 对fMRI数据进行z-score归一化
            roi_data_normalized = zscore_normalize_fmri_data(roi_data_raw)
            
            # 计算z-score归一化后的RDM
            data_for_rdm = roi_data_normalized.T  # [n_images, n_roi_voxels]
            rdm = pdist(data_for_rdm, metric='correlation')
            
            # 存储z-score版本的结果
            roi_result = {
                'roi_label': roi_label,
                'roi_name': roi_name,
                'hemisphere': hemisphere,
                'n_voxels': n_voxels,
                'n_images': n_images,
                'roi_data': roi_data_normalized,  # z-score归一化后的数据
                'rdm': rdm  # z-score归一化后的RDM
            }
            
            subject_zscore_rdms.append(roi_result)
            
            # 收集ROI标签和被试ID
            if roi_idx == 0:  # 只在第一个ROI时添加被试ID
                subject_ids.append(subject)
            
            if subject == 's1':  # 只在第一个被试时收集ROI标签
                roi_labels.append(f"{hemisphere}_{roi_label}")
        
        zscore_data[subject] = np.array(subject_zscore_rdms, dtype=object)
        print(f"  ✅ 被试 {subject} 处理完成")
    
    # 重新组织数据为矩阵格式
    print("\n🔄 重新组织数据为矩阵格式...")
    
    # 收集所有ROI的RDM
    all_rdms = []
    for subject in subjects:
        if subject in zscore_data:
            subject_rdms = []
            for roi_idx in range(len(zscore_data[subject])):
                roi_rdm = zscore_data[subject][roi_idx]['rdm']
                subject_rdms.append(roi_rdm)
            all_rdms.append(subject_rdms)
    
    # 转换为numpy数组
    all_rdms_array = np.array(all_rdms)  # [n_subjects, n_rois, rdm_length]
    roi_labels_array = np.array(roi_labels)
    subject_ids_array = np.array(subject_ids)
    
    print(f"  数据形状: {all_rdms_array.shape}")
    print(f"  ROI标签: {roi_labels_array}")
    print(f"  被试ID: {subject_ids_array}")
    
    # 保存z-score版本的数据
    print("\n💾 保存z-score版本数据...")
    
    # 更新原始MAT文件，添加z-score版本
    roi_data['roi_rdms_zscore'] = all_rdms_array
    roi_data['roi_labels_zscore'] = roi_labels_array
    roi_data['subject_ids_zscore'] = subject_ids_array
    
    # 保存更新后的MAT文件
    savemat(roi_file, roi_data)
    print(f"✅ z-score数据已添加到: {roi_file}")
    
    # 保存单独的z-score版本文件
    zscore_file = 'roi_rdm_results/all_subjects_roi_rdms_zscore.mat'
    savemat(zscore_file, {
        'roi_rdms_zscore': all_rdms_array,
        'roi_labels_zscore': roi_labels_array,
        'subject_ids_zscore': subject_ids_array,
        'description': 'z-score normalized ROI RDM data',
        'normalization_method': 'z-score per voxel across 1000 images'
    })
    print(f"✅ z-score数据已保存: {zscore_file}")
    
    # 生成统计报告
    print("\n📊 生成统计报告...")
    
    with open('roi_rdm_results/zscore_generation_report.txt', 'w', encoding='utf-8') as f:
        f.write("Z-score归一化ROI RDM数据生成报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("归一化方法:\n")
        f.write("- 对每个体素在1000张图片上进行z-score归一化\n")
        f.write("- 公式: (response - mean) / std\n")
        f.write("- 如果std=0，则只进行中心化: response - mean\n\n")
        
        f.write("数据统计:\n")
        f.write(f"- 被试数量: {len(subject_ids_array)}\n")
        f.write(f"- ROI数量: {len(roi_labels_array)}\n")
        f.write(f"- RDM长度: {all_rdms_array.shape[2]}\n")
        f.write(f"- 被试: {', '.join(subject_ids_array)}\n")
        f.write(f"- ROI: {', '.join(roi_labels_array)}\n\n")
        
        f.write("文件说明:\n")
        f.write(f"- 原始文件: {roi_file} (包含原始和z-score版本)\n")
        f.write(f"- z-score文件: {zscore_file} (仅包含z-score版本)\n")
    
    print("✅ 统计报告已保存: roi_rdm_results/zscore_generation_report.txt")
    
    print("\n" + "=" * 60)
    print("Z-score归一化数据生成完成!")
    print("=" * 60)
    print(f"现在可以使用以下embedding类型进行RSA分析:")
    print(f"  python rsa_with_embedding_choice.py --embedding_type image --use_zscore")
    print(f"  python rsa_with_embedding_choice.py --embedding_type word_average --use_zscore")
    print(f"  python rsa_with_embedding_choice.py --embedding_type noun --use_zscore")
    print(f"  python rsa_with_embedding_choice.py --embedding_type verb --use_zscore")

if __name__ == "__main__":
    generate_zscore_rdms()
