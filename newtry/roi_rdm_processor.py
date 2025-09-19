#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI RDM处理器
根据streams.mgz文件提取ROI，从fMRI数据计算RDM
"""

import os
import numpy as np
import nibabel as nib
from scipy.spatial.distance import pdist
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ROIRDMProcessor:
    """ROI RDM处理器"""
    
    def __init__(self, data_dir="E:/lunzhuan1/rotation2025/Human"):
        """初始化"""
        self.data_dir = data_dir
        self.subjects = ['s1', 's2', 's5', 's7']
        self.hemispheres = ['lh', 'rh']
        
        # ROI标签定义
        self.roi_labels = {
            0: "Unknown",
            1: "early (早期视觉)",
            2: "midventral (中腹侧)", 
            3: "midlateral (中外侧)",
            4: "midparietal (中顶叶)",
            5: "ventral (腹侧)",
            6: "lateral (外侧)",
            7: "parietal (顶叶)"
        }
        
        # 只处理标签1-7，跳过0
        self.target_rois = [1, 2, 3, 4, 5, 6, 7]
    
    def load_roi_masks(self, subject):
        """加载ROI掩码"""
        print(f"加载 {subject} 的ROI掩码...")
        
        roi_masks = {}
        
        for hemisphere in self.hemispheres:
            # 构建文件路径
            streams_file = os.path.join(self.data_dir, subject, 'fs', 'label', f'{hemisphere}.streams.mgz')
            
            if not os.path.exists(streams_file):
                print(f"❌ 文件不存在: {streams_file}")
                continue
            
            try:
                # 读取.mgz文件
                img = nib.load(streams_file)
                roi_data = img.get_fdata().squeeze()
                
                print(f"  {hemisphere}: 形状 {roi_data.shape}")
                
                # 分析ROI标签
                unique_labels = np.unique(roi_data)
                print(f"  {hemisphere}: ROI标签 {unique_labels}")
                
                # 为每个目标ROI创建掩码
                for roi_label in self.target_rois:
                    if roi_label in unique_labels:
                        mask = roi_data == roi_label
                        roi_masks[f'{hemisphere}_{roi_label}'] = mask
                        
                        n_voxels = np.sum(mask)
                        print(f"    ROI {roi_label} ({self.roi_labels[roi_label]}): {n_voxels} 个顶点")
                    else:
                        print(f"    ⚠️  ROI {roi_label} 在 {hemisphere} 中不存在")
                
            except Exception as e:
                print(f"❌ 读取 {streams_file} 失败: {e}")
        
        return roi_masks
    
    def load_fmri_data(self, subject):
        """加载fMRI数据"""
        print(f"加载 {subject} 的fMRI数据...")
        
        fmri_data = {}
        
        for hemisphere in self.hemispheres:
            # 构建文件路径
            fmri_file = os.path.join(self.data_dir, f'{subject.upper()}_{hemisphere}_Rsp.mat')
            
            if not os.path.exists(fmri_file):
                print(f"❌ 文件不存在: {fmri_file}")
                continue
            
            try:
                # 读取.mat文件
                mat_data = loadmat(fmri_file)
                
                # 查找包含fMRI数据的变量
                data_key = None
                for key in mat_data.keys():
                    if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                        if mat_data[key].ndim == 2:  # 2D数组，可能是fMRI数据
                            data_key = key
                            break
                
                if data_key is None:
                    print(f"❌ 在 {fmri_file} 中未找到fMRI数据")
                    continue
                
                data = mat_data[data_key]
                print(f"  {hemisphere}: 数据形状 {data.shape}, 变量名: {data_key}")
                
                # 确保数据格式正确 [n_voxels, n_images]
                if data.shape[0] < data.shape[1]:
                    data = data.T
                    print(f"  {hemisphere}: 转置后形状 {data.shape}")
                
                fmri_data[hemisphere] = data
                
            except Exception as e:
                print(f"❌ 读取 {fmri_file} 失败: {e}")
        
        return fmri_data
    
    def extract_roi_data(self, fmri_data, roi_mask):
        """从fMRI数据中提取ROI数据"""
        # fmri_data: [n_voxels, n_images] - 体素×图像
        # roi_mask: [n_voxels] 布尔数组 - 体素掩码
        # 返回: [n_roi_voxels, n_images] - ROI内体素×图像
        
        roi_data = fmri_data[roi_mask, :]
        return roi_data
    
    def compute_rdm(self, roi_data, metric='correlation'):
        """计算RDM"""
        # roi_data: [n_roi_voxels, n_images]
        # 需要转置为: [n_images, n_roi_voxels] 才能对图像计算RDM
        data_for_rdm = roi_data.T
        
        # 计算RDM - 对图像计算成对距离
        rdm = pdist(data_for_rdm, metric=metric)
        
        return rdm
    
    def process_subject(self, subject):
        """处理单个被试"""
        print(f"\n{'='*60}")
        print(f"处理被试: {subject}")
        print(f"{'='*60}")
        
        # 1. 加载ROI掩码
        roi_masks = self.load_roi_masks(subject)
        
        # 2. 加载fMRI数据
        fmri_data = self.load_fmri_data(subject)
        
        if not roi_masks or not fmri_data:
            print(f"❌ {subject} 数据加载失败")
            return None
        
        # 3. 计算每个ROI的RDM
        subject_results = {}
        
        for roi_key, roi_mask in roi_masks.items():
            hemisphere, roi_label = roi_key.split('_')
            roi_label = int(roi_label)
            
            print(f"\n处理 {hemisphere} ROI {roi_label} ({self.roi_labels[roi_label]})...")
            
            # 检查fMRI数据是否存在
            if hemisphere not in fmri_data:
                print(f"❌ {hemisphere} fMRI数据不存在")
                continue
            
            # 检查数据维度匹配
            if roi_mask.shape[0] != fmri_data[hemisphere].shape[0]:
                print(f"❌ 维度不匹配: ROI掩码 {roi_mask.shape[0]} vs fMRI数据 {fmri_data[hemisphere].shape[0]}")
                continue
            
            try:
                # 提取ROI数据
                roi_data = self.extract_roi_data(fmri_data[hemisphere], roi_mask)
                print(f"  ROI数据形状: {roi_data.shape}")
                
                # 计算RDM
                rdm = self.compute_rdm(roi_data)
                print(f"  RDM长度: {len(rdm)}")
                
                # 存储结果
                subject_results[roi_key] = {
                    'roi_label': roi_label,
                    'roi_name': self.roi_labels[roi_label],
                    'hemisphere': hemisphere,
                    'n_voxels': roi_data.shape[0],
                    'n_images': roi_data.shape[1],
                    'roi_data': roi_data,
                    'rdm': rdm
                }
                
                print(f"  ✅ 完成")
                
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
        
        return subject_results
    
    def process_all_subjects(self):
        """处理所有被试"""
        print("ROI RDM处理器")
        print("="*100)
        
        all_results = {}
        
        for subject in self.subjects:
            subject_results = self.process_subject(subject)
            if subject_results:
                all_results[subject] = subject_results
        
        return all_results
    
    def save_results(self, all_results, save_dir="roi_rdm_results"):
        """保存结果"""
        print(f"\n保存结果到 {save_dir}...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存所有结果
        savemat(os.path.join(save_dir, 'all_subjects_roi_rdms.mat'), all_results)
        print(f"✅ 所有结果已保存: {save_dir}/all_subjects_roi_rdms.mat")
        
        # 保存汇总信息
        summary = {}
        for subject, subject_results in all_results.items():
            summary[subject] = {}
            for roi_key, roi_result in subject_results.items():
                summary[subject][roi_key] = {
                    'roi_label': roi_result['roi_label'],
                    'roi_name': roi_result['roi_name'],
                    'hemisphere': roi_result['hemisphere'],
                    'n_voxels': roi_result['n_voxels'],
                    'n_images': roi_result['n_images'],
                    'rdm_length': len(roi_result['rdm'])
                }
        
        savemat(os.path.join(save_dir, 'roi_rdm_summary.mat'), summary)
        print(f"✅ 汇总信息已保存: {save_dir}/roi_rdm_summary.mat")
        
        # 保存文本摘要
        with open(os.path.join(save_dir, 'roi_rdm_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("ROI RDM计算结果摘要\n")
            f.write("="*50 + "\n\n")
            
            for subject, subject_results in all_results.items():
                f.write(f"被试 {subject}:\n")
                for roi_key, roi_result in subject_results.items():
                    f.write(f"  {roi_key}: {roi_result['roi_name']}\n")
                    f.write(f"    半球: {roi_result['hemisphere']}\n")
                    f.write(f"    顶点数: {roi_result['n_voxels']}\n")
                    f.write(f"    图像数: {roi_result['n_images']}\n")
                    f.write(f"    RDM长度: {len(roi_result['rdm'])}\n\n")
        
        print(f"✅ 文本摘要已保存: {save_dir}/roi_rdm_summary.txt")
    
    def visualize_results(self, all_results, save_dir="roi_rdm_results"):
        """可视化结果"""
        print(f"\n生成可视化图表...")
        
        # 统计信息
        roi_stats = {}
        for subject, subject_results in all_results.items():
            for roi_key, roi_result in subject_results.items():
                if roi_key not in roi_stats:
                    roi_stats[roi_key] = []
                roi_stats[roi_key].append(roi_result['n_voxels'])
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ROI大小分布
        ax1 = axes[0, 0]
        roi_names = []
        roi_sizes = []
        for roi_key, sizes in roi_stats.items():
            roi_names.append(roi_key)
            roi_sizes.append(np.mean(sizes))
        
        bars = ax1.bar(range(len(roi_names)), roi_sizes, alpha=0.7, color='skyblue')
        ax1.set_xlabel('ROI')
        ax1.set_ylabel('平均顶点数')
        ax1.set_title('ROI大小分布')
        ax1.set_xticks(range(len(roi_names)))
        ax1.set_xticklabels(roi_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, size in zip(bars, roi_sizes):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                    f'{int(size)}', ha='center', va='bottom', fontsize=10)
        
        # 2. 被试间比较
        ax2 = axes[0, 1]
        subjects = list(all_results.keys())
        roi_keys = list(roi_stats.keys())
        
        x = np.arange(len(roi_keys))
        width = 0.2
        
        for i, subject in enumerate(subjects):
            sizes = []
            for roi_key in roi_keys:
                if roi_key in all_results[subject]:
                    sizes.append(all_results[subject][roi_key]['n_voxels'])
                else:
                    sizes.append(0)
            
            ax2.bar(x + i*width, sizes, width, label=subject, alpha=0.7)
        
        ax2.set_xlabel('ROI')
        ax2.set_ylabel('顶点数')
        ax2.set_title('被试间ROI大小比较')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(roi_keys, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 左右半球比较
        ax3 = axes[1, 0]
        lh_sizes = []
        rh_sizes = []
        roi_labels = []
        
        for roi_key, sizes in roi_stats.items():
            if roi_key.startswith('lh_'):
                roi_label = roi_key.split('_')[1]
                roi_labels.append(roi_label)
                lh_sizes.append(np.mean(sizes))
            elif roi_key.startswith('rh_'):
                roi_label = roi_key.split('_')[1]
                if roi_label in roi_labels:
                    idx = roi_labels.index(roi_label)
                    rh_sizes.insert(idx, np.mean(sizes))
                else:
                    roi_labels.append(roi_label)
                    rh_sizes.append(np.mean(sizes))
        
        x = np.arange(len(roi_labels))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, lh_sizes, width, label='左半球', alpha=0.7, color='lightblue')
        bars2 = ax3.bar(x + width/2, rh_sizes, width, label='右半球', alpha=0.7, color='lightcoral')
        
        ax3.set_xlabel('ROI标签')
        ax3.set_ylabel('平均顶点数')
        ax3.set_title('左右半球ROI大小比较')
        ax3.set_xticks(x)
        ax3.set_xticklabels(roi_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 数据完整性
        ax4 = axes[1, 1]
        completeness = []
        labels = []
        
        for roi_key in roi_stats.keys():
            total_possible = len(self.subjects)
            actual_count = len(roi_stats[roi_key])
            completeness.append(actual_count / total_possible * 100)
            labels.append(roi_key)
        
        bars = ax4.bar(range(len(labels)), completeness, alpha=0.7, color='lightgreen')
        ax4.set_xlabel('ROI')
        ax4.set_ylabel('数据完整性 (%)')
        ax4.set_title('ROI数据完整性')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, comp in zip(bars, completeness):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{comp:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roi_rdm_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 可视化图表已保存: {save_dir}/roi_rdm_analysis.png")


def main():
    """主函数"""
    # 创建处理器
    processor = ROIRDMProcessor()
    
    # 处理所有被试
    all_results = processor.process_all_subjects()
    
    if all_results:
        # 保存结果
        processor.save_results(all_results)
        
        # 可视化结果
        processor.visualize_results(all_results)
        
        print("\n" + "="*100)
        print("🎉 ROI RDM计算完成!")
        print("="*100)
        
        print(f"\n📊 处理结果:")
        total_rois = sum(len(subject_results) for subject_results in all_results.values())
        print(f"  - 处理了 {len(all_results)} 个被试")
        print(f"  - 计算了 {total_rois} 个ROI RDM")
        
        print(f"\n📁 生成的文件:")
        print(f"  - roi_rdm_results/all_subjects_roi_rdms.mat (所有RDM数据)")
        print(f"  - roi_rdm_results/roi_rdm_summary.mat (汇总信息)")
        print(f"  - roi_rdm_results/roi_rdm_summary.txt (文本摘要)")
        print(f"  - roi_rdm_results/roi_rdm_analysis.png (可视化图表)")
        
    else:
        print("❌ 没有成功处理任何数据")


if __name__ == "__main__":
    main()
