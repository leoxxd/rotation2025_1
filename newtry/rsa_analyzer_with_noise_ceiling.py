#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带噪声天花板校正的RSA分析器
"""

import os
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def corr_rdms(X, Y):
    """原始项目的RDM相关性计算函数"""
    X = X - X.mean(axis=1, keepdims=True)
    X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y /= np.sqrt(np.einsum("ij,ij->i", Y, Y))[:, None]
    return np.einsum("ik,jk", X, Y)


def give_vector_pos(list_index, m):
    """计算上三角RDM中特定索引对的向量位置"""
    from itertools import permutations
    
    perm = permutations(list_index, 2)
    upper_list_index = []
    for i in perm:
        if i[0] < i[1]:
            upper_list_index.append(i)
    
    vector_pos = []
    for j in upper_list_index:
        if j[0] == 0:
            vector = j[1] - j[0] - 1
        else:
            vector = (int((m*j[0]) - (j[0]*(j[0]+1)/2) + j[1]-j[0]) - 1)
        vector_pos.append(vector)
    
    vector_pos.sort()
    return vector_pos


class RSAAnalyzerWithNoiseCeiling:
    """带噪声天花板校正的RSA分析器"""
    
    def __init__(self, roi_rdm_file="roi_rdm_results/all_subjects_roi_rdms.mat", 
                 embedding_file="captions/embeddings_output/image_embeddings.npy"):
        """初始化"""
        self.roi_rdm_file = roi_rdm_file
        self.embedding_file = embedding_file
        
        # ROI标签定义
        self.roi_labels = {
            1: "early (早期视觉)",
            2: "midventral (中腹侧)", 
            3: "midlateral (中外侧)",
            4: "midparietal (中顶叶)",
            5: "ventral (腹侧)",
            6: "lateral (外侧)",
            7: "parietal (顶叶)"
        }
    
    def load_roi_rdms(self):
        """加载ROI RDM数据"""
        print("加载ROI RDM数据...")
        
        if not os.path.exists(self.roi_rdm_file):
            raise FileNotFoundError(f"ROI RDM文件不存在: {self.roi_rdm_file}")
        
        roi_data = loadmat(self.roi_rdm_file)
        print(f"✅ ROI RDM数据加载成功")
        
        return roi_data
    
    def load_embeddings(self):
        """加载embedding数据"""
        print("加载embedding数据...")
        
        if not os.path.exists(self.embedding_file):
            raise FileNotFoundError(f"embedding文件不存在: {self.embedding_file}")
        
        embeddings = np.load(self.embedding_file)
        print(f"✅ embedding数据加载成功，形状: {embeddings.shape}")
        
        return embeddings
    
    def compute_embedding_rdm(self, embeddings, metric='correlation'):
        """计算embedding的RDM"""
        print("计算embedding RDM...")
        
        embedding_rdm = pdist(embeddings, metric=metric)
        print(f"✅ embedding RDM计算完成，长度: {len(embedding_rdm)}")
        return embedding_rdm
    
    def subsample_rdm_with_indices(self, rdm, selected_indices):
        """使用预定义的索引子采样RDM"""
        n_total_images = int((1 + np.sqrt(1 + 8 * len(rdm))) / 2)
        rdm_indices = give_vector_pos(selected_indices, n_total_images)
        subsampled_rdm = rdm[rdm_indices]
        return subsampled_rdm
    
    def compute_noise_ceilings(self, roi_data):
        """计算噪声天花板 - 基于fMRI数据，不涉及embedding"""
        print("\n计算噪声天花板（基于fMRI数据）...")
        
        # 收集所有被试的ROI RDM
        all_subject_rdms = {}
        
        for subject in roi_data.keys():
            if subject.startswith('__'):
                continue
            
            print(f"  处理被试: {subject}")
            subject_data = roi_data[subject]
            
            if isinstance(subject_data, dict):
                roi_dict = subject_data
            elif isinstance(subject_data, np.ndarray) and subject_data.dtype.names:
                roi_dict = {}
                for field_name in subject_data.dtype.names:
                    roi_dict[field_name] = subject_data[field_name][0, 0]
            else:
                continue
            
            all_subject_rdms[subject] = {}
            
            for roi_key in roi_dict.keys():
                if roi_key.startswith('__'):
                    continue
                
                try:
                    roi_info = roi_dict[roi_key]
                    
                    if isinstance(roi_info, np.ndarray) and roi_info.dtype.names:
                        roi_rdm = roi_info['rdm'][0, 0].flatten()
                    elif isinstance(roi_info, dict):
                        roi_rdm = roi_info['rdm'].flatten()
                    else:
                        roi_rdm = roi_info.flatten()
                    
                    all_subject_rdms[subject][roi_key] = roi_rdm
                    
                except Exception as e:
                    print(f"    ❌ {roi_key} 处理失败: {e}")
        
        # 计算每个被试每个ROI的噪声天花板
        # 噪声天花板 = 该被试的fMRI RDM与其他被试平均fMRI RDM的相关性
        noise_ceilings = {}
        subjects = list(all_subject_rdms.keys())
        
        print(f"  被试列表: {subjects}")
        
        for subject in subjects:
            noise_ceilings[subject] = {}
            
            for roi_key in all_subject_rdms[subject].keys():
                # 该被试的fMRI RDM
                subject_rdm = all_subject_rdms[subject][roi_key]
                
                # 其他被试的fMRI RDM
                other_subjects = [s for s in subjects if s != subject]
                other_rdms = [all_subject_rdms[s][roi_key] for s in other_subjects]
                
                # 计算其他被试的平均fMRI RDM
                mean_other_rdm = np.mean(other_rdms, axis=0)
                
                # 计算噪声天花板：该被试fMRI RDM与其他被试平均fMRI RDM的相关性
                # 这代表了该ROI在该被试中的最大可能相关性
                noise_ceiling = corr_rdms(subject_rdm.reshape(1, -1), mean_other_rdm.reshape(1, -1))[0, 0]
                noise_ceilings[subject][roi_key] = noise_ceiling
                
                print(f"    {subject} {roi_key}: 噪声天花板 = {noise_ceiling:.3f}")
                print(f"      该被试fMRI RDM与其他{len(other_subjects)}个被试平均fMRI RDM的相关性")
        
        return noise_ceilings
    
    def analyze_subject_roi_with_noise_ceiling(self, subject, roi_key, roi_rdm, embedding_rdm, 
                                             noise_ceiling, n_samples=100, n_images=100, seed=42):
        """带噪声天花板校正的RSA分析"""
        
        # 1. 生成一致的采样索引
        n_total_images = int((1 + np.sqrt(1 + 8 * len(roi_rdm))) / 2)
        np.random.seed(seed)
        
        # 2. 生成采样索引
        sample_indices_list = []
        for i in range(n_samples):
            selected_indices = np.random.choice(n_total_images, size=n_images, replace=False)
            selected_indices = np.sort(selected_indices)
            sample_indices_list.append(selected_indices)
        
        # 3. 使用相同的采样索引对ROI和embedding RDM进行子采样
        correlations = []
        for i, selected_indices in enumerate(sample_indices_list):
            roi_subsample = self.subsample_rdm_with_indices(roi_rdm, selected_indices)
            embedding_subsample = self.subsample_rdm_with_indices(embedding_rdm, selected_indices)
            
            corr = corr_rdms(roi_subsample.reshape(1, -1), embedding_subsample.reshape(1, -1))[0, 0]
            correlations.append(corr)
        
        # 4. 计算平均相关性
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
        
        # 5. 应用噪声天花板校正
        if noise_ceiling > 0:
            corrected_correlation = mean_correlation / noise_ceiling
            
            # 调试信息：如果原始相关性大于噪声天花板，说明模型表现很好
            if mean_correlation > noise_ceiling:
                print(f"  ✅ 优秀表现: {subject} {roi_key} 原始相关性({mean_correlation:.3f}) > 噪声天花板({noise_ceiling:.3f})")
                print(f"     这表明embedding模型在该ROI表现超过fMRI噪声天花板")
                print(f"     校正后相关性: {corrected_correlation:.3f}")
        else:
            corrected_correlation = mean_correlation
        
        # 6. 计算传统方法的相关性作为对比
        traditional_corr, traditional_p = spearmanr(roi_rdm, embedding_rdm)
        
        # 提取ROI信息
        hemisphere, roi_label = roi_key.split('_')
        roi_label = int(roi_label)
        
        result = {
            'subject': subject,
            'roi_key': roi_key,
            'hemisphere': hemisphere,
            'roi_label': roi_label,
            'roi_name': self.roi_labels[roi_label],
            'raw_correlation': mean_correlation,
            'noise_ceiling': noise_ceiling,
            'corrected_correlation': corrected_correlation,
            'traditional_corr': traditional_corr,
            'traditional_p': traditional_p,
            'roi_rdm_length': len(roi_rdm),
            'embedding_rdm_length': len(embedding_rdm),
            'n_subsamples': len(correlations),
            'n_images_per_sample': n_images,
            'all_correlations': correlations
        }
        
        return result
    
    def run_rsa_analysis_with_noise_ceiling(self):
        """运行带噪声天花板校正的RSA分析"""
        print("开始带噪声天花板校正的RSA分析...")
        print("="*60)
        
        # 1. 加载数据
        roi_data = self.load_roi_rdms()
        embeddings = self.load_embeddings()
        
        # 2. 计算embedding RDM
        embedding_rdm = self.compute_embedding_rdm(embeddings)
        
        # 3. 计算噪声天花板
        noise_ceilings = self.compute_noise_ceilings(roi_data)
        
        # 4. 分析每个被试每个ROI
        rsa_results = []
        
        for subject in roi_data.keys():
            if subject.startswith('__'):
                continue
            
            print(f"\n分析被试: {subject}")
            subject_data = roi_data[subject]
            
            if isinstance(subject_data, dict):
                roi_dict = subject_data
            elif isinstance(subject_data, np.ndarray) and subject_data.dtype.names:
                roi_dict = {}
                for field_name in subject_data.dtype.names:
                    roi_dict[field_name] = subject_data[field_name][0, 0]
            else:
                print(f"  ❌ {subject} 数据格式错误，跳过")
                continue
            
            for roi_key in roi_dict.keys():
                if roi_key.startswith('__'):
                    continue
                
                try:
                    # 获取ROI RDM
                    roi_info = roi_dict[roi_key]
                    
                    if isinstance(roi_info, np.ndarray) and roi_info.dtype.names:
                        roi_rdm = roi_info['rdm'][0, 0].flatten()
                    elif isinstance(roi_info, dict):
                        roi_rdm = roi_info['rdm'].flatten()
                    else:
                        roi_rdm = roi_info.flatten()
                    
                    # 获取噪声天花板
                    noise_ceiling = noise_ceilings[subject][roi_key]
                    
                    print(f"  {roi_key}: RDM长度 {len(roi_rdm)}, 噪声天花板 {noise_ceiling:.3f}")
                    
                    # 运行带噪声天花板校正的RSA分析
                    result = self.analyze_subject_roi_with_noise_ceiling(
                        subject, roi_key, roi_rdm, embedding_rdm, noise_ceiling)
                    rsa_results.append(result)
                    
                    print(f"    原始相关性: {result['raw_correlation']:.3f}")
                    print(f"    校正后相关性: {result['corrected_correlation']:.3f}")
                    print(f"    传统方法: {result['traditional_corr']:.3f}")
                    
                except Exception as e:
                    print(f"  ❌ {roi_key} 分析失败: {e}")
        
        return rsa_results
    
    def save_results_with_noise_ceiling(self, rsa_results, save_dir="rsa_results_noise_ceiling"):
        """保存带噪声天花板校正的结果 - 按被试和ROI分开保存"""
        print(f"\n保存带噪声天花板校正的RSA结果到 {save_dir}...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存所有结果
        savemat(os.path.join(save_dir, 'rsa_results_with_noise_ceiling.mat'), {'rsa_results': rsa_results})
        print(f"✅ 结果已保存: {save_dir}/rsa_results_with_noise_ceiling.mat")
        
        # 保存CSV格式
        import pandas as pd
        df = pd.DataFrame(rsa_results)
        df.to_csv(os.path.join(save_dir, 'rsa_results_with_noise_ceiling.csv'), index=False, encoding='utf-8')
        print(f"✅ 结果已保存: {save_dir}/rsa_results_with_noise_ceiling.csv")
        
        # 按被试分开保存
        subjects_dir = os.path.join(save_dir, "by_subject")
        os.makedirs(subjects_dir, exist_ok=True)
        
        subjects = set([r['subject'] for r in rsa_results])
        for subject in subjects:
            subject_results = [r for r in rsa_results if r['subject'] == subject]
            savemat(os.path.join(subjects_dir, f'{subject}_results.mat'), {'rsa_results': subject_results})
            
            # 保存CSV
            df_subject = pd.DataFrame(subject_results)
            df_subject.to_csv(os.path.join(subjects_dir, f'{subject}_results.csv'), index=False, encoding='utf-8')
        
        print(f"✅ 按被试分开保存: {subjects_dir}/")
        
        # 按ROI分开保存
        rois_dir = os.path.join(save_dir, "by_roi")
        os.makedirs(rois_dir, exist_ok=True)
        
        rois = set([r['roi_key'] for r in rsa_results])
        for roi in rois:
            roi_results = [r for r in rsa_results if r['roi_key'] == roi]
            savemat(os.path.join(rois_dir, f'{roi}_results.mat'), {'rsa_results': roi_results})
            
            # 保存CSV
            df_roi = pd.DataFrame(roi_results)
            df_roi.to_csv(os.path.join(rois_dir, f'{roi}_results.csv'), index=False, encoding='utf-8')
        
        print(f"✅ 按ROI分开保存: {rois_dir}/")
        
        # 创建可视化
        self.create_visualizations(rsa_results, save_dir)
        
        # 保存详细分析
        with open(os.path.join(save_dir, 'noise_ceiling_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write("带噪声天花板校正的RSA分析结果\n")
            f.write("="*50 + "\n\n")
            
            # 按ROI分组统计
            roi_stats = {}
            for result in rsa_results:
                roi_key = result['roi_key']
                if roi_key not in roi_stats:
                    roi_stats[roi_key] = {'raw': [], 'corrected': [], 'noise_ceilings': []}
                roi_stats[roi_key]['raw'].append(result['raw_correlation'])
                roi_stats[roi_key]['corrected'].append(result['corrected_correlation'])
                roi_stats[roi_key]['noise_ceilings'].append(result['noise_ceiling'])
            
            f.write("按ROI分组统计:\n")
            f.write("-"*30 + "\n")
            for roi_key, stats in roi_stats.items():
                f.write(f"{roi_key} ({self.roi_labels[int(roi_key.split('_')[1])]}):\n")
                f.write(f"  原始相关性: {np.mean(stats['raw']):.3f} ± {np.std(stats['raw']):.3f}\n")
                f.write(f"  校正后相关性: {np.mean(stats['corrected']):.3f} ± {np.std(stats['corrected']):.3f}\n")
                f.write(f"  噪声天花板: {np.mean(stats['noise_ceilings']):.3f} ± {np.std(stats['noise_ceilings']):.3f}\n")
                f.write(f"  被试数: {len(stats['raw'])}\n\n")
            
            # 总体统计
            raw_corrs = [r['raw_correlation'] for r in rsa_results]
            corrected_corrs = [r['corrected_correlation'] for r in rsa_results]
            noise_ceilings = [r['noise_ceiling'] for r in rsa_results]
            
            f.write("总体统计:\n")
            f.write("-"*30 + "\n")
            f.write(f"原始相关性: {np.mean(raw_corrs):.3f} ± {np.std(raw_corrs):.3f}\n")
            f.write(f"校正后相关性: {np.mean(corrected_corrs):.3f} ± {np.std(corrected_corrs):.3f}\n")
            f.write(f"噪声天花板: {np.mean(noise_ceilings):.3f} ± {np.std(noise_ceilings):.3f}\n")
            f.write(f"校正效果: {np.mean(corrected_corrs) - np.mean(raw_corrs):.3f}\n")
        
        print(f"✅ 详细分析已保存: {save_dir}/noise_ceiling_analysis.txt")
    
    def create_visualizations(self, rsa_results, save_dir):
        """创建可视化图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            plots_dir = os.path.join(save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 转换为DataFrame
            df = pd.DataFrame(rsa_results)
            
            # 1. 原始相关性 vs 校正后相关性散点图
            plt.figure(figsize=(10, 8))
            plt.scatter(df['raw_correlation'], df['corrected_correlation'], alpha=0.7, s=60)
            plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
            plt.xlabel('原始相关性')
            plt.ylabel('校正后相关性')
            plt.title('噪声天花板校正效果')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'correlation_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 按ROI分组的箱线图
            plt.figure(figsize=(15, 8))
            
            # 准备数据
            roi_data = []
            for _, row in df.iterrows():
                roi_name = self.roi_labels[int(row['roi_key'].split('_')[1])]
                roi_data.append({
                    'ROI': roi_name,
                    '原始相关性': row['raw_correlation'],
                    '校正后相关性': row['corrected_correlation'],
                    '噪声天花板': row['noise_ceiling']
                })
            
            roi_df = pd.DataFrame(roi_data)
            
            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 原始相关性
            sns.boxplot(data=roi_df, x='ROI', y='原始相关性', ax=axes[0])
            axes[0].set_title('原始相关性')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 校正后相关性
            sns.boxplot(data=roi_df, x='ROI', y='校正后相关性', ax=axes[1])
            axes[1].set_title('校正后相关性')
            axes[1].tick_params(axis='x', rotation=45)
            
            # 噪声天花板
            sns.boxplot(data=roi_df, x='ROI', y='噪声天花板', ax=axes[2])
            axes[2].set_title('噪声天花板')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'roi_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 按被试分组的箱线图
            plt.figure(figsize=(15, 8))
            
            # 准备数据
            subject_data = []
            for _, row in df.iterrows():
                subject_data.append({
                    '被试': row['subject'],
                    '原始相关性': row['raw_correlation'],
                    '校正后相关性': row['corrected_correlation'],
                    '噪声天花板': row['noise_ceiling']
                })
            
            subject_df = pd.DataFrame(subject_data)
            
            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 原始相关性
            sns.boxplot(data=subject_df, x='被试', y='原始相关性', ax=axes[0])
            axes[0].set_title('原始相关性')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 校正后相关性
            sns.boxplot(data=subject_df, x='被试', y='校正后相关性', ax=axes[1])
            axes[1].set_title('校正后相关性')
            axes[1].tick_params(axis='x', rotation=45)
            
            # 噪声天花板
            sns.boxplot(data=subject_df, x='被试', y='噪声天花板', ax=axes[2])
            axes[2].set_title('噪声天花板')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'subject_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. 相关性分布直方图
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(df['raw_correlation'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('原始相关性分布')
            plt.xlabel('相关性')
            plt.ylabel('频次')
            
            plt.subplot(2, 2, 2)
            plt.hist(df['corrected_correlation'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('校正后相关性分布')
            plt.xlabel('相关性')
            plt.ylabel('频次')
            
            plt.subplot(2, 2, 3)
            plt.hist(df['noise_ceiling'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.title('噪声天花板分布')
            plt.xlabel('噪声天花板')
            plt.ylabel('频次')
            
            plt.subplot(2, 2, 4)
            improvement = df['corrected_correlation'] - df['raw_correlation']
            plt.hist(improvement, bins=20, alpha=0.7, color='gold', edgecolor='black')
            plt.title('校正效果分布')
            plt.xlabel('相关性提升')
            plt.ylabel('频次')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'correlation_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. 热力图：被试 x ROI
            plt.figure(figsize=(12, 8))
            
            # 创建透视表
            pivot_raw = df.pivot(index='subject', columns='roi_key', values='raw_correlation')
            pivot_corrected = df.pivot(index='subject', columns='roi_key', values='corrected_correlation')
            
            # 原始相关性热力图
            plt.subplot(1, 2, 1)
            sns.heatmap(pivot_raw, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': '原始相关性'})
            plt.title('原始相关性热力图')
            plt.xlabel('ROI')
            plt.ylabel('被试')
            
            # 校正后相关性热力图
            plt.subplot(1, 2, 2)
            sns.heatmap(pivot_corrected, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': '校正后相关性'})
            plt.title('校正后相关性热力图')
            plt.xlabel('ROI')
            plt.ylabel('被试')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 6. 为每个被试单独绘制ROI比较图
            subjects = df['subject'].unique()
            for subject in subjects:
                subject_data = df[df['subject'] == subject]
                
                plt.figure(figsize=(12, 8))
                
                # 准备数据
                roi_names = []
                raw_corrs = []
                corrected_corrs = []
                noise_ceilings = []
                
                for _, row in subject_data.iterrows():
                    roi_name = self.roi_labels[int(row['roi_key'].split('_')[1])]
                    roi_names.append(roi_name)
                    raw_corrs.append(row['raw_correlation'])
                    corrected_corrs.append(row['corrected_correlation'])
                    noise_ceilings.append(row['noise_ceiling'])
                
                # 创建子图
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 原始相关性
                axes[0, 0].bar(roi_names, raw_corrs, color='skyblue', alpha=0.7, edgecolor='black')
                axes[0, 0].set_title(f'{subject} - 原始相关性')
                axes[0, 0].set_ylabel('相关性')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
                
                # 校正后相关性
                axes[0, 1].bar(roi_names, corrected_corrs, color='lightgreen', alpha=0.7, edgecolor='black')
                axes[0, 1].set_title(f'{subject} - 校正后相关性')
                axes[0, 1].set_ylabel('相关性')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
                
                # 噪声天花板
                axes[1, 0].bar(roi_names, noise_ceilings, color='lightcoral', alpha=0.7, edgecolor='black')
                axes[1, 0].set_title(f'{subject} - 噪声天花板')
                axes[1, 0].set_ylabel('噪声天花板')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
                
                # 校正效果
                improvement = [c - r for c, r in zip(corrected_corrs, raw_corrs)]
                axes[1, 1].bar(roi_names, improvement, color='gold', alpha=0.7, edgecolor='black')
                axes[1, 1].set_title(f'{subject} - 校正效果')
                axes[1, 1].set_ylabel('相关性提升')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{subject}_roi_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"✅ 可视化图表已保存: {plots_dir}/")
            print(f"  - correlation_comparison.png: 相关性比较")
            print(f"  - roi_comparison.png: ROI比较")
            print(f"  - subject_comparison.png: 被试比较")
            print(f"  - correlation_distributions.png: 相关性分布")
            print(f"  - correlation_heatmap.png: 相关性热力图")
            print(f"  - {len(subjects)}个被试的ROI分析图: {[f'{s}_roi_analysis.png' for s in subjects]}")
            
        except ImportError as e:
            print(f"⚠️ 可视化库未安装，跳过图表生成: {e}")
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")


def main():
    """主函数"""
    # 创建带噪声天花板校正的RSA分析器
    analyzer = RSAAnalyzerWithNoiseCeiling()
    
    # 运行分析
    rsa_results = analyzer.run_rsa_analysis_with_noise_ceiling()
    
    if rsa_results:
        # 保存结果
        analyzer.save_results_with_noise_ceiling(rsa_results)
        
        print("\n" + "="*60)
        print("🎉 带噪声天花板校正的RSA分析完成!")
        print("="*60)
        
        print(f"\n📊 分析结果:")
        raw_corrs = [r['raw_correlation'] for r in rsa_results]
        corrected_corrs = [r['corrected_correlation'] for r in rsa_results]
        noise_ceilings = [r['noise_ceiling'] for r in rsa_results]
        
        print(f"  - 原始相关性: {np.mean(raw_corrs):.3f} ± {np.std(raw_corrs):.3f}")
        print(f"  - 校正后相关性: {np.mean(corrected_corrs):.3f} ± {np.std(corrected_corrs):.3f}")
        print(f"  - 噪声天花板: {np.mean(noise_ceilings):.3f} ± {np.std(noise_ceilings):.3f}")
        print(f"  - 校正效果: {np.mean(corrected_corrs) - np.mean(raw_corrs):.3f}")
        
        print(f"\n📁 生成的文件:")
        print(f"  - rsa_results_noise_ceiling/rsa_results_with_noise_ceiling.mat")
        print(f"  - rsa_results_noise_ceiling/rsa_results_with_noise_ceiling.csv")
        print(f"  - rsa_results_noise_ceiling/by_subject/ (按被试分开)")
        print(f"  - rsa_results_noise_ceiling/by_roi/ (按ROI分开)")
        print(f"  - rsa_results_noise_ceiling/plots/ (可视化图表)")
        print(f"    * 总体分析图: 5个汇总图表")
        print(f"    * 个别被试图: 每个被试的ROI分析图")
        print(f"  - rsa_results_noise_ceiling/noise_ceiling_analysis.txt")
        
    else:
        print("❌ 没有成功分析任何数据")


if __name__ == "__main__":
    main()
