#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human fMRI RSA分析 - 支持多种embedding类型选择

支持以下embedding类型：
1. image_embeddings - 完整caption embedding
2. word_average_embeddings - 单词平均embedding  
3. noun_embeddings - 名词embedding
4. verb_embeddings - 动词embedding

使用方法：
python rsa_with_embedding_choice.py --embedding_type image
python rsa_with_embedding_choice.py --embedding_type word_average
python rsa_with_embedding_choice.py --embedding_type noun
python rsa_with_embedding_choice.py --embedding_type verb
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat, savemat

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HumanRSAAnalyzer:
    def __init__(self, embedding_type='image', use_zscore=True):
        """
        初始化Human RSA分析器
        
        Args:
            embedding_type: embedding类型 ('image', 'word_average', 'noun', 'verb')
            use_zscore: 是否使用z-score归一化的fMRI数据
        """
        self.embedding_type = embedding_type
        self.use_zscore = use_zscore
        
        # 设置文件路径
        self.embedding_base_path = r"E:\lunzhuan1\visuo_llm-main\newtry\captions\embeddings_output"
        self.roi_rdm_file = 'roi_rdm_results/all_subjects_roi_rdms.mat'
        
        # 设置输出目录
        self.output_suffix = '_zscore' if use_zscore else '_original'
        self.output_dir = f'rsa_results_{embedding_type}{self.output_suffix}'
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Human RSA分析器初始化完成")
        print(f"  Embedding类型: {embedding_type}")
        print(f"  使用Z-score归一化: {use_zscore}")
        print(f"  输出目录: {self.output_dir}")
    
    def load_data(self):
        """加载embedding和fMRI数据"""
        print("\n正在加载数据...")
        
        # 加载embedding数据
        embedding_files = {
            'image': 'image_embeddings.npy',
            'word_average': 'word_average_embeddings.npy',
            'noun': 'noun_embeddings.npy',
            'verb': 'verb_embeddings.npy'
        }
        
        embedding_path = os.path.join(self.embedding_base_path, embedding_files[self.embedding_type])
        
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding文件不存在: {embedding_path}")
        
        self.embeddings = np.load(embedding_path)
        print(f"  加载embedding: {self.embeddings.shape}")
        
        # 加载fMRI ROI数据
        if not os.path.exists(self.roi_rdm_file):
            raise FileNotFoundError(f"ROI RDM文件不存在: {self.roi_rdm_file}")
        
        roi_data = loadmat(self.roi_rdm_file)
        
        if self.use_zscore:
            # 使用z-score归一化的数据
            self.roi_rdms = roi_data['roi_rdms_zscore']
            self.roi_labels = roi_data['roi_labels_zscore']
            self.subject_ids = roi_data['subject_ids_zscore']
            normalization_method = 'z-score per voxel'
        else:
            # 使用原始数据
            self.roi_rdms = roi_data['roi_rdms']
            self.roi_labels = roi_data['roi_labels']
            self.subject_ids = roi_data['subject_ids']
            normalization_method = 'None'
        
        print(f"  加载ROI RDM: {self.roi_rdms.shape}")
        print(f"  ROI标签: {len(self.roi_labels)} 个")
        print(f"  被试数量: {len(self.subject_ids)}")
        print(f"  归一化方法: {normalization_method}")
        
        return normalization_method
    
    def compute_model_rdm(self):
        """计算模型RDM"""
        print("\n正在计算模型RDM...")
        
        # 计算embedding之间的距离矩阵
        distances = pdist(self.embeddings, metric='correlation')
        self.model_rdm = squareform(distances)
        
        print(f"  模型RDM形状: {self.model_rdm.shape}")
        print(f"  模型RDM范围: {self.model_rdm.min():.4f} - {self.model_rdm.max():.4f}")
        
        return self.model_rdm
    
    def compute_rsa_correlation(self, neural_rdm):
        """计算RSA相关性"""
        # 提取上三角矩阵（不包括对角线）
        model_upper = self.model_rdm[np.triu_indices_from(self.model_rdm, k=1)]
        neural_upper = neural_rdm[np.triu_indices_from(neural_rdm, k=1)]
        
        # 计算Spearman相关性
        correlation, p_value = spearmanr(model_upper, neural_upper)
        
        return correlation, p_value
    
    def analyze_roi_rsa(self):
        """分析ROI级别的RSA"""
        print(f"\n开始ROI级别RSA分析...")
        
        results = []
        
        for roi_idx in range(len(self.roi_labels)):
            roi_label = self.roi_labels[roi_idx][0]  # 提取ROI标签字符串
            print(f"  分析ROI: {roi_label}")
            
            # 获取该ROI的所有被试数据
            roi_data = self.roi_rdms[roi_idx]  # shape: (n_subjects, n_stimuli, n_stimuli)
            
            roi_correlations = []
            roi_p_values = []
            
            for subj_idx in range(roi_data.shape[0]):
                subject_id = self.subject_ids[subj_idx][0]
                neural_rdm = roi_data[subj_idx]
                
                # 计算RSA相关性
                correlation, p_value = self.compute_rsa_correlation(neural_rdm)
                
                roi_correlations.append(correlation)
                roi_p_values.append(p_value)
                
                results.append({
                    'ROI': roi_label,
                    'Subject': subject_id,
                    'Correlation': correlation,
                    'P_Value': p_value,
                    'Embedding_Type': self.embedding_type,
                    'Use_Zscore': self.use_zscore
                })
            
            # 计算该ROI的平均相关性
            mean_correlation = np.mean(roi_correlations)
            std_correlation = np.std(roi_correlations)
            mean_p_value = np.mean(roi_p_values)
            
            print(f"    平均相关性: {mean_correlation:.4f} ± {std_correlation:.4f}")
            print(f"    平均P值: {mean_p_value:.4f}")
        
        return results
    
    def plot_rsa_results(self, results):
        """绘制RSA结果"""
        print("\n正在绘制RSA结果...")
        
        # 按ROI分组计算统计
        roi_stats = {}
        for result in results:
            roi = result['ROI']
            if roi not in roi_stats:
                roi_stats[roi] = []
            roi_stats[roi].append(result['Correlation'])
        
        # 准备绘图数据
        roi_labels = list(roi_stats.keys())
        roi_means = [np.mean(roi_stats[roi]) for roi in roi_labels]
        roi_stds = [np.std(roi_stats[roi]) for roi in roi_labels]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1: ROI平均相关性条形图
        bars = ax1.bar(range(len(roi_labels)), roi_means, yerr=roi_stds, 
                      capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('ROI')
        ax1.set_ylabel('RSA相关性')
        ax1.set_title(f'ROI级别RSA分析 - {self.embedding_type} embedding')
        ax1.set_xticks(range(len(roi_labels)))
        ax1.set_xticklabels(roi_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (mean, std) in enumerate(zip(roi_means, roi_stds)):
            ax1.text(i, mean + std + 0.01, f'{mean:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 子图2: 所有被试相关性分布
        all_correlations = [result['Correlation'] for result in results]
        ax2.hist(all_correlations, bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('RSA相关性')
        ax2.set_ylabel('频次')
        ax2.set_title(f'所有被试相关性分布 - {self.embedding_type} embedding')
        ax2.axvline(np.mean(all_correlations), color='red', linestyle='--', 
                   label=f'均值: {np.mean(all_correlations):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形
        plot_file = os.path.join(self.output_dir, f'rsa_analysis_{self.embedding_type}{self.output_suffix}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  图形已保存: {plot_file}")
    
    def save_results(self, results):
        """保存RSA结果"""
        print("\n正在保存结果...")
        
        # 保存为CSV
        import pandas as pd
        df = pd.DataFrame(results)
        csv_file = os.path.join(self.output_dir, f'rsa_results_{self.embedding_type}{self.output_suffix}.csv')
        df.to_csv(csv_file, index=False)
        print(f"  CSV文件已保存: {csv_file}")
        
        # 保存为MAT文件
        mat_file = os.path.join(self.output_dir, f'rsa_results_{self.embedding_type}{self.output_suffix}.mat')
        savemat(mat_file, {
            'results': results,
            'embedding_type': self.embedding_type,
            'use_zscore': self.use_zscore,
            'model_rdm': self.model_rdm
        })
        print(f"  MAT文件已保存: {mat_file}")
        
        # 保存为Pickle文件
        pickle_file = os.path.join(self.output_dir, f'rsa_results_{self.embedding_type}{self.output_suffix}.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'embedding_type': self.embedding_type,
                'use_zscore': self.use_zscore,
                'model_rdm': self.model_rdm,
                'roi_labels': self.roi_labels,
                'subject_ids': self.subject_ids
            }, f)
        print(f"  Pickle文件已保存: {pickle_file}")
    
    def run_analysis(self):
        """运行完整的RSA分析"""
        print("=" * 60)
        print(f"Human fMRI RSA分析 - {self.embedding_type} embedding")
        print("=" * 60)
        
        # 1. 加载数据
        normalization_method = self.load_data()
        
        # 2. 计算模型RDM
        self.compute_model_rdm()
        
        # 3. 分析ROI RSA
        results = self.analyze_roi_rsa()
        
        # 4. 绘制结果
        self.plot_rsa_results(results)
        
        # 5. 保存结果
        self.save_results(results)
        
        # 6. 打印总结
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        
        # 计算总体统计
        all_correlations = [result['Correlation'] for result in results]
        print(f"总体统计:")
        print(f"  总分析数: {len(results)}")
        print(f"  平均相关性: {np.mean(all_correlations):.4f} ± {np.std(all_correlations):.4f}")
        print(f"  相关性范围: {np.min(all_correlations):.4f} - {np.max(all_correlations):.4f}")
        print(f"  显著相关数 (p<0.05): {sum(1 for r in results if r['P_Value'] < 0.05)}")
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Human fMRI RSA分析 - 支持多种embedding类型')
    parser.add_argument('--embedding_type', type=str, default='image',
                       choices=['image', 'word_average', 'noun', 'verb'],
                       help='Embedding类型选择')
    parser.add_argument('--use_zscore', action='store_true', default=True,
                       help='是否使用z-score归一化的fMRI数据')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = HumanRSAAnalyzer(
        embedding_type=args.embedding_type,
        use_zscore=args.use_zscore
    )
    
    # 运行分析
    results = analyzer.run_analysis()
    
    print(f"\n分析完成! 结果保存在: {analyzer.output_dir}")

if __name__ == "__main__":
    main()
