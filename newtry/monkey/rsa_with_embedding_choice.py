#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
猴子神经元RSA分析 - 支持多种embedding类型选择

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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MonkeyRSAAnalyzer:
    def __init__(self, embedding_type='image', use_zscore=True):
        """
        初始化RSA分析器
        
        Args:
            embedding_type: embedding类型 ('image', 'word_average', 'noun', 'verb')
            use_zscore: 是否使用z-score归一化的猴子数据
        """
        self.embedding_type = embedding_type
        self.use_zscore = use_zscore
        
        # 设置文件路径
        self.embedding_base_path = r"E:\lunzhuan1\visuo_llm-main\newtry\captions\embeddings_output"
        self.rdm_file = 'all_rdms_correlation_zscore.pkl' if use_zscore else 'all_rdms_correlation.pkl'
        self.original_data_file = 'extracted_monkey_responses.pkl'
        
        # 设置输出目录
        self.output_suffix = '_zscore' if use_zscore else '_original'
        self.output_dir = f'rsa_results_{embedding_type}{self.output_suffix}'
        
        # embedding文件映射
        self.embedding_files = {
            'image': 'image_embeddings.npy',
            'word_average': 'word_average_embeddings.npy', 
            'noun': 'noun_embeddings.npy',
            'verb': 'verb_embeddings.npy'
        }
        
        # embedding描述
        self.embedding_descriptions = {
            'image': '完整Caption Embedding',
            'word_average': '单词平均Embedding',
            'noun': '名词Embedding', 
            'verb': '动词Embedding'
        }
        
        print(f"🔧 RSA分析器初始化完成")
        print(f"   Embedding类型: {self.embedding_descriptions[self.embedding_type]}")
        print(f"   数据归一化: {'z-score归一化' if use_zscore else '原始数据'}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_data(self):
        """加载数据"""
        print(f"\n📂 1. 加载数据...")
        
        # 加载猴子RDM
        print(f"   加载猴子RDM: {self.rdm_file}")
        with open(self.rdm_file, 'rb') as f:
            rdm_data = pickle.load(f)
            monkey_rdms = rdm_data['all_rdms']
        
        # 加载原始数据
        print(f"   加载原始数据: {self.original_data_file}")
        with open(self.original_data_file, 'rb') as f:
            original_data = pickle.load(f)
        
        # 加载选择的embedding
        embedding_file = os.path.join(self.embedding_base_path, self.embedding_files[self.embedding_type])
        print(f"   加载Embedding: {embedding_file}")
        
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding文件不存在: {embedding_file}")
        
        embeddings = np.load(embedding_file)
        
        # 确保embedding形状正确 (1000, embedding_dim)
        if embeddings.shape[0] == 1000:
            pass
        elif embeddings.shape[1] == 1000:
            embeddings = embeddings.T
        else:
            if embeddings.shape[0] > 1000:
                embeddings = embeddings[:1000, :]
            elif embeddings.shape[1] > 1000:
                embeddings = embeddings[:, :1000]
        
        # 计算embedding RDM
        llm_rdm = 1 - np.corrcoef(embeddings)
        
        print(f"   ✅ 猴子RDM数量: {len(monkey_rdms)}")
        print(f"   ✅ Embedding形状: {embeddings.shape}")
        print(f"   ✅ LLM RDM形状: {llm_rdm.shape}")
        print(f"   ✅ 归一化方法: {'z-score per neuron' if self.use_zscore else '原始数据'}")
        
        return monkey_rdms, original_data, llm_rdm, embeddings
    
    def compute_noise_ceiling(self, rdms):
        """
        计算noise ceiling
        
        Args:
            rdms: list of RDM matrices for the same arealabel
            
        Returns:
            individual_noise_ceilings: 每个session的独立noise ceiling值
            mean_noise_ceiling: 平均的noise ceiling值
        """
        if len(rdms) < 2:
            return [], None
        
        individual_noise_ceilings = []
        
        # 对每个session计算独立的noise ceiling
        for i in range(len(rdms)):
            # 当前session的RDM
            current_rdm = rdms[i]
            if isinstance(current_rdm, dict):
                current_rdm = current_rdm['rdm']
            
            # 其他session的RDM
            other_rdms = [rdms[j] for j in range(len(rdms)) if j != i]
            
            # 计算其他session的平均RDM
            mean_other_rdm = np.zeros_like(current_rdm)
            for other_rdm in other_rdms:
                if isinstance(other_rdm, dict):
                    other_rdm = other_rdm['rdm']
                mean_other_rdm += other_rdm
            mean_other_rdm /= len(other_rdms)
            
            # 计算当前session RDM与其他session平均RDM的相关性
            rsa, _ = self.compute_rsa(current_rdm, mean_other_rdm)
            individual_noise_ceilings.append(rsa)
        
        # 计算平均noise ceiling
        mean_noise_ceiling = np.mean(individual_noise_ceilings)
        
        return individual_noise_ceilings, mean_noise_ceiling
    
    def compute_rsa(self, rdm1, rdm2):
        """计算RSA"""
        # 如果RDM是字典格式，提取实际的RDM矩阵
        if isinstance(rdm1, dict):
            rdm1 = rdm1['rdm']
        if isinstance(rdm2, dict):
            rdm2 = rdm2['rdm']
        
        # 确保两个RDM形状相同
        if rdm1.shape != rdm2.shape:
            min_size = min(rdm1.shape[0], rdm2.shape[0])
            rdm1 = rdm1[:min_size, :min_size]
            rdm2 = rdm2[:min_size, :min_size]
        
        # 计算上三角矩阵的相关性（排除对角线）
        mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
        rdm1_values = rdm1[mask]
        rdm2_values = rdm2[mask]
        
        correlation, p_value = spearmanr(rdm1_values, rdm2_values)
        return correlation, p_value
    
    def compute_corrected_rsa(self, rsa_value, noise_ceiling):
        """
        计算矫正后的RSA值
        
        Args:
            rsa_value: 原始RSA值
            noise_ceiling: noise ceiling值
            
        Returns:
            corrected_rsa: 矫正后的RSA值
            warning: 警告信息（如果有）
        """
        if noise_ceiling is None or noise_ceiling <= 0:
            return rsa_value, None
        
        # 直接除法：RSA / noise_ceiling
        corrected_rsa = rsa_value / noise_ceiling
        
        # 检查是否有问题
        warning = None
        if corrected_rsa > 1.0:
            warning = f"⚠️ 矫正后RSA > 1.0 ({corrected_rsa:.3f})，可能数据有问题"
            corrected_rsa = 1.0  # 限制在1以内
        elif corrected_rsa > 0.8:
            warning = f"⚠️ 矫正后RSA较高 ({corrected_rsa:.3f})，请检查数据质量"
        
        return corrected_rsa, warning
    
    def analyze_arealabel_with_noise_ceiling(self, arealabel, sessions_data, llm_rdm):
        """分析单个arealabel的RSA和noise ceiling"""
        
        print(f"\n🔍 分析 {arealabel} ({len(sessions_data)}个session) - {self.embedding_descriptions[self.embedding_type]}:")
        
        # 提取RDM
        rdms = [session['rdm'] for session in sessions_data]
        session_nums = [session['session'] for session in sessions_data]
        
        # 计算noise ceiling
        individual_noise_ceilings, mean_noise_ceiling = self.compute_noise_ceiling(rdms)
        
        if not individual_noise_ceilings:
            print(f"   ⚠️  只有1个session，无法计算noise ceiling")
            return None
        
        print(f"   平均Noise ceiling: {mean_noise_ceiling:.4f}")
        print(f"   各session Noise ceiling: {[f'{nc:.4f}' for nc in individual_noise_ceilings]}")
        
        # 计算每个session与LLM的RSA
        session_rsas = []
        corrected_rsas = []
        warnings = []
        
        for i, (session_num, rdm) in enumerate(zip(session_nums, rdms)):
            rsa, p_value = self.compute_rsa(rdm, llm_rdm)
            
            # 使用该session的独立noise ceiling进行矫正
            corrected_rsa, warning = self.compute_corrected_rsa(rsa, individual_noise_ceilings[i])
            
            session_rsas.append(rsa)
            corrected_rsas.append(corrected_rsa)
            warnings.append(warning)
            
            print(f"     Session {session_num}:")
            print(f"       原始RSA: {rsa:.4f} (p={p_value:.4f})")
            print(f"       矫正RSA: {corrected_rsa:.4f}")
            if warning:
                print(f"       {warning}")
        
        # 计算统计量
        mean_rsa = np.mean(session_rsas)
        std_rsa = np.std(session_rsas)
        mean_corrected_rsa = np.mean(corrected_rsas)
        std_corrected_rsa = np.std(corrected_rsas)
        
        print(f"   原始RSA: {mean_rsa:.4f} ± {std_rsa:.4f}")
        print(f"   矫正RSA: {mean_corrected_rsa:.4f} ± {std_corrected_rsa:.4f}")
        
        return {
            'arealabel': arealabel,
            'n_sessions': len(sessions_data),
            'mean_noise_ceiling': mean_noise_ceiling,
            'individual_noise_ceilings': individual_noise_ceilings,
            'session_rsas': session_rsas,
            'corrected_rsas': corrected_rsas,
            'warnings': warnings,
            'mean_rsa': mean_rsa,
            'std_rsa': std_rsa,
            'mean_corrected_rsa': mean_corrected_rsa,
            'std_corrected_rsa': std_corrected_rsa,
            'session_nums': session_nums,
            'embedding_type': self.embedding_type,
            'use_zscore': self.use_zscore
        }
    
    def create_individual_arealabel_plots(self, results):
        """为每个arealabel创建单独的图表"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 过滤有效结果
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("没有有效的结果可以绘图")
            return
        
        for result in valid_results:
            arealabel = result['arealabel']
            session_nums = result['session_nums']
            session_rsas = result['session_rsas']
            corrected_rsas = result['corrected_rsas']
            warnings = result['warnings']
            mean_noise_ceiling = result['mean_noise_ceiling']
            individual_noise_ceilings = result['individual_noise_ceilings']
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            title = f'RSA Analysis for {arealabel} - {self.embedding_descriptions[self.embedding_type]}'
            if self.use_zscore:
                title += ' (z-score归一化)'
            title += f'\nMean Noise Ceiling: {mean_noise_ceiling:.4f}'
            
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # 1. 原始RSA值条形图
            sessions = [f"Session {s}" for s in session_nums]
            bars1 = ax1.bar(sessions, session_rsas, color='skyblue', alpha=0.7, edgecolor='navy')
            ax1.set_title(f'Original RSA Values by Session\nMean: {np.mean(session_rsas):.4f} ± {np.std(session_rsas):.4f}')
            ax1.set_ylabel('Original RSA Value')
            ax1.set_xlabel('Session')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, rsa_value, nc_value in zip(bars1, session_rsas, individual_noise_ceilings):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{rsa_value:.4f}', ha='center', va='bottom', fontsize=9)
                ax1.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                        f'NC: {nc_value:.3f}', ha='center', va='top', fontsize=8, color='red')
            
            ax1.axhline(y=np.mean(session_rsas), color='red', linestyle='--', alpha=0.7, 
                        label=f'Mean: {np.mean(session_rsas):.4f}')
            ax1.legend()
            
            # 2. 矫正后RSA值条形图
            bars2 = ax2.bar(sessions, corrected_rsas, color='lightcoral', alpha=0.7, edgecolor='darkred')
            ax2.set_title(f'Corrected RSA Values by Session\nMean: {np.mean(corrected_rsas):.4f} ± {np.std(corrected_rsas):.4f}')
            ax2.set_ylabel('Corrected RSA Value')
            ax2.set_xlabel('Session')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, corr_value, nc_value in zip(bars2, corrected_rsas, individual_noise_ceilings):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{corr_value:.4f}', ha='center', va='bottom', fontsize=9)
                ax2.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                        f'NC: {nc_value:.3f}', ha='center', va='top', fontsize=8, color='red')
            
            ax2.axhline(y=np.mean(corrected_rsas), color='red', linestyle='--', alpha=0.7, 
                        label=f'Mean: {np.mean(corrected_rsas):.4f}')
            ax2.legend()
            
            # 3. 原始vs矫正对比
            x = np.arange(len(sessions))
            width = 0.35
            
            bars3_orig = ax3.bar(x - width/2, session_rsas, width, label='Original RSA', alpha=0.7, color='skyblue')
            bars3_corr = ax3.bar(x + width/2, corrected_rsas, width, label='Corrected RSA', alpha=0.7, color='lightcoral')
            
            ax3.set_title('Original vs Corrected RSA Comparison')
            ax3.set_ylabel('RSA Value')
            ax3.set_xlabel('Session')
            ax3.set_xticks(x)
            ax3.set_xticklabels(sessions, rotation=45)
            ax3.legend()
            
            # 添加数值标签
            for bar, value, nc_value in zip(bars3_orig, session_rsas, individual_noise_ceilings):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                ax3.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                        f'NC:{nc_value:.2f}', ha='center', va='top', fontsize=7, color='red')
            
            for bar, value in zip(bars3_corr, corrected_rsas):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 4. 矫正比例信息
            correction_ratios = [corrected_rsas[i] / session_rsas[i] for i in range(len(session_rsas))]
            
            bars4 = ax4.bar(sessions, correction_ratios, color='orange', alpha=0.7, edgecolor='darkorange')
            ax4.set_title('Correction Ratio by Session (Corrected/Original)')
            ax4.set_ylabel('Correction Ratio')
            ax4.set_xlabel('Session')
            ax4.tick_params(axis='x', rotation=45)
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No correction')
            ax4.legend()
            
            # 添加数值标签
            for bar, ratio in zip(bars4, correction_ratios):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = f'{self.output_dir}/rsa_{arealabel}_{self.embedding_type}{self.output_suffix}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"   图表已保存: {plot_file}")
            plt.close()
            
            # 创建详细表格
            import pandas as pd
            table_data = []
            for i, session_num in enumerate(session_nums):
                table_data.append({
                    'Session': session_num,
                    'Original_RSA': session_rsas[i],
                    'Corrected_RSA': corrected_rsas[i],
                    'Correction_Ratio': correction_ratios[i],
                    'Noise_Ceiling': individual_noise_ceilings[i],
                    'Warning': warnings[i] if warnings[i] else '',
                    'Embedding_Type': self.embedding_type,
                    'Use_Zscore': self.use_zscore
                })
            
            df = pd.DataFrame(table_data)
            df = df.sort_values('Corrected_RSA', ascending=False)
            
            # 保存表格
            table_file = f'{self.output_dir}/rsa_table_{arealabel}_{self.embedding_type}{self.output_suffix}.csv'
            df.to_csv(table_file, index=False)
            print(f"   表格已保存: {table_file}")
    
    def create_summary_plots(self, results):
        """创建汇总图表"""
        # 过滤有效结果
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("没有有效的结果可以绘图")
            return None
        
        arealabels = [r['arealabel'] for r in valid_results]
        noise_ceilings = [r['mean_noise_ceiling'] for r in valid_results]
        mean_rsas = [r['mean_rsa'] for r in valid_results]
        mean_corrected_rsas = [r['mean_corrected_rsa'] for r in valid_results]
        n_sessions = [r['n_sessions'] for r in valid_results]
        
        # 创建2x2的子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        title = f'RSA Analysis with {self.embedding_descriptions[self.embedding_type]}'
        if self.use_zscore:
            title += ' (z-score归一化版本)'
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Noise ceiling分布
        bars1 = ax1.bar(arealabels, noise_ceilings, color='lightblue', alpha=0.7, edgecolor='navy')
        ax1.set_title('Noise Ceiling by Arealabel', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Noise Ceiling')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars1, noise_ceilings):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 原始RSA vs 矫正RSA
        x = np.arange(len(arealabels))
        width = 0.35
        
        bars2_orig = ax2.bar(x - width/2, mean_rsas, width, label='Original RSA', alpha=0.7, color='skyblue')
        bars2_corr = ax2.bar(x + width/2, mean_corrected_rsas, width, label='Corrected RSA', alpha=0.7, color='lightcoral')
        
        ax2.set_title('Original vs Corrected RSA', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RSA Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels(arealabels, rotation=45)
        ax2.legend()
        
        # 添加数值标签
        for bar, value in zip(bars2_orig, mean_rsas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2_corr, mean_corrected_rsas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 矫正后的RSA排序
        sorted_indices = np.argsort(mean_corrected_rsas)[::-1]
        sorted_arealabels = [arealabels[i] for i in sorted_indices]
        sorted_corrected_rsas = [mean_corrected_rsas[i] for i in sorted_indices]
        
        bars3 = ax3.bar(sorted_arealabels, sorted_corrected_rsas, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax3.set_title('Corrected RSA (Sorted)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Corrected RSA Value')
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars3, sorted_corrected_rsas):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 矫正效果（矫正比例）
        correction_ratios = [corrected/original for original, corrected in zip(mean_rsas, mean_corrected_rsas)]
        bars4 = ax4.bar(arealabels, correction_ratios, color='orange', alpha=0.7, edgecolor='darkorange')
        ax4.set_title('Correction Ratio (Corrected/Original)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Correction Ratio')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No correction')
        ax4.legend()
        
        # 添加数值标签
        for bar, value in zip(bars4, correction_ratios):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = f'{self.output_dir}/rsa_summary_{self.embedding_type}{self.output_suffix}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   汇总图表已保存: {plot_file}")
        plt.close()
        
        # 创建详细表格
        import pandas as pd
        table_data = []
        for result in results:
            if result is not None:
                table_data.append({
                    'Arealabel': result['arealabel'],
                    'N_Sessions': result['n_sessions'],
                    'Mean_Noise_Ceiling': result['mean_noise_ceiling'],
                    'Mean_Original_RSA': result['mean_rsa'],
                    'Std_Original_RSA': result['std_rsa'],
                    'Mean_Corrected_RSA': result['mean_corrected_rsa'],
                    'Std_Corrected_RSA': result['std_corrected_rsa'],
                    'Mean_Correction_Ratio': np.mean([result['corrected_rsas'][i] / result['session_rsas'][i] for i in range(len(result['session_rsas']))]) if len(result['session_rsas']) > 0 else 1.0,
                    'Embedding_Type': self.embedding_type,
                    'Use_Zscore': self.use_zscore,
                    'Normalization': 'zscore_per_neuron' if self.use_zscore else 'original_data'
                })
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('Mean_Corrected_RSA', ascending=False)
        
        # 保存表格
        table_file = f'{self.output_dir}/rsa_summary_{self.embedding_type}{self.output_suffix}.csv'
        df.to_csv(table_file, index=False)
        print(f"   汇总表格已保存: {table_file}")
        
        return df
    
    def run_analysis(self):
        """运行完整的RSA分析"""
        print(f"\n🚀 开始RSA分析...")
        print(f"   Embedding类型: {self.embedding_descriptions[self.embedding_type]}")
        print(f"   数据归一化: {'z-score归一化' if self.use_zscore else '原始数据'}")
        
        # 加载数据
        monkey_rdms, original_data, llm_rdm, embeddings = self.load_data()
        
        # 按arealabel分组
        print(f"\n📊 2. 按arealabel分组...")
        arealabel_groups = {}
        for session_num, session_data in original_data['extracted_data'].items():
            if session_num in monkey_rdms:
                for roi_index, roi_data in session_data['rois'].items():
                    arealabel = roi_data['arealabel']
                    if arealabel not in arealabel_groups:
                        arealabel_groups[arealabel] = []
                    
                    # 从数据结构中获取RDM
                    if arealabel in monkey_rdms[session_num]:
                        arealabel_groups[arealabel].append({
                            'session': session_num,
                            'rdm': monkey_rdms[session_num][arealabel]
                        })
        
        print("   Arealabel分组结果:")
        for arealabel, sessions in arealabel_groups.items():
            print(f"     {arealabel}: {len(sessions)}个session")
        
        # 分析每个arealabel
        print(f"\n🔍 3. 分析每个arealabel...")
        results = []
        
        for arealabel, sessions in arealabel_groups.items():
            result = self.analyze_arealabel_with_noise_ceiling(arealabel, sessions, llm_rdm)
            results.append(result)
        
        # 创建图表和表格
        print(f"\n📈 4. 创建图表和表格...")
        
        # 为每个arealabel创建单独的图表
        print("   创建每个arealabel的单独图表...")
        self.create_individual_arealabel_plots(results)
        
        # 创建汇总图表
        print("   创建汇总图表...")
        df = self.create_summary_plots(results)
        
        # 显示结果摘要
        print(f"\n📋 === 结果摘要 ===")
        print(f"   Embedding类型: {self.embedding_descriptions[self.embedding_type]}")
        print(f"   数据归一化: {'z-score归一化' if self.use_zscore else '原始数据'}")
        if df is not None:
            print(df.to_string(index=False))
        
        # 保存完整结果
        results_file = f'rsa_results_{self.embedding_type}{self.output_suffix}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n💾 完整结果已保存到: {results_file}")
        
        print(f"\n✅ RSA分析完成！")
        print(f"📁 结果保存在: {self.output_dir}/ 目录")
        print(f"📊 Embedding类型: {self.embedding_descriptions[self.embedding_type]}")
        print(f"🔧 归一化方法: {'z-score per neuron' if self.use_zscore else '原始数据'}")
        
        return results, df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='猴子神经元RSA分析 - 支持多种embedding类型选择')
    parser.add_argument('--embedding_type', type=str, default='image',
                       choices=['image', 'word_average', 'noun', 'verb'],
                       help='选择embedding类型: image, word_average, noun, verb')
    parser.add_argument('--use_zscore', action='store_true', default=True,
                       help='是否使用z-score归一化的猴子数据')
    parser.add_argument('--no_zscore', action='store_true', default=False,
                       help='使用原始数据（不使用z-score归一化）')
    
    args = parser.parse_args()
    
    # 处理z-score参数
    use_zscore = args.use_zscore and not args.no_zscore
    
    print("=" * 80)
    print("🐒 猴子神经元RSA分析 - 多Embedding类型支持")
    print("=" * 80)
    
    # 创建分析器
    analyzer = MonkeyRSAAnalyzer(embedding_type=args.embedding_type, use_zscore=use_zscore)
    
    # 运行分析
    results, summary_df = analyzer.run_analysis()
    
    print("\n" + "=" * 80)
    print("🎉 分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
