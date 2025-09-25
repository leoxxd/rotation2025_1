#!/usr/bin/env python3
"""
RSA分析器 - 集成不同LLM embedding选择的z-score归一化版本
支持image, word_average, noun, verb四种embedding类型
"""

import os
import numpy as np
import json
import argparse
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist

class RSAAnalyzerWithEmbeddingChoice:
    def __init__(self, embedding_type='image'):
        """
        初始化RSA分析器
        
        Args:
            embedding_type: embedding类型 ('image', 'word_average', 'noun', 'verb')
        """
        self.embedding_type = embedding_type
        
        # 文件路径配置
        self.roi_file = "roi_rdm_results/all_subjects_roi_rdms.mat"
        
        # 根据embedding类型设置文件路径
        self.embedding_paths = {
            'image': "../captions/embeddings_output/image_embeddings.npy",
            'word_average': "../captions/embeddings_output/word_average_embeddings.npy", 
            'noun': "../captions/embeddings_output/noun_embeddings.npy",
            'verb': "../captions/embeddings_output/verb_embeddings.npy"
        }
        
        self.embedding_file = self.embedding_paths[embedding_type]
        
        # ROI标签映射
        self.roi_labels = {
            1: 'early',
            2: 'midventral', 
            3: 'midlateral',
            4: 'midparietal',
            5: 'ventral',
            6: 'lateral',
            7: 'parietal'
        }
        
        # 被试列表
        self.subjects = ['s1', 's2', 's5', 's7']
        
        # 输出目录设置
        self.output_suffix = f'_{embedding_type}_zscore'
        self.save_dir = f"rsa_results{self.output_suffix}"
        
        print(f"🎯 RSA分析器初始化完成")
        print(f"   Embedding类型: {embedding_type}")
        print(f"   Embedding文件: {self.embedding_file}")
        print(f"   ROI文件: {self.roi_file}")
        print(f"   输出目录: {self.save_dir}")
    
    def corr_rdms(self, X, Y):
        """原始项目的相关性计算函数"""
        X = X - X.mean(axis=1, keepdims=True)
        X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
        Y = Y - Y.mean(axis=1, keepdims=True)
        Y /= np.sqrt(np.einsum("ij,ij->i", Y, Y))[:, None]
        return np.einsum("ik,jk", X, Y)
    
    def zscore_normalize_fmri_data(self, roi_data):
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
    
    def corr_rdms(self, X, Y):
        """原始项目的相关性计算函数"""
        X = X - X.mean(axis=1, keepdims=True)
        X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
        Y = Y - Y.mean(axis=1, keepdims=True)
        Y /= np.sqrt(np.einsum("ij,ij->i", Y, Y))[:, None]
        return np.einsum("ik,jk", X, Y)
    
    def load_roi_rdms(self):
        """加载ROI RDM数据"""
        print(f"📁 加载ROI数据: {self.roi_file}")
        
        if not os.path.exists(self.roi_file):
            print(f"❌ ROI数据文件不存在: {self.roi_file}")
            return None
        
        roi_data = loadmat(self.roi_file)
        print(f"✅ ROI数据加载成功")
        
        return roi_data
    
    def load_embedding_rdm(self):
        """加载embedding RDM数据"""
        print(f"📁 加载embedding数据: {self.embedding_file}")
        
        if not os.path.exists(self.embedding_file):
            print(f"❌ embedding数据文件不存在: {self.embedding_file}")
            return None
        
        # 加载embedding数据
        embeddings = np.load(self.embedding_file)
        print(f"✅ embedding数据加载成功，形状: {embeddings.shape}")
        
        # 计算embedding RDM
        print("🔄 计算embedding RDM...")
        embedding_rdm = pdist(embeddings, metric='correlation')
        print(f"✅ embedding RDM计算完成，长度: {len(embedding_rdm)}")
        
        return embedding_rdm
    
    def preprocess_fmri_data(self, roi_data):
        """预处理fMRI数据，一次性计算所有z-score归一化的RDM"""
        # 检查缓存文件是否存在
        cache_file = "roi_rdm_results/zscore_processed_data.mat"
        
        if os.path.exists(cache_file):
            print(f"📁 发现缓存文件，加载z-score归一化数据: {cache_file}")
            try:
                cache_data = loadmat(cache_file)
                processed_data = {}
                
                # 检查缓存文件格式
                if 's1' in cache_data and isinstance(cache_data['s1'], dict):
                    # 新格式：直接是字典
                    print("  使用新格式缓存文件...")
                    for subject in self.subjects:
                        if subject in cache_data:
                            subject_rdms = cache_data[subject]
                            subject_processed = {}
                            
                            # 需要重新加载原始数据来获取ROI信息
                            roi_data = self.load_roi_rdms()
                            if roi_data and subject in roi_data:
                                subject_data = roi_data[subject][0, 0]
                                roi_keys = [f'lh_{i}' for i in range(1, 8)] + [f'rh_{i}' for i in range(1, 8)]
                                
                                for roi_key in roi_keys:
                                    if roi_key in subject_data.dtype.names and roi_key in subject_rdms:
                                        roi_info = subject_data[roi_key][0, 0]
                                        subject_processed[roi_key] = {
                                            'roi_label': roi_info['roi_label'][0, 0],
                                            'roi_name': roi_info['roi_name'][0],
                                            'hemisphere': roi_info['hemisphere'][0],
                                            'n_voxels': roi_info['n_voxels'][0, 0],
                                            'n_images': roi_info['n_images'][0, 0],
                                            'roi_data_normalized': None,  # 缓存中不保存原始数据
                                            'rdm': subject_rdms[roi_key]
                                        }
                            
                            processed_data[subject] = subject_processed
                else:
                    # 结构化数组格式：每个ROI是一个字段，只包含RDM数据
                    print("  使用结构化数组缓存文件...")
                    for subject in self.subjects:
                        if subject in cache_data:
                            subject_data = cache_data[subject][0, 0]
                            subject_processed = {}
                            
                            # 需要重新加载原始数据来获取ROI信息
                            roi_data = self.load_roi_rdms()
                            if roi_data and subject in roi_data:
                                original_data = roi_data[subject][0, 0]
                                roi_keys = [f'lh_{i}' for i in range(1, 8)] + [f'rh_{i}' for i in range(1, 8)]
                                
                                for roi_key in roi_keys:
                                    if roi_key in subject_data.dtype.names and roi_key in original_data.dtype.names:
                                        # 从缓存获取RDM，需要解包额外的维度
                                        cached_rdm = subject_data[roi_key]
                                        if cached_rdm.shape == (1, 1):
                                            cached_rdm = cached_rdm[0, 0]
                                        if cached_rdm.shape == (1, 499500):
                                            cached_rdm = cached_rdm[0]
                                        # 从原始数据获取ROI信息
                                        roi_info = original_data[roi_key][0, 0]
                                        
                                        subject_processed[roi_key] = {
                                            'roi_label': roi_info['roi_label'][0, 0],
                                            'roi_name': roi_info['roi_name'][0],
                                            'hemisphere': roi_info['hemisphere'][0],
                                            'n_voxels': roi_info['n_voxels'][0, 0],
                                            'n_images': roi_info['n_images'][0, 0],
                                            'roi_data_normalized': None,  # 缓存中不保存原始数据
                                            'rdm': cached_rdm
                                        }
                            
                            processed_data[subject] = subject_processed
                
                print(f"✅ 缓存数据加载成功")
                return processed_data
                
            except Exception as e:
                print(f"⚠️ 缓存文件加载失败: {e}")
                print("  将重新计算z-score归一化数据...")
        
        print("\n🔄 预处理fMRI数据（z-score归一化 + RDM计算）...")
        print("  首次运行或缓存文件不存在，正在计算...")
        
        # 存储所有被试的z-score归一化数据和RDM
        processed_data = {}
        
        for subject in self.subjects:
            if subject not in roi_data:
                print(f"  ⚠️ 被试 {subject} 数据不存在")
                continue
                
            print(f"  处理被试 {subject}...")
            subject_data = roi_data[subject][0, 0]  # 获取结构化数组
            
            subject_processed = {}
            
            # 遍历所有ROI (lh_1 到 lh_7, rh_1 到 rh_7)
            roi_keys = [f'lh_{i}' for i in range(1, 8)] + [f'rh_{i}' for i in range(1, 8)]
            
            for roi_key in roi_keys:
                if roi_key in subject_data.dtype.names:
                    roi_info = subject_data[roi_key][0, 0]
                    
                    # 提取ROI信息
                    roi_label = roi_info['roi_label'][0, 0]
                    roi_name = roi_info['roi_name'][0]
                    hemisphere = roi_info['hemisphere'][0]
                    n_voxels = roi_info['n_voxels'][0, 0]
                    n_images = roi_info['n_images'][0, 0]
                    roi_data_raw = roi_info['roi_data']  # [n_voxels, n_images]
                    
                    # 对fMRI数据进行z-score归一化
                    roi_data_normalized = self.zscore_normalize_fmri_data(roi_data_raw)
                    
                    # 计算RDM
                    data_for_rdm = roi_data_normalized.T  # [n_images, n_roi_voxels]
                    rdm = pdist(data_for_rdm, metric='correlation')
                    
                    # 存储处理后的数据
                    subject_processed[roi_key] = {
                        'roi_label': roi_label,
                        'roi_name': roi_name,
                        'hemisphere': hemisphere,
                        'n_voxels': n_voxels,
                        'n_images': n_images,
                        'roi_data_normalized': roi_data_normalized,  # z-score归一化后的数据
                        'rdm': rdm  # 计算好的RDM
                    }
            
            processed_data[subject] = subject_processed
            print(f"    完成，ROI数量: {len(subject_processed)}")
        
        # 保存缓存文件
        print(f"\n💾 保存z-score归一化数据到缓存文件: {cache_file}")
        self.save_processed_data_cache(processed_data, cache_file)
        
        return processed_data
    
    def save_processed_data_cache(self, processed_data, cache_file):
        """保存预处理数据到缓存文件"""
        try:
            # 简化保存方法 - 只保存RDM数据，不保存原始fMRI数据
            cache_data = {}
            
            for subject in self.subjects:
                if subject in processed_data:
                    subject_data = processed_data[subject]
                    subject_rdms = {}
                    
                    # 只保存RDM数据
                    for roi_key, roi_info in subject_data.items():
                        subject_rdms[roi_key] = roi_info['rdm']
                    
                    cache_data[subject] = subject_rdms
            
            # 添加元数据
            cache_data['description'] = 'z-score normalized RDM cache'
            cache_data['normalization_method'] = 'zscore_per_voxel_across_images'
            cache_data['created_timestamp'] = str(np.datetime64('now'))
            
            # 保存到MAT文件
            savemat(cache_file, cache_data)
            print(f"✅ 缓存文件保存成功: {cache_file}")
            
        except Exception as e:
            print(f"⚠️ 缓存文件保存失败: {e}")
            print("  分析将继续进行，但下次运行将重新计算z-score数据")
    
    def compute_noise_ceilings(self, processed_data):
        """计算噪声天花板 - 基于预处理的数据"""
        print("\n🎯 计算噪声天花板...")
        print("  每个被试的噪声天花板 = 该被试与其他所有被试在该ROI上的RDM相关性的平均值")
        
        noise_ceilings = {}
        
        for subject in self.subjects:
            if subject not in processed_data:
                continue
                
            print(f"  计算被试 {subject} 的噪声天花板...")
            subject_noise_ceilings = {}
            
            for roi_key in processed_data[subject]:
                # 获取该被试在该ROI的RDM
                subject_rdm = processed_data[subject][roi_key]['rdm']
                
                # 获取其他被试的RDM
                other_rdms = []
                for other_subject in self.subjects:
                    if other_subject != subject and other_subject in processed_data:
                        if roi_key in processed_data[other_subject]:
                            other_rdm = processed_data[other_subject][roi_key]['rdm']
                            other_rdms.append(other_rdm)
                
                # 计算其他被试的平均RDM
                if len(other_rdms) > 0:
                    mean_other_rdm = np.mean(other_rdms, axis=0)
                    
                    # 计算噪声天花板：该被试RDM与其他被试平均RDM的相关性
                    noise_ceiling = self.corr_rdms(subject_rdm.reshape(1, -1), mean_other_rdm.reshape(1, -1))[0, 0]
                    subject_noise_ceilings[roi_key] = noise_ceiling
                    
                    roi_name = processed_data[subject][roi_key]['roi_name']
                    print(f"    {roi_key} ({roi_name}): {noise_ceiling:.4f} (n={len(other_rdms)})")
            
            noise_ceilings[subject] = subject_noise_ceilings
        
        return noise_ceilings
    
    def analyze_single_roi_subject(self, fmri_rdm, embedding_rdm, roi_key, subject):
        """分析单个被试单个ROI（使用预处理的数据）"""
        # 使用与原始文件相同的相关性计算方法
        correlation = self.corr_rdms(fmri_rdm.reshape(1, -1), embedding_rdm.reshape(1, -1))[0, 0]
        
        return {
            'roi_key': roi_key,
            'subject': subject,
            'correlation': correlation,
            'method': 'zscore_normalized_1000_images',
            'normalization': 'zscore_per_voxel_across_images'
        }
    
    def run_analysis(self):
        """运行RSA分析（使用z-score归一化）"""
        print("🚀 开始RSA分析（z-score归一化版本）...")
        print(f"   Embedding类型: {self.embedding_type}")
        
        # 1. 加载数据
        roi_data = self.load_roi_rdms()
        if roi_data is None:
            return
        
        embedding_rdm = self.load_embedding_rdm()
        if embedding_rdm is None:
            return
        
        # 2. 预处理fMRI数据（z-score归一化 + RDM计算）
        processed_data = self.preprocess_fmri_data(roi_data)
        
        # 3. 计算噪声天花板
        noise_ceilings = self.compute_noise_ceilings(processed_data)
        
        # 4. 进行RSA分析
        print(f"\n🔄 开始RSA分析...")
        rsa_results = []
        
        for subject in self.subjects:
            if subject not in processed_data:
                print(f"  ⚠️ 被试 {subject} 数据不存在")
                continue
                
            print(f"  分析被试 {subject}...")
            
            # 遍历所有ROI
            for roi_key in processed_data[subject]:
                roi_info = processed_data[subject][roi_key]
                
                # 分析该被试该ROI（使用预处理的数据）
                result = self.analyze_single_roi_subject(roi_info['rdm'], embedding_rdm, roi_key, subject)
                
                # 添加噪声天花板信息（按被试和ROI查找）
                if subject in noise_ceilings and roi_key in noise_ceilings[subject]:
                    result['noise_ceiling'] = noise_ceilings[subject][roi_key]
                    result['corrected_correlation'] = result['correlation'] / noise_ceilings[subject][roi_key]
                else:
                    result['noise_ceiling'] = np.nan
                    result['corrected_correlation'] = np.nan
                
                # 添加ROI信息
                result.update({
                    'roi_label': roi_info['roi_label'],
                    'roi_name': roi_info['roi_name'],
                    'hemisphere': roi_info['hemisphere'],
                    'n_voxels': roi_info['n_voxels'],
                    'n_images': roi_info['n_images'],
                    'embedding_type': self.embedding_type
                })
                
                rsa_results.append(result)
        
        print(f"✅ RSA分析完成，共 {len(rsa_results)} 个结果")
        
        # 5. 保存结果
        self.save_results(rsa_results, noise_ceilings)
        
        # 6. 创建可视化图表
        self.create_visualizations(rsa_results, noise_ceilings)
        
        return rsa_results, noise_ceilings
    
    def save_results(self, rsa_results, noise_ceilings):
        """保存结果"""
        print(f"\n💾 保存结果到 {self.save_dir}...")
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存所有结果
        savemat(os.path.join(self.save_dir, f'all_rsa_results{self.output_suffix}.mat'), {'rsa_results': rsa_results})
        print(f"✅ 结果已保存: {self.save_dir}/all_rsa_results{self.output_suffix}.mat")
        
        # 保存CSV格式
        try:
            import pandas as pd
            df = pd.DataFrame(rsa_results)
            csv_file = os.path.join(self.save_dir, f'all_rsa_results{self.output_suffix}.csv')
            
            # 如果文件已存在且被占用，先删除
            if os.path.exists(csv_file):
                try:
                    os.remove(csv_file)
                except PermissionError:
                    # 如果无法删除，尝试重命名
                    backup_file = csv_file + '.backup'
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                    os.rename(csv_file, backup_file)
            
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"✅ 结果已保存: {self.save_dir}/all_rsa_results{self.output_suffix}.csv")
        except Exception as e:
            print(f"⚠️ CSV文件保存失败: {e}")
            print("  将只保存MAT格式文件")
        
        # 按ROI分类保存（分左右脑，按照rsa_results_zscore的结构）
        print("📁 按ROI分类保存结果（分左右脑）...")
        roi_groups = df.groupby('roi_key')
        for roi_key, roi_df in roi_groups:
            roi_dir = os.path.join(self.save_dir, f'roi_{roi_key}')
            os.makedirs(roi_dir, exist_ok=True)
            
            savemat(os.path.join(roi_dir, f'{roi_key}_rsa_results{self.output_suffix}.mat'), {'rsa_results': roi_df.to_dict('records')})
            roi_df.to_csv(os.path.join(roi_dir, f'{roi_key}_rsa_results{self.output_suffix}.csv'), index=False, encoding='utf-8')
            
            # 创建ROI分析图
            self.create_roi_analysis_plot(roi_key, roi_df.to_dict('records'))
        
        # 按被试分类保存
        print("📁 按被试分类保存结果...")
        subject_groups = df.groupby('subject')
        for subject, subject_df in subject_groups:
            subject_dir = os.path.join(self.save_dir, f'subject_{subject}')
            os.makedirs(subject_dir, exist_ok=True)
            
            savemat(os.path.join(subject_dir, f'{subject}_rsa_results{self.output_suffix}.mat'), {'rsa_results': subject_df.to_dict('records')})
            subject_df.to_csv(os.path.join(subject_dir, f'{subject}_rsa_results{self.output_suffix}.csv'), index=False, encoding='utf-8')
            
            # 创建被试分析图
            self.create_subject_analysis_plot(subject, subject_df.to_dict('records'))
        
        # 生成分析报告
        self.generate_analysis_report(rsa_results, noise_ceilings)
        
        print(f"✅ 所有结果已保存到: {self.save_dir}")
    
    def generate_analysis_report(self, rsa_results, noise_ceilings):
        """生成分析报告"""
        print("📊 生成分析报告...")
        
        import pandas as pd
        df = pd.DataFrame(rsa_results)
        
        with open(os.path.join(self.save_dir, f'analysis_report{self.output_suffix}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"RSA分析结果报告（{self.embedding_type} embedding + z-score归一化）\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("分析配置:\n")
            f.write(f"- Embedding类型: {self.embedding_type}\n")
            f.write(f"- 归一化方法: z-score（每个体素在1000张图片上归一化）\n")
            f.write(f"- 分析方法: 直接计算，无采样\n")
            f.write(f"- 被试数量: {df['subject'].nunique()}\n")
            f.write(f"- ROI数量: {df['roi_key'].nunique()}\n")
            f.write(f"- 总分析数: {len(df)}\n\n")
            
            f.write("相关性统计:\n")
            f.write(f"- 平均原始相关性: {df['correlation'].mean():.4f} ± {df['correlation'].std():.4f}\n")
            f.write(f"- 平均校正后相关性: {df['corrected_correlation'].mean():.4f} ± {df['corrected_correlation'].std():.4f}\n")
            f.write(f"- 平均噪声天花板: {df['noise_ceiling'].mean():.4f} ± {df['noise_ceiling'].std():.4f}\n\n")
            
            f.write("按ROI统计:\n")
            roi_stats = df.groupby('roi_key').agg({
                'correlation': ['mean', 'std', 'count'],
                'corrected_correlation': ['mean', 'std'],
                'noise_ceiling': ['mean', 'std']
            }).round(4)
            f.write(roi_stats.to_string())
            f.write("\n\n")
            
            f.write("按被试统计:\n")
            subject_stats = df.groupby('subject').agg({
                'correlation': ['mean', 'std', 'count'],
                'corrected_correlation': ['mean', 'std'],
                'noise_ceiling': ['mean', 'std']
            }).round(4)
            f.write(subject_stats.to_string())
            f.write("\n\n")
            
            f.write("噪声天花板详情:\n")
            # 噪声天花板现在是按被试组织的，需要重新组织显示
            roi_summary = {}
            for subject, subject_noise in noise_ceilings.items():
                for roi_key, noise_value in subject_noise.items():
                    if roi_key not in roi_summary:
                        roi_summary[roi_key] = []
                    roi_summary[roi_key].append(noise_value)
            
            for roi_key, noise_values in roi_summary.items():
                avg_noise = np.mean(noise_values)
                f.write(f"- {roi_key}: {avg_noise:.4f} (n={len(noise_values)})\n")
        
        print(f"✅ 分析报告已保存: {self.save_dir}/analysis_report{self.output_suffix}.txt")
    
    def create_visualizations(self, rsa_results, noise_ceilings):
        """创建可视化图表"""
        print("\n🎨 创建可视化图表...")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            plots_dir = os.path.join(self.save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 10
            
            # 转换为DataFrame
            df = pd.DataFrame(rsa_results)
            
            # 1. 原始相关性 vs 校正后相关性散点图
            plt.figure(figsize=(10, 8))
            plt.scatter(df['correlation'], df['corrected_correlation'], alpha=0.7, s=60)
            plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
            plt.xlabel('原始相关性')
            plt.ylabel('校正后相关性')
            plt.title(f'噪声天花板校正效果（{self.embedding_type} embedding + z-score归一化）')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'correlation_comparison{self.output_suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 按ROI分组的箱线图
            plt.figure(figsize=(15, 8))
            
            # 准备数据
            roi_data = []
            for _, row in df.iterrows():
                roi_name = self.roi_labels[int(row['roi_key'].split('_')[1])]
                roi_data.append({
                    'ROI': roi_name,
                    '原始相关性': row['correlation'],
                    '校正后相关性': row['corrected_correlation'],
                    '噪声天花板': row['noise_ceiling']
                })
            
            roi_df = pd.DataFrame(roi_data)
            
            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 原始相关性
            sns.boxplot(data=roi_df, x='ROI', y='原始相关性', ax=axes[0])
            axes[0].set_title(f'原始相关性（{self.embedding_type} embedding + z-score归一化）')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 校正后相关性
            sns.boxplot(data=roi_df, x='ROI', y='校正后相关性', ax=axes[1])
            axes[1].set_title(f'校正后相关性（{self.embedding_type} embedding + z-score归一化）')
            axes[1].tick_params(axis='x', rotation=45)
            
            # 噪声天花板
            sns.boxplot(data=roi_df, x='ROI', y='噪声天花板', ax=axes[2])
            axes[2].set_title(f'噪声天花板（{self.embedding_type} embedding + z-score归一化）')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'roi_comparison{self.output_suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 按被试分组的箱线图
            plt.figure(figsize=(15, 8))
            
            # 准备数据
            subject_data = []
            for _, row in df.iterrows():
                subject_data.append({
                    '被试': row['subject'],
                    '原始相关性': row['correlation'],
                    '校正后相关性': row['corrected_correlation'],
                    '噪声天花板': row['noise_ceiling']
                })
            
            subject_df = pd.DataFrame(subject_data)
            
            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 原始相关性
            sns.boxplot(data=subject_df, x='被试', y='原始相关性', ax=axes[0])
            axes[0].set_title(f'原始相关性（{self.embedding_type} embedding + z-score归一化）')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 校正后相关性
            sns.boxplot(data=subject_df, x='被试', y='校正后相关性', ax=axes[1])
            axes[1].set_title(f'校正后相关性（{self.embedding_type} embedding + z-score归一化）')
            axes[1].tick_params(axis='x', rotation=45)
            
            # 噪声天花板
            sns.boxplot(data=subject_df, x='被试', y='噪声天花板', ax=axes[2])
            axes[2].set_title(f'噪声天花板（{self.embedding_type} embedding + z-score归一化）')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'subject_comparison{self.output_suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. 为每个被试单独绘制ROI比较图
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
                    raw_corrs.append(row['correlation'])
                    corrected_corrs.append(row['corrected_correlation'])
                    noise_ceilings.append(row['noise_ceiling'])
                
                # 创建子图
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 原始相关性
                axes[0, 0].bar(roi_names, raw_corrs, color='skyblue', alpha=0.7, edgecolor='black')
                axes[0, 0].set_title(f'{subject} - 原始相关性（{self.embedding_type} embedding + z-score归一化，1000张图片）')
                axes[0, 0].set_ylabel('相关性')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
                
                # 校正后相关性
                axes[0, 1].bar(roi_names, corrected_corrs, color='lightgreen', alpha=0.7, edgecolor='black')
                axes[0, 1].set_title(f'{subject} - 校正后相关性（{self.embedding_type} embedding + z-score归一化）')
                axes[0, 1].set_ylabel('相关性')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
                
                # 噪声天花板
                axes[1, 0].bar(roi_names, noise_ceilings, color='lightcoral', alpha=0.7, edgecolor='black')
                axes[1, 0].set_title(f'{subject} - 噪声天花板（{self.embedding_type} embedding + z-score归一化）')
                axes[1, 0].set_ylabel('噪声天花板')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
                
                # 校正效果
                improvements = [c - r for c, r in zip(corrected_corrs, raw_corrs)]
                axes[1, 1].bar(roi_names, improvements, color='gold', alpha=0.7, edgecolor='black')
                axes[1, 1].set_title(f'{subject} - 校正效果（{self.embedding_type} embedding + z-score归一化）')
                axes[1, 1].set_ylabel('相关性提升')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{subject}_roi_analysis{self.output_suffix}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"✅ 可视化图表已保存: {plots_dir}/")
            print(f"  - correlation_comparison{self.output_suffix}.png: 相关性比较")
            print(f"  - roi_comparison{self.output_suffix}.png: ROI比较")
            print(f"  - subject_comparison{self.output_suffix}.png: 被试比较")
            print(f"  - {len(subjects)}个被试的ROI分析图")
            
        except ImportError as e:
            print(f"⚠️ 可视化库未安装，跳过图表生成: {e}")
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
    
    def create_roi_analysis_plot(self, roi_key, roi_results):
        """为单个ROI创建分析图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 10
            
            roi_dir = os.path.join(self.save_dir, f'roi_{roi_key}')
            
            # 准备数据
            subjects = [r['subject'] for r in roi_results]
            raw_corrs = [r['correlation'] for r in roi_results]
            corrected_corrs = [r['corrected_correlation'] for r in roi_results]
            noise_ceilings = [r['noise_ceiling'] for r in roi_results]
            
            # 创建2x2子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 原始相关性
            bars = axes[0, 0].bar(subjects, raw_corrs, color='skyblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title(f'{roi_key} - 原始相关性（{self.embedding_type} embedding + z-score归一化）')
            axes[0, 0].set_ylabel('相关性')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, raw_corrs):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 校正后相关性
            bars = axes[0, 1].bar(subjects, corrected_corrs, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title(f'{roi_key} - 校正后相关性（{self.embedding_type} embedding + z-score归一化）')
            axes[0, 1].set_ylabel('相关性')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, corrected_corrs):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 噪声天花板
            bars = axes[1, 0].bar(subjects, noise_ceilings, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title(f'{roi_key} - 噪声天花板（{self.embedding_type} embedding + z-score归一化）')
            axes[1, 0].set_ylabel('噪声天花板')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, noise_ceilings):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 校正效果
            improvements = [c - r for c, r in zip(corrected_corrs, raw_corrs)]
            bars = axes[1, 1].bar(subjects, improvements, color='gold', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title(f'{roi_key} - 校正效果（{self.embedding_type} embedding + z-score归一化）')
            axes[1, 1].set_ylabel('相关性提升')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, improvements):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(roi_dir, f'{roi_key}_rsa_analysis{self.output_suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ 创建ROI {roi_key} 分析图失败: {e}")
    
    def create_subject_analysis_plot(self, subject, subject_results):
        """为单个被试创建分析图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 10
            
            subject_dir = os.path.join(self.save_dir, f'subject_{subject}')
            
            # 准备数据 - 按ROI分组，每个ROI包含左右脑数据
            roi_data = {}
            for result in subject_results:
                roi_label = int(result['roi_key'].split('_')[1])
                hemisphere = result['roi_key'].split('_')[0]
                roi_name = self.roi_labels[roi_label]
                
                if roi_name not in roi_data:
                    roi_data[roi_name] = {}
                
                roi_data[roi_name][hemisphere] = {
                    'correlation': result['correlation'],
                    'corrected_correlation': result['corrected_correlation'],
                    'noise_ceiling': result['noise_ceiling']
                }
            
            # 获取ROI名称和半球
            roi_names = sorted(roi_data.keys())
            hemispheres = ['lh', 'rh']
            
            # 创建2x2子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 原始相关性 - 按ROI分组显示左右脑
            x = np.arange(len(roi_names))
            width = 0.35
            
            lh_raw = [roi_data[roi]['lh']['correlation'] if 'lh' in roi_data[roi] else 0 for roi in roi_names]
            rh_raw = [roi_data[roi]['rh']['correlation'] if 'rh' in roi_data[roi] else 0 for roi in roi_names]
            
            bars1 = axes[0, 0].bar(x - width/2, lh_raw, width, label='左脑(lh)', color='skyblue', alpha=0.8, edgecolor='black')
            bars2 = axes[0, 0].bar(x + width/2, rh_raw, width, label='右脑(rh)', color='lightblue', alpha=0.8, edgecolor='black')
            
            axes[0, 0].set_title(f'{subject} - 原始相关性（{self.embedding_type} embedding + z-score归一化，1000张图片）')
            axes[0, 0].set_ylabel('相关性')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(roi_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars1, lh_raw):
                if value != 0:
                    height = bar.get_height()
                    if value >= 0:
                        # 正值：标签在柱子上方
                        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        # 负值：标签在柱子下方
                        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                                       f'{value:.3f}', ha='center', va='top', fontsize=8)
            
            for bar, value in zip(bars2, rh_raw):
                if value != 0:
                    height = bar.get_height()
                    if value >= 0:
                        # 正值：标签在柱子上方
                        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        # 负值：标签在柱子下方
                        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                                       f'{value:.3f}', ha='center', va='top', fontsize=8)
            
            # 2. 校正后相关性 - 按ROI分组显示左右脑
            lh_corrected = [roi_data[roi]['lh']['corrected_correlation'] if 'lh' in roi_data[roi] else 0 for roi in roi_names]
            rh_corrected = [roi_data[roi]['rh']['corrected_correlation'] if 'rh' in roi_data[roi] else 0 for roi in roi_names]
            
            bars1 = axes[0, 1].bar(x - width/2, lh_corrected, width, label='左脑(lh)', color='lightgreen', alpha=0.8, edgecolor='black')
            bars2 = axes[0, 1].bar(x + width/2, rh_corrected, width, label='右脑(rh)', color='darkgreen', alpha=0.8, edgecolor='black')
            
            axes[0, 1].set_title(f'{subject} - 校正后相关性（{self.embedding_type} embedding + z-score归一化）')
            axes[0, 1].set_ylabel('相关性')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(roi_names, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars1, lh_corrected):
                if value != 0:
                    height = bar.get_height()
                    if value >= 0:
                        # 正值：标签在柱子上方
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        # 负值：标签在柱子下方
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                                       f'{value:.3f}', ha='center', va='top', fontsize=8)
            
            for bar, value in zip(bars2, rh_corrected):
                if value != 0:
                    height = bar.get_height()
                    if value >= 0:
                        # 正值：标签在柱子上方
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        # 负值：标签在柱子下方
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                                       f'{value:.3f}', ha='center', va='top', fontsize=8)
            
            # 3. 噪声天花板 - 按ROI分组显示左右脑
            lh_noise = [roi_data[roi]['lh']['noise_ceiling'] if 'lh' in roi_data[roi] else 0 for roi in roi_names]
            rh_noise = [roi_data[roi]['rh']['noise_ceiling'] if 'rh' in roi_data[roi] else 0 for roi in roi_names]
            
            bars1 = axes[1, 0].bar(x - width/2, lh_noise, width, label='左脑(lh)', color='lightcoral', alpha=0.8, edgecolor='black')
            bars2 = axes[1, 0].bar(x + width/2, rh_noise, width, label='右脑(rh)', color='darkred', alpha=0.8, edgecolor='black')
            
            axes[1, 0].set_title(f'{subject} - 噪声天花板（{self.embedding_type} embedding + z-score归一化）')
            axes[1, 0].set_ylabel('噪声天花板')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(roi_names, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars1, lh_noise):
                if value != 0:
                    height = bar.get_height()
                    if value >= 0:
                        # 正值：标签在柱子上方
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        # 负值：标签在柱子下方
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                                       f'{value:.3f}', ha='center', va='top', fontsize=8)
            
            for bar, value in zip(bars2, rh_noise):
                if value != 0:
                    height = bar.get_height()
                    if value >= 0:
                        # 正值：标签在柱子上方
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        # 负值：标签在柱子下方
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                                       f'{value:.3f}', ha='center', va='top', fontsize=8)
            
            # 4. 校正效果 - 按ROI分组显示左右脑
            lh_improvement = [lh_corrected[i] - lh_raw[i] for i in range(len(roi_names))]
            rh_improvement = [rh_corrected[i] - rh_raw[i] for i in range(len(roi_names))]
            
            bars1 = axes[1, 1].bar(x - width/2, lh_improvement, width, label='左脑(lh)', color='gold', alpha=0.8, edgecolor='black')
            bars2 = axes[1, 1].bar(x + width/2, rh_improvement, width, label='右脑(rh)', color='orange', alpha=0.8, edgecolor='black')
            
            axes[1, 1].set_title(f'{subject} - 校正效果（{self.embedding_type} embedding + z-score归一化）')
            axes[1, 1].set_ylabel('相关性提升')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(roi_names, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars1, lh_improvement):
                if value != 0:
                    height = bar.get_height()
                    if value >= 0:
                        # 正值：标签在柱子上方
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        # 负值：标签在柱子下方
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                                       f'{value:.3f}', ha='center', va='top', fontsize=8)
            
            for bar, value in zip(bars2, rh_improvement):
                if value != 0:
                    height = bar.get_height()
                    if value >= 0:
                        # 正值：标签在柱子上方
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        # 负值：标签在柱子下方
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                                       f'{value:.3f}', ha='center', va='top', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(subject_dir, f'{subject}_rsa_analysis{self.output_suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ 创建被试 {subject} 分析图失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RSA分析器 - 支持不同LLM embedding选择')
    parser.add_argument('--embedding_type', type=str, default='image',
                       choices=['image', 'word_average', 'noun', 'verb'],
                       help='选择embedding类型')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"RSA分析器 - {args.embedding_type} embedding + z-score归一化")
    print("=" * 80)
    
    # 创建分析器
    analyzer = RSAAnalyzerWithEmbeddingChoice(embedding_type=args.embedding_type)
    
    # 运行分析
    try:
        rsa_results, noise_ceilings = analyzer.run_analysis()
        
        print("\n" + "=" * 80)
        print("🎉 RSA分析完成!")
        print("=" * 80)
        print(f"结果已保存到: {analyzer.save_dir}")
        print(f"主要文件:")
        print(f"  - all_rsa_results{analyzer.output_suffix}.mat")
        print(f"  - all_rsa_results{analyzer.output_suffix}.csv")
        print(f"  - analysis_report{analyzer.output_suffix}.txt")
        print(f"  - 按ROI分类的结果目录")
        print(f"  - 按被试分类的结果目录")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
