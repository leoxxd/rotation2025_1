#!/usr/bin/env python3
"""
对比RSA分析方法：原始版本 vs z-score归一化版本
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RSAComparison:
    def __init__(self):
        # 文件路径
        self.original_file = "rsa_results_simple/all_rsa_results.mat"
        self.zscore_file = "rsa_results_zscore/all_rsa_results_zscore.mat"
        
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
    
    def load_results(self):
        """加载两种方法的结果"""
        print("📁 加载RSA分析结果...")
        
        # 加载原始版本结果
        if os.path.exists(self.original_file):
            original_data = loadmat(self.original_file)
            self.original_results = original_data['rsa_results']
            print(f"✅ 原始版本结果加载成功: {len(self.original_results)} 个结果")
        else:
            print(f"❌ 原始版本结果文件不存在: {self.original_file}")
            self.original_results = None
        
        # 加载z-score版本结果
        if os.path.exists(self.zscore_file):
            zscore_data = loadmat(self.zscore_file)
            self.zscore_results = zscore_data['rsa_results']
            print(f"✅ z-score版本结果加载成功: {len(self.zscore_results)} 个结果")
        else:
            print(f"❌ z-score版本结果文件不存在: {self.zscore_file}")
            self.zscore_results = None
        
        return self.original_results is not None and self.zscore_results is not None
    
    def create_comparison_dataframe(self):
        """创建对比数据框"""
        print("🔄 创建对比数据框...")
        
        comparison_data = []
        
        # 处理原始版本结果
        if self.original_results is not None:
            for result in self.original_results:
                comparison_data.append({
                    'subject': result['subject'][0],
                    'roi_key': result['roi_key'][0],
                    'hemisphere': result['hemisphere'][0],
                    'roi_label': result['roi_label'][0, 0],
                    'roi_name': result['roi_name'][0],
                    'method': 'original',
                    'raw_correlation': result['raw_correlation'][0, 0],
                    'corrected_correlation': result['corrected_correlation'][0, 0],
                    'noise_ceiling': result['noise_ceiling'][0, 0],
                    'traditional_correlation': result['traditional_correlation'][0, 0],
                    'traditional_p_value': result['traditional_p_value'][0, 0]
                })
        
        # 处理z-score版本结果
        if self.zscore_results is not None:
            for result in self.zscore_results:
                comparison_data.append({
                    'subject': result['subject'][0],
                    'roi_key': result['roi_key'][0],
                    'hemisphere': result['hemisphere'][0],
                    'roi_label': result['roi_label'][0, 0],
                    'roi_name': result['roi_name'][0],
                    'method': 'zscore',
                    'raw_correlation': result['raw_correlation'][0, 0],
                    'corrected_correlation': result['corrected_correlation'][0, 0],
                    'noise_ceiling': result['noise_ceiling'][0, 0],
                    'traditional_correlation': result['traditional_correlation'][0, 0],
                    'traditional_p_value': result['traditional_p_value'][0, 0]
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        print(f"✅ 对比数据框创建完成: {len(self.comparison_df)} 行数据")
        
        return self.comparison_df
    
    def create_comparison_plots(self, save_dir="rsa_comparison"):
        """创建对比图表"""
        print(f"📊 创建对比图表...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 原始相关性对比
        plt.figure(figsize=(15, 10))
        
        # 按ROI分组对比
        plt.subplot(2, 2, 1)
        roi_comparison = self.comparison_df.groupby(['roi_name', 'method'])['raw_correlation'].mean().unstack()
        roi_comparison.plot(kind='bar', ax=plt.gca(), alpha=0.7)
        plt.title('原始相关性对比（按ROI）')
        plt.xlabel('ROI')
        plt.ylabel('原始相关性')
        plt.legend(['原始方法', 'z-score归一化'])
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 按被试分组对比
        plt.subplot(2, 2, 2)
        subject_comparison = self.comparison_df.groupby(['subject', 'method'])['raw_correlation'].mean().unstack()
        subject_comparison.plot(kind='bar', ax=plt.gca(), alpha=0.7)
        plt.title('原始相关性对比（按被试）')
        plt.xlabel('被试')
        plt.ylabel('原始相关性')
        plt.legend(['原始方法', 'z-score归一化'])
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. 校正后相关性对比
        plt.subplot(2, 2, 3)
        roi_corrected_comparison = self.comparison_df.groupby(['roi_name', 'method'])['corrected_correlation'].mean().unstack()
        roi_corrected_comparison.plot(kind='bar', ax=plt.gca(), alpha=0.7)
        plt.title('校正后相关性对比（按ROI）')
        plt.xlabel('ROI')
        plt.ylabel('校正后相关性')
        plt.legend(['原始方法', 'z-score归一化'])
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 按被试分组对比
        plt.subplot(2, 2, 4)
        subject_corrected_comparison = self.comparison_df.groupby(['subject', 'method'])['corrected_correlation'].mean().unstack()
        subject_corrected_comparison.plot(kind='bar', ax=plt.gca(), alpha=0.7)
        plt.title('校正后相关性对比（按被试）')
        plt.xlabel('被试')
        plt.ylabel('校正后相关性')
        plt.legend(['原始方法', 'z-score归一化'])
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rsa_methods_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 散点图对比
        plt.figure(figsize=(12, 5))
        
        # 原始相关性散点图
        plt.subplot(1, 2, 1)
        original_data = self.comparison_df[self.comparison_df['method'] == 'original']
        zscore_data = self.comparison_df[self.comparison_df['method'] == 'zscore']
        
        plt.scatter(original_data['raw_correlation'], zscore_data['raw_correlation'], 
                   alpha=0.7, s=60, c='blue', label='原始 vs z-score')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        plt.xlabel('原始方法 - 原始相关性')
        plt.ylabel('z-score方法 - 原始相关性')
        plt.title('原始相关性对比散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 校正后相关性散点图
        plt.subplot(1, 2, 2)
        plt.scatter(original_data['corrected_correlation'], zscore_data['corrected_correlation'], 
                   alpha=0.7, s=60, c='green', label='原始 vs z-score')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        plt.xlabel('原始方法 - 校正后相关性')
        plt.ylabel('z-score方法 - 校正后相关性')
        plt.title('校正后相关性对比散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rsa_correlation_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 热力图对比
        plt.figure(figsize=(15, 6))
        
        # 原始方法热力图
        plt.subplot(1, 2, 1)
        original_pivot = original_data.pivot(index='subject', columns='roi_key', values='corrected_correlation')
        sns.heatmap(original_pivot, annot=True, cmap='viridis', fmt='.3f', 
                   cbar_kws={'label': '校正后相关性'})
        plt.title('原始方法 - 校正后相关性热力图')
        plt.xlabel('ROI')
        plt.ylabel('被试')
        
        # z-score方法热力图
        plt.subplot(1, 2, 2)
        zscore_pivot = zscore_data.pivot(index='subject', columns='roi_key', values='corrected_correlation')
        sns.heatmap(zscore_pivot, annot=True, cmap='viridis', fmt='.3f', 
                   cbar_kws={'label': '校正后相关性'})
        plt.title('z-score方法 - 校正后相关性热力图')
        plt.xlabel('ROI')
        plt.ylabel('被试')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rsa_heatmap_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 对比图表已保存: {save_dir}/")
        print(f"  - rsa_methods_comparison.png: 方法对比")
        print(f"  - rsa_correlation_scatter.png: 相关性散点图")
        print(f"  - rsa_heatmap_comparison.png: 热力图对比")
    
    def generate_comparison_report(self, save_dir="rsa_comparison"):
        """生成对比报告"""
        print(f"📝 生成对比报告...")
        
        # 计算统计信息
        original_data = self.comparison_df[self.comparison_df['method'] == 'original']
        zscore_data = self.comparison_df[self.comparison_df['method'] == 'zscore']
        
        # 总体统计
        original_raw_mean = original_data['raw_correlation'].mean()
        original_raw_std = original_data['raw_correlation'].std()
        original_corrected_mean = original_data['corrected_correlation'].mean()
        original_corrected_std = original_data['corrected_correlation'].std()
        
        zscore_raw_mean = zscore_data['raw_correlation'].mean()
        zscore_raw_std = zscore_data['raw_correlation'].std()
        zscore_corrected_mean = zscore_data['corrected_correlation'].mean()
        zscore_corrected_std = zscore_data['corrected_correlation'].std()
        
        # 差异分析
        raw_diff = zscore_raw_mean - original_raw_mean
        corrected_diff = zscore_corrected_mean - original_corrected_mean
        
        # 相关性分析
        raw_correlation = np.corrcoef(original_data['raw_correlation'], zscore_data['raw_correlation'])[0, 1]
        corrected_correlation = np.corrcoef(original_data['corrected_correlation'], zscore_data['corrected_correlation'])[0, 1]
        
        # 保存报告
        with open(os.path.join(save_dir, 'comparison_report.txt'), 'w', encoding='utf-8') as f:
            f.write("RSA分析方法对比报告\n")
            f.write("="*50 + "\n\n")
            
            f.write("1. 总体统计对比\n")
            f.write("-"*30 + "\n")
            f.write(f"原始方法:\n")
            f.write(f"  原始相关性: {original_raw_mean:.3f} ± {original_raw_std:.3f}\n")
            f.write(f"  校正后相关性: {original_corrected_mean:.3f} ± {original_corrected_std:.3f}\n\n")
            
            f.write(f"z-score归一化方法:\n")
            f.write(f"  原始相关性: {zscore_raw_mean:.3f} ± {zscore_raw_std:.3f}\n")
            f.write(f"  校正后相关性: {zscore_corrected_mean:.3f} ± {zscore_corrected_std:.3f}\n\n")
            
            f.write("2. 差异分析\n")
            f.write("-"*30 + "\n")
            f.write(f"原始相关性差异: {raw_diff:.3f} ({raw_diff/original_raw_mean*100:.1f}%)\n")
            f.write(f"校正后相关性差异: {corrected_diff:.3f} ({corrected_diff/original_corrected_mean*100:.1f}%)\n\n")
            
            f.write("3. 相关性分析\n")
            f.write("-"*30 + "\n")
            f.write(f"原始相关性相关系数: {raw_correlation:.3f}\n")
            f.write(f"校正后相关性相关系数: {corrected_correlation:.3f}\n\n")
            
            f.write("4. 按ROI分组统计\n")
            f.write("-"*30 + "\n")
            for roi_name in self.comparison_df['roi_name'].unique():
                roi_original = original_data[original_data['roi_name'] == roi_name]
                roi_zscore = zscore_data[zscore_data['roi_name'] == roi_name]
                
                if len(roi_original) > 0 and len(roi_zscore) > 0:
                    f.write(f"{roi_name}:\n")
                    f.write(f"  原始方法 - 原始: {roi_original['raw_correlation'].mean():.3f} ± {roi_original['raw_correlation'].std():.3f}\n")
                    f.write(f"  原始方法 - 校正: {roi_original['corrected_correlation'].mean():.3f} ± {roi_original['corrected_correlation'].std():.3f}\n")
                    f.write(f"  z-score方法 - 原始: {roi_zscore['raw_correlation'].mean():.3f} ± {roi_zscore['raw_correlation'].std():.3f}\n")
                    f.write(f"  z-score方法 - 校正: {roi_zscore['corrected_correlation'].mean():.3f} ± {roi_zscore['corrected_correlation'].std():.3f}\n")
                    f.write(f"  原始相关性差异: {roi_zscore['raw_correlation'].mean() - roi_original['raw_correlation'].mean():.3f}\n")
                    f.write(f"  校正后相关性差异: {roi_zscore['corrected_correlation'].mean() - roi_original['corrected_correlation'].mean():.3f}\n\n")
            
            f.write("5. 按被试分组统计\n")
            f.write("-"*30 + "\n")
            for subject in self.comparison_df['subject'].unique():
                subj_original = original_data[original_data['subject'] == subject]
                subj_zscore = zscore_data[zscore_data['subject'] == subject]
                
                if len(subj_original) > 0 and len(subj_zscore) > 0:
                    f.write(f"{subject}:\n")
                    f.write(f"  原始方法 - 原始: {subj_original['raw_correlation'].mean():.3f} ± {subj_original['raw_correlation'].std():.3f}\n")
                    f.write(f"  原始方法 - 校正: {subj_original['corrected_correlation'].mean():.3f} ± {subj_original['corrected_correlation'].std():.3f}\n")
                    f.write(f"  z-score方法 - 原始: {subj_zscore['raw_correlation'].mean():.3f} ± {subj_zscore['raw_correlation'].std():.3f}\n")
                    f.write(f"  z-score方法 - 校正: {subj_zscore['corrected_correlation'].mean():.3f} ± {subj_zscore['corrected_correlation'].std():.3f}\n")
                    f.write(f"  原始相关性差异: {subj_zscore['raw_correlation'].mean() - subj_original['raw_correlation'].mean():.3f}\n")
                    f.write(f"  校正后相关性差异: {subj_zscore['corrected_correlation'].mean() - subj_original['corrected_correlation'].mean():.3f}\n\n")
            
            f.write("6. 结论\n")
            f.write("-"*30 + "\n")
            if abs(raw_diff) < 0.01:
                f.write("- 原始相关性差异很小，两种方法结果基本一致\n")
            elif raw_diff > 0:
                f.write("- z-score归一化方法提高了原始相关性\n")
            else:
                f.write("- z-score归一化方法降低了原始相关性\n")
            
            if abs(corrected_diff) < 0.01:
                f.write("- 校正后相关性差异很小，两种方法结果基本一致\n")
            elif corrected_diff > 0:
                f.write("- z-score归一化方法提高了校正后相关性\n")
            else:
                f.write("- z-score归一化方法降低了校正后相关性\n")
            
            f.write(f"- 两种方法的相关性很高（原始: {raw_correlation:.3f}, 校正: {corrected_correlation:.3f}）\n")
            f.write("- z-score归一化主要影响个体差异，对相对模式影响较小\n")
        
        # 保存对比数据
        self.comparison_df.to_csv(os.path.join(save_dir, 'comparison_data.csv'), index=False, encoding='utf-8')
        
        print(f"✅ 对比报告已保存: {save_dir}/comparison_report.txt")
        print(f"✅ 对比数据已保存: {save_dir}/comparison_data.csv")
    
    def run_comparison(self):
        """运行完整对比分析"""
        print("🔍 开始RSA方法对比分析...")
        print("="*60)
        
        # 1. 加载结果
        if not self.load_results():
            print("❌ 无法加载结果，请先运行两种方法的分析")
            return None
        
        # 2. 创建对比数据框
        self.create_comparison_dataframe()
        
        # 3. 创建对比图表
        self.create_comparison_plots()
        
        # 4. 生成对比报告
        self.generate_comparison_report()
        
        print("\n" + "="*60)
        print("🎉 RSA方法对比分析完成!")
        print("="*60)
        
        # 显示简要统计
        original_data = self.comparison_df[self.comparison_df['method'] == 'original']
        zscore_data = self.comparison_df[self.comparison_df['method'] == 'zscore']
        
        print(f"\n📊 简要统计:")
        print(f"  原始方法 - 原始相关性: {original_data['raw_correlation'].mean():.3f} ± {original_data['raw_correlation'].std():.3f}")
        print(f"  原始方法 - 校正后相关性: {original_data['corrected_correlation'].mean():.3f} ± {original_data['corrected_correlation'].std():.3f}")
        print(f"  z-score方法 - 原始相关性: {zscore_data['raw_correlation'].mean():.3f} ± {zscore_data['raw_correlation'].std():.3f}")
        print(f"  z-score方法 - 校正后相关性: {zscore_data['corrected_correlation'].mean():.3f} ± {zscore_data['corrected_correlation'].std():.3f}")
        
        raw_diff = zscore_data['raw_correlation'].mean() - original_data['raw_correlation'].mean()
        corrected_diff = zscore_data['corrected_correlation'].mean() - original_data['corrected_correlation'].mean()
        print(f"  原始相关性差异: {raw_diff:.3f}")
        print(f"  校正后相关性差异: {corrected_diff:.3f}")
        
        print(f"\n📁 生成的文件:")
        print(f"  - rsa_comparison/comparison_report.txt")
        print(f"  - rsa_comparison/comparison_data.csv")
        print(f"  - rsa_comparison/rsa_methods_comparison.png")
        print(f"  - rsa_comparison/rsa_correlation_scatter.png")
        print(f"  - rsa_comparison/rsa_heatmap_comparison.png")


def main():
    """主函数"""
    # 创建对比分析器
    comparator = RSAComparison()
    
    # 运行对比分析
    comparator.run_comparison()


if __name__ == "__main__":
    main()

