"""
比较猴子神经元信号RSA分析的原始版本和z-score归一化版本

运行原始rsa_with_noise_ceiling.py和z-score归一化版本，并比较结果
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

def load_original_results():
    """加载原始RSA分析结果"""
    try:
        with open('rsa_with_noise_ceiling_results.pkl', 'rb') as f:
            results = pickle.load(f)
        print("✅ 成功加载原始RSA分析结果")
        return results
    except FileNotFoundError:
        print("❌ 未找到原始RSA分析结果文件，请先运行 rsa_with_noise_ceiling.py")
        return None

def load_zscore_results():
    """加载z-score归一化RSA分析结果"""
    try:
        with open('rsa_with_zscore_results.pkl', 'rb') as f:
            results = pickle.load(f)
        print("✅ 成功加载z-score归一化RSA分析结果")
        return results
    except FileNotFoundError:
        print("❌ 未找到z-score归一化RSA分析结果文件，请先运行 rsa_with_zscore_normalization.py")
        return None

def compare_results(original_results, zscore_results):
    """比较两种方法的结果"""
    print("\n=== 比较原始版本和z-score归一化版本 ===")
    
    # 过滤有效结果
    original_valid = [r for r in original_results if r is not None]
    zscore_valid = [r for r in zscore_results if r is not None]
    
    print(f"原始版本有效结果: {len(original_valid)} 个arealabel")
    print(f"z-score版本有效结果: {len(zscore_valid)} 个arealabel")
    
    # 创建比较表格
    comparison_data = []
    
    for orig_result in original_valid:
        arealabel = orig_result['arealabel']
        
        # 找到对应的z-score结果
        zscore_result = None
        for zs_result in zscore_valid:
            if zs_result['arealabel'] == arealabel:
                zscore_result = zs_result
                break
        
        if zscore_result is not None:
            comparison_data.append({
                'Arealabel': arealabel,
                'N_Sessions': orig_result['n_sessions'],
                'Original_Mean_RSA': orig_result['mean_rsa'],
                'Original_Std_RSA': orig_result['std_rsa'],
                'Original_Corrected_RSA': orig_result['mean_corrected_rsa'],
                'Original_Std_Corrected_RSA': orig_result['std_corrected_rsa'],
                'Original_Noise_Ceiling': orig_result['mean_noise_ceiling'],
                'Zscore_Mean_RSA': zscore_result['mean_rsa'],
                'Zscore_Std_RSA': zscore_result['std_rsa'],
                'Zscore_Corrected_RSA': zscore_result['mean_corrected_rsa'],
                'Zscore_Std_Corrected_RSA': zscore_result['std_corrected_rsa'],
                'Zscore_Noise_Ceiling': zscore_result['mean_noise_ceiling'],
                'RSA_Difference': zscore_result['mean_rsa'] - orig_result['mean_rsa'],
                'Corrected_RSA_Difference': zscore_result['mean_corrected_rsa'] - orig_result['mean_corrected_rsa'],
                'Noise_Ceiling_Difference': zscore_result['mean_noise_ceiling'] - orig_result['mean_noise_ceiling']
            })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Zscore_Corrected_RSA', ascending=False)
    
    return df

def create_comparison_plots(df, save_dir='comparison_plots'):
    """创建比较图表"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2的子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RSA Analysis Comparison: Original vs Z-score Normalized', 
                fontsize=16, fontweight='bold')
    
    arealabels = df['Arealabel'].tolist()
    
    # 1. 原始RSA vs z-score RSA
    x = np.arange(len(arealabels))
    width = 0.35
    
    bars1_orig = ax1.bar(x - width/2, df['Original_Mean_RSA'], width, 
                        label='Original RSA', alpha=0.7, color='skyblue')
    bars1_zscore = ax1.bar(x + width/2, df['Zscore_Mean_RSA'], width, 
                          label='Z-score RSA', alpha=0.7, color='lightcoral')
    
    ax1.set_title('Original vs Z-score Normalized RSA', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RSA Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(arealabels, rotation=45)
    ax1.legend()
    
    # 添加数值标签
    for bar, value in zip(bars1_orig, df['Original_Mean_RSA']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars1_zscore, df['Zscore_Mean_RSA']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 矫正后RSA比较
    bars2_orig = ax2.bar(x - width/2, df['Original_Corrected_RSA'], width, 
                        label='Original Corrected RSA', alpha=0.7, color='skyblue')
    bars2_zscore = ax2.bar(x + width/2, df['Zscore_Corrected_RSA'], width, 
                          label='Z-score Corrected RSA', alpha=0.7, color='lightcoral')
    
    ax2.set_title('Corrected RSA Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Corrected RSA Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(arealabels, rotation=45)
    ax2.legend()
    
    # 添加数值标签
    for bar, value in zip(bars2_orig, df['Original_Corrected_RSA']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars2_zscore, df['Zscore_Corrected_RSA']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 差异分析
    bars3 = ax3.bar(arealabels, df['RSA_Difference'], alpha=0.7, color='orange')
    ax3.set_title('RSA Difference (Z-score - Original)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RSA Difference')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar, value in zip(bars3, df['RSA_Difference']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 噪声天花板比较
    bars4_orig = ax4.bar(x - width/2, df['Original_Noise_Ceiling'], width, 
                        label='Original Noise Ceiling', alpha=0.7, color='lightgreen')
    bars4_zscore = ax4.bar(x + width/2, df['Zscore_Noise_Ceiling'], width, 
                          label='Z-score Noise Ceiling', alpha=0.7, color='lightpink')
    
    ax4.set_title('Noise Ceiling Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Noise Ceiling')
    ax4.set_xticks(x)
    ax4.set_xticklabels(arealabels, rotation=45)
    ax4.legend()
    
    # 添加数值标签
    for bar, value in zip(bars4_orig, df['Original_Noise_Ceiling']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars4_zscore, df['Zscore_Noise_Ceiling']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = f'{save_dir}/rsa_comparison_monkey.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"比较图表已保存: {plot_file}")
    plt.close()
    
    return plot_file

def create_correlation_analysis(df, save_dir='comparison_plots'):
    """创建相关性分析图表"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2的子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Correlation Analysis: Original vs Z-score Normalized', 
                fontsize=16, fontweight='bold')
    
    # 1. 原始RSA vs z-score RSA散点图
    ax1.scatter(df['Original_Mean_RSA'], df['Zscore_Mean_RSA'], alpha=0.7, s=100)
    ax1.set_xlabel('Original RSA')
    ax1.set_ylabel('Z-score RSA')
    ax1.set_title('Original vs Z-score RSA Correlation')
    
    # 计算相关系数
    corr_rsa, p_rsa = spearmanr(df['Original_Mean_RSA'], df['Zscore_Mean_RSA'])
    ax1.text(0.05, 0.95, f'Spearman r = {corr_rsa:.3f}\np = {p_rsa:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加对角线
    min_val = min(df['Original_Mean_RSA'].min(), df['Zscore_Mean_RSA'].min())
    max_val = max(df['Original_Mean_RSA'].max(), df['Zscore_Mean_RSA'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # 2. 矫正后RSA散点图
    ax2.scatter(df['Original_Corrected_RSA'], df['Zscore_Corrected_RSA'], alpha=0.7, s=100)
    ax2.set_xlabel('Original Corrected RSA')
    ax2.set_ylabel('Z-score Corrected RSA')
    ax2.set_title('Corrected RSA Correlation')
    
    # 计算相关系数
    corr_corrected, p_corrected = spearmanr(df['Original_Corrected_RSA'], df['Zscore_Corrected_RSA'])
    ax2.text(0.05, 0.95, f'Spearman r = {corr_corrected:.3f}\np = {p_corrected:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加对角线
    min_val = min(df['Original_Corrected_RSA'].min(), df['Zscore_Corrected_RSA'].min())
    max_val = max(df['Original_Corrected_RSA'].max(), df['Zscore_Corrected_RSA'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # 3. 噪声天花板散点图
    ax3.scatter(df['Original_Noise_Ceiling'], df['Zscore_Noise_Ceiling'], alpha=0.7, s=100)
    ax3.set_xlabel('Original Noise Ceiling')
    ax3.set_ylabel('Z-score Noise Ceiling')
    ax3.set_title('Noise Ceiling Correlation')
    
    # 计算相关系数
    corr_nc, p_nc = spearmanr(df['Original_Noise_Ceiling'], df['Zscore_Noise_Ceiling'])
    ax3.text(0.05, 0.95, f'Spearman r = {corr_nc:.3f}\np = {p_nc:.3f}', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加对角线
    min_val = min(df['Original_Noise_Ceiling'].min(), df['Zscore_Noise_Ceiling'].min())
    max_val = max(df['Original_Noise_Ceiling'].max(), df['Zscore_Noise_Ceiling'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # 4. 差异分布
    ax4.hist(df['RSA_Difference'], bins=10, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('RSA Difference (Z-score - Original)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of RSA Differences')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # 添加统计信息
    mean_diff = df['RSA_Difference'].mean()
    std_diff = df['RSA_Difference'].std()
    ax4.text(0.05, 0.95, f'Mean = {mean_diff:.3f}\nStd = {std_diff:.3f}', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = f'{save_dir}/correlation_analysis_monkey.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"相关性分析图表已保存: {plot_file}")
    plt.close()
    
    return plot_file

def main():
    """主函数"""
    print("=== 比较猴子神经元信号RSA分析方法 ===")
    
    # 加载结果
    original_results = load_original_results()
    zscore_results = load_zscore_results()
    
    if original_results is None or zscore_results is None:
        print("❌ 无法加载结果，请先运行相应的分析脚本")
        return
    
    # 比较结果
    df = compare_results(original_results, zscore_results)
    
    # 显示比较结果
    print("\n=== 比较结果摘要 ===")
    print(df.to_string(index=False))
    
    # 创建比较图表
    print("\n=== 创建比较图表 ===")
    comparison_plot = create_comparison_plots(df)
    correlation_plot = create_correlation_analysis(df)
    
    # 保存比较结果
    df.to_csv('rsa_comparison_monkey.csv', index=False)
    print(f"比较结果已保存到: rsa_comparison_monkey.csv")
    
    # 统计摘要
    print("\n=== 统计摘要 ===")
    print(f"平均RSA差异: {df['RSA_Difference'].mean():.4f} ± {df['RSA_Difference'].std():.4f}")
    print(f"平均矫正RSA差异: {df['Corrected_RSA_Difference'].mean():.4f} ± {df['Corrected_RSA_Difference'].std():.4f}")
    print(f"平均噪声天花板差异: {df['Noise_Ceiling_Difference'].mean():.4f} ± {df['Noise_Ceiling_Difference'].std():.4f}")
    
    # 相关性分析
    corr_rsa, p_rsa = spearmanr(df['Original_Mean_RSA'], df['Zscore_Mean_RSA'])
    corr_corrected, p_corrected = spearmanr(df['Original_Corrected_RSA'], df['Zscore_Corrected_RSA'])
    corr_nc, p_nc = spearmanr(df['Original_Noise_Ceiling'], df['Zscore_Noise_Ceiling'])
    
    print(f"\n=== 相关性分析 ===")
    print(f"原始RSA vs Z-score RSA: r = {corr_rsa:.3f}, p = {p_rsa:.3f}")
    print(f"矫正RSA vs Z-score矫正RSA: r = {corr_corrected:.3f}, p = {p_corrected:.3f}")
    print(f"噪声天花板 vs Z-score噪声天花板: r = {corr_nc:.3f}, p = {p_nc:.3f}")
    
    print(f"\n✅ 比较分析完成！")
    print(f"📊 图表保存在: comparison_plots/ 目录")
    print(f"📁 数据保存在: rsa_comparison_monkey.csv")

if __name__ == "__main__":
    main()