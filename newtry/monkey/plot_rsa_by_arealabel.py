"""
为每个arealabel生成单独的RSA分析图表和表格
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

def load_rsa_results():
    """加载RSA分析结果"""
    with open('improved_rsa_results.pkl', 'rb') as f:
        results = pickle.load(f)
    return results

def load_monkey_data():
    """加载猴子数据用于获取详细信息"""
    with open('extracted_monkey_responses.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def create_arealabel_plot(arealabel, rsa_data, monkey_data, save_dir='rsa_plots'):
    """为单个arealabel创建详细的图表和表格"""
    
    # 创建保存目录
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取该arealabel的所有session信息
    sessions_info = []
    for session_num, session_data in monkey_data['extracted_data'].items():
        for roi_index, roi_data in session_data['rois'].items():
            if roi_data['arealabel'] == arealabel:
                sessions_info.append({
                    'session': session_num,
                    'roi': roi_index,
                    'n_neurons': roi_data['n_neurons'],
                    'y1': roi_data['y1'],
                    'y2': roi_data['y2']
                })
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'RSA Analysis for {arealabel}', fontsize=16, fontweight='bold')
    
    # 1. RSA值条形图
    sessions = [f"Session {info['session']}" for info in sessions_info]
    rsa_values = rsa_data['rsa_values']
    
    bars = ax1.bar(sessions, rsa_values, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_title(f'RSA Values by Session\nMean: {rsa_data["mean_rsa"]:.4f} ± {rsa_data["std_rsa"]:.4f}')
    ax1.set_ylabel('RSA Value')
    ax1.set_xlabel('Session')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, rsa_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 添加平均线
    ax1.axhline(y=rsa_data['mean_rsa'], color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {rsa_data["mean_rsa"]:.4f}')
    ax1.legend()
    
    # 2. RSA值分布直方图
    ax2.hist(rsa_values, bins=max(3, len(rsa_values)//2), color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax2.set_title('RSA Values Distribution')
    ax2.set_xlabel('RSA Value')
    ax2.set_ylabel('Frequency')
    ax2.axvline(x=rsa_data['mean_rsa'], color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {rsa_data["mean_rsa"]:.4f}')
    ax2.legend()
    
    # 3. 神经元数量 vs RSA值散点图
    neuron_counts = [info['n_neurons'] for info in sessions_info]
    ax3.scatter(neuron_counts, rsa_values, s=100, alpha=0.7, color='orange')
    ax3.set_title('Neuron Count vs RSA Value')
    ax3.set_xlabel('Number of Neurons')
    ax3.set_ylabel('RSA Value')
    
    # 添加相关系数
    if len(neuron_counts) > 1:
        corr, p_val = spearmanr(neuron_counts, rsa_values)
        ax3.text(0.05, 0.95, f'Spearman r = {corr:.3f}\np = {p_val:.3f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. 位置范围信息
    y1_values = [info['y1'] for info in sessions_info]
    y2_values = [info['y2'] for info in sessions_info]
    
    ax4.bar(sessions, [y2-y1 for y1, y2 in zip(y1_values, y2_values)], 
            color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax4.set_title('ROI Position Range (y2 - y1)')
    ax4.set_ylabel('Position Range')
    ax4.set_xlabel('Session')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = f'{save_dir}/rsa_{arealabel}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_file}")
    plt.close()
    
    # 创建详细表格
    table_data = []
    for i, info in enumerate(sessions_info):
        table_data.append({
            'Session': info['session'],
            'ROI': info['roi'],
            'RSA Value': rsa_values[i],
            'Neurons': info['n_neurons'],
            'Y1': info['y1'],
            'Y2': info['y2'],
            'Range': info['y2'] - info['y1']
        })
    
    df = pd.DataFrame(table_data)
    df = df.sort_values('RSA Value', ascending=False)
    
    # 保存表格
    table_file = f'{save_dir}/rsa_table_{arealabel}.csv'
    df.to_csv(table_file, index=False)
    print(f"表格已保存: {table_file}")
    
    # 打印摘要
    print(f"\n=== {arealabel} 摘要 ===")
    print(f"Session数量: {len(sessions_info)}")
    print(f"平均RSA: {rsa_data['mean_rsa']:.4f} ± {rsa_data['std_rsa']:.4f}")
    print(f"RSA范围: {min(rsa_values):.4f} - {max(rsa_values):.4f}")
    print(f"总神经元数: {sum(neuron_counts)}")
    print(f"平均神经元数: {np.mean(neuron_counts):.1f} ± {np.std(neuron_counts):.1f}")
    
    return df

def create_summary_plot(all_results, save_dir='rsa_plots'):
    """创建所有arealabel的汇总图表"""
    
    # 准备数据
    arealabels = list(all_results.keys())
    mean_rsa = [all_results[label]['mean_rsa'] for label in arealabels]
    std_rsa = [all_results[label]['std_rsa'] for label in arealabels]
    n_sessions = [len(all_results[label]['rsa_values']) for label in arealabels]
    
    # 按RSA值排序
    sorted_data = sorted(zip(arealabels, mean_rsa, std_rsa, n_sessions), 
                        key=lambda x: x[1], reverse=True)
    arealabels, mean_rsa, std_rsa, n_sessions = zip(*sorted_data)
    
    # 创建汇总图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. 所有arealabel的RSA值比较
    bars = ax1.bar(range(len(arealabels)), mean_rsa, yerr=std_rsa, 
                   color='lightblue', alpha=0.7, edgecolor='navy', capsize=5)
    ax1.set_title('RSA Values by Arealabel', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean RSA Value')
    ax1.set_xlabel('Arealabel')
    ax1.set_xticks(range(len(arealabels)))
    ax1.set_xticklabels(arealabels, rotation=45, ha='right')
    
    # 添加数值标签
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, mean_rsa, std_rsa)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.005,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Session数量 vs RSA值
    ax2.scatter(n_sessions, mean_rsa, s=100, alpha=0.7, color='red')
    ax2.set_title('Number of Sessions vs Mean RSA Value')
    ax2.set_xlabel('Number of Sessions')
    ax2.set_ylabel('Mean RSA Value')
    
    # 添加arealabel标签
    for i, label in enumerate(arealabels):
        ax2.annotate(label, (n_sessions[i], mean_rsa[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 添加相关系数
    if len(n_sessions) > 1:
        corr, p_val = spearmanr(n_sessions, mean_rsa)
        ax2.text(0.05, 0.95, f'Spearman r = {corr:.3f}\np = {p_val:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存汇总图表
    summary_file = f'{save_dir}/rsa_summary_all_arealabels.png'
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    print(f"汇总图表已保存: {summary_file}")
    plt.close()
    
    # 创建汇总表格
    summary_data = []
    for i, label in enumerate(arealabels):
        summary_data.append({
            'Arealabel': label,
            'Mean RSA': mean_rsa[i],
            'Std RSA': std_rsa[i],
            'N Sessions': n_sessions[i],
            'Min RSA': min(all_results[label]['rsa_values']),
            'Max RSA': max(all_results[label]['rsa_values'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file_csv = f'{save_dir}/rsa_summary_table.csv'
    summary_df.to_csv(summary_file_csv, index=False)
    print(f"汇总表格已保存: {summary_file_csv}")
    
    return summary_df

def main():
    """主函数"""
    print("=== 生成RSA分析图表和表格 ===")
    
    # 加载数据
    print("1. 加载数据...")
    rsa_results = load_rsa_results()
    monkey_data = load_monkey_data()
    
    method = rsa_results['method']
    results = rsa_results['results']
    
    print(f"   使用方法: {method}")
    print(f"   Arealabel数量: {len(results)}")
    
    # 为每个arealabel创建图表
    print("\n2. 为每个arealabel创建图表...")
    all_tables = {}
    
    for arealabel in results.keys():
        print(f"\n处理 {arealabel}...")
        df = create_arealabel_plot(arealabel, results[arealabel], monkey_data)
        all_tables[arealabel] = df
    
    # 创建汇总图表
    print("\n3. 创建汇总图表...")
    summary_df = create_summary_plot(results)
    
    # 显示汇总结果
    print("\n=== 最终汇总结果 ===")
    print(summary_df.to_string(index=False))
    
    print(f"\n✅ 所有图表和表格已生成完成！")
    print(f"文件保存在: rsa_plots/ 目录中")

if __name__ == "__main__":
    main()
