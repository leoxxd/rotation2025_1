"""
计算RSA的noise ceiling并进行矫正 - z-score归一化版本

完全基于rsa_with_noise_ceiling.py，只是使用z-score归一化的RDM文件
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    print("1. 加载数据...")
    
    # 加载猴子RDM（z-score归一化版本）
    with open('all_rdms_correlation_zscore.pkl', 'rb') as f:
        rdm_data = pickle.load(f)
        monkey_rdms = rdm_data['all_rdms']  # 提取实际的RDM数据
    
    # 加载原始数据
    with open('extracted_monkey_responses.pkl', 'rb') as f:
        original_data = pickle.load(f)
    
    # 加载LLM embedding
    embedding_file = r"E:\lunzhuan1\visuo_llm-main\newtry\captions\embeddings_output\image_embeddings.npy"
    embeddings = np.load(embedding_file)
    
    if embeddings.shape[0] == 1000:
        pass
    elif embeddings.shape[1] == 1000:
        embeddings = embeddings.T
    else:
        if embeddings.shape[0] > 1000:
            embeddings = embeddings[:1000, :]
        elif embeddings.shape[1] > 1000:
            embeddings = embeddings[:, :1000]
    
    # 计算LLM RDM
    llm_rdm = 1 - np.corrcoef(embeddings)
    
    print(f"   猴子RDM数量: {len(monkey_rdms)}")
    print(f"   LLM RDM形状: {llm_rdm.shape}")
    print(f"   归一化方法: z-score per neuron across images")
    
    return monkey_rdms, original_data, llm_rdm

def compute_noise_ceiling(rdms):
    """
    计算noise ceiling (使用原始项目的方法：每个session与其他session平均RDM的相关性)
    
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
        rsa, _ = compute_rsa(current_rdm, mean_other_rdm)
        individual_noise_ceilings.append(rsa)
    
    # 计算平均noise ceiling
    mean_noise_ceiling = np.mean(individual_noise_ceilings)
    
    return individual_noise_ceilings, mean_noise_ceiling

def compute_rsa(rdm1, rdm2):
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

def compute_corrected_rsa(rsa_value, noise_ceiling):
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

def analyze_arealabel_with_noise_ceiling(arealabel, sessions_data, llm_rdm):
    """分析单个arealabel的RSA和noise ceiling"""
    
    print(f"\n分析 {arealabel} ({len(sessions_data)}个session) - z-score归一化版本:")
    
    # 提取RDM
    rdms = [session['rdm'] for session in sessions_data]
    session_nums = [session['session'] for session in sessions_data]
    
    # 计算noise ceiling
    individual_noise_ceilings, mean_noise_ceiling = compute_noise_ceiling(rdms)
    
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
        rsa, p_value = compute_rsa(rdm, llm_rdm)
        
        # 使用该session的独立noise ceiling进行矫正
        corrected_rsa, warning = compute_corrected_rsa(rsa, individual_noise_ceilings[i])
        
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
        'session_nums': session_nums
    }

def create_individual_arealabel_plots(results, save_dir='noise_ceiling_plots_zscore'):
    """为每个arealabel创建单独的图表"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
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
        fig.suptitle(f'RSA Analysis for {arealabel} (z-score归一化, Mean Noise Ceiling: {mean_noise_ceiling:.4f})', 
                    fontsize=16, fontweight='bold')
        
        # 1. 原始RSA值条形图（带noise ceiling标注）
        sessions = [f"Session {s}" for s in session_nums]
        bars1 = ax1.bar(sessions, session_rsas, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_title(f'Original RSA Values by Session (z-score归一化)\nMean: {np.mean(session_rsas):.4f} ± {np.std(session_rsas):.4f}')
        ax1.set_ylabel('Original RSA Value')
        ax1.set_xlabel('Session')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签（RSA值和对应的noise ceiling）
        for bar, rsa_value, nc_value in zip(bars1, session_rsas, individual_noise_ceilings):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{rsa_value:.4f}', ha='center', va='bottom', fontsize=9)
            # 在下方显示noise ceiling
            ax1.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                    f'NC: {nc_value:.3f}', ha='center', va='top', fontsize=8, color='red')
        
        # 添加平均线
        ax1.axhline(y=np.mean(session_rsas), color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {np.mean(session_rsas):.4f}')
        ax1.legend()
        
        # 2. 矫正后RSA值条形图（带noise ceiling标注）
        bars2 = ax2.bar(sessions, corrected_rsas, color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax2.set_title(f'Corrected RSA Values by Session (z-score归一化)\nMean: {np.mean(corrected_rsas):.4f} ± {np.std(corrected_rsas):.4f}')
        ax2.set_ylabel('Corrected RSA Value')
        ax2.set_xlabel('Session')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签（矫正RSA值和对应的noise ceiling）
        for bar, corr_value, nc_value in zip(bars2, corrected_rsas, individual_noise_ceilings):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{corr_value:.4f}', ha='center', va='bottom', fontsize=9)
            # 在下方显示noise ceiling
            ax2.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                    f'NC: {nc_value:.3f}', ha='center', va='top', fontsize=8, color='red')
        
        # 添加平均线
        ax2.axhline(y=np.mean(corrected_rsas), color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {np.mean(corrected_rsas):.4f}')
        ax2.legend()
        
        # 3. 原始vs矫正对比
        x = np.arange(len(sessions))
        width = 0.35
        
        bars3_orig = ax3.bar(x - width/2, session_rsas, width, label='Original RSA', alpha=0.7, color='skyblue')
        bars3_corr = ax3.bar(x + width/2, corrected_rsas, width, label='Corrected RSA', alpha=0.7, color='lightcoral')
        
        ax3.set_title('Original vs Corrected RSA Comparison (z-score归一化)')
        ax3.set_ylabel('RSA Value')
        ax3.set_xlabel('Session')
        ax3.set_xticks(x)
        ax3.set_xticklabels(sessions, rotation=45)
        ax3.legend()
        
        # 添加数值标签（带noise ceiling）
        for bar, value, nc_value in zip(bars3_orig, session_rsas, individual_noise_ceilings):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            # 在下方显示noise ceiling
            ax3.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                    f'NC:{nc_value:.2f}', ha='center', va='top', fontsize=7, color='red')
        
        for bar, value in zip(bars3_corr, corrected_rsas):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 矫正比例信息（每个session独立）
        correction_ratios = [corrected_rsas[i] / session_rsas[i] for i in range(len(session_rsas))]
        
        # 显示矫正比例信息（条形图形式）
        bars4 = ax4.bar(sessions, correction_ratios, color='orange', alpha=0.7, edgecolor='darkorange')
        ax4.set_title('Correction Ratio by Session (z-score归一化)')
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
        plot_file = f'{save_dir}/rsa_{arealabel}_zscore.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {plot_file}")
        plt.close()
        
        # 创建详细表格
        table_data = []
        for i, session_num in enumerate(session_nums):
            table_data.append({
                'Session': session_num,
                'Original_RSA': session_rsas[i],
                'Corrected_RSA': corrected_rsas[i],
                'Correction_Ratio': correction_ratios[i],
                'Noise_Ceiling': individual_noise_ceilings[i],
                'Warning': warnings[i] if warnings[i] else ''
            })
        
        import pandas as pd
        df = pd.DataFrame(table_data)
        df = df.sort_values('Corrected_RSA', ascending=False)
        
        # 保存表格
        table_file = f'{save_dir}/rsa_table_{arealabel}_zscore.csv'
        df.to_csv(table_file, index=False)
        print(f"表格已保存: {table_file}")

def create_noise_ceiling_plots(results, save_dir='noise_ceiling_plots_zscore'):
    """创建noise ceiling相关的图表"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 过滤有效结果
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("没有有效的结果可以绘图")
        return
    
    arealabels = [r['arealabel'] for r in valid_results]
    noise_ceilings = [r['mean_noise_ceiling'] for r in valid_results]
    mean_rsas = [r['mean_rsa'] for r in valid_results]
    mean_corrected_rsas = [r['mean_corrected_rsa'] for r in valid_results]
    n_sessions = [r['n_sessions'] for r in valid_results]
    
    # 创建2x2的子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RSA Analysis with Noise Ceiling Correction (z-score归一化版本)', 
                fontsize=16, fontweight='bold')
    
    # 1. Noise ceiling分布
    bars1 = ax1.bar(arealabels, noise_ceilings, color='lightblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('Noise Ceiling by Arealabel (z-score归一化)', fontsize=12, fontweight='bold')
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
    
    ax2.set_title('Original vs Corrected RSA (z-score归一化)', fontsize=12, fontweight='bold')
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
    ax3.set_title('Corrected RSA (Sorted, z-score归一化)', fontsize=12, fontweight='bold')
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
    ax4.set_title('Correction Ratio (z-score归一化)', fontsize=12, fontweight='bold')
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
    plot_file = f'{save_dir}/noise_ceiling_analysis_zscore.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"汇总图表已保存: {plot_file}")
    plt.close()
    
    # 创建详细表格
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
                'Method': 'zscore_normalized',
                'Normalization': 'zscore_per_neuron_across_images'
            })
    
    import pandas as pd
    df = pd.DataFrame(table_data)
    df = df.sort_values('Mean_Corrected_RSA', ascending=False)
    
    # 保存表格
    table_file = f'{save_dir}/noise_ceiling_results_zscore.csv'
    df.to_csv(table_file, index=False)
    print(f"汇总表格已保存: {table_file}")
    
    return df

def main():
    """主函数"""
    print("=== 猴子神经元信号RSA分析with z-score归一化 ===")
    
    # 加载数据
    monkey_rdms, original_data, llm_rdm = load_data()
    
    # 按arealabel分组
    print("\n2. 按arealabel分组...")
    arealabel_groups = {}
    for session_num, session_data in original_data['extracted_data'].items():
        if session_num in monkey_rdms:
            for roi_index, roi_data in session_data['rois'].items():
                arealabel = roi_data['arealabel']
                if arealabel not in arealabel_groups:
                    arealabel_groups[arealabel] = []
                
                # 从新的数据结构中获取RDM
                if arealabel in monkey_rdms[session_num]:
                    arealabel_groups[arealabel].append({
                        'session': session_num,
                        'rdm': monkey_rdms[session_num][arealabel]
                    })
    
    print("   Arealabel分组结果:")
    for arealabel, sessions in arealabel_groups.items():
        print(f"     {arealabel}: {len(sessions)}个session")
    
    # 分析每个arealabel
    print("\n3. 分析每个arealabel（z-score归一化版本）...")
    results = []
    
    for arealabel, sessions in arealabel_groups.items():
        result = analyze_arealabel_with_noise_ceiling(arealabel, sessions, llm_rdm)
        results.append(result)
    
    # 创建图表和表格
    print("\n4. 创建图表和表格...")
    
    # 为每个arealabel创建单独的图表
    print("   创建每个arealabel的单独图表...")
    create_individual_arealabel_plots(results)
    
    # 创建汇总图表
    print("   创建汇总图表...")
    df = create_noise_ceiling_plots(results)
    
    # 显示结果摘要
    print("\n=== 结果摘要（z-score归一化版本）===")
    print(df.to_string(index=False))
    
    # 保存完整结果
    with open('rsa_with_zscore_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n完整结果已保存到: rsa_with_zscore_results.pkl")
    
    print(f"\n✅ z-score归一化RSA分析完成！")
    print(f"📊 归一化方法: 每个神经元在1000张图片上进行z-score归一化")
    print(f"📁 结果保存在: noise_ceiling_plots_zscore/ 目录")

if __name__ == "__main__":
    main()