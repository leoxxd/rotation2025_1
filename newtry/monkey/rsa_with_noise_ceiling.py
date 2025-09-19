"""
计算RSA的noise ceiling并进行矫正
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

def load_data():
    """加载数据"""
    print("1. 加载数据...")
    
    # 加载猴子RDM
    with open('all_rdms.pkl', 'rb') as f:
        monkey_rdms = pickle.load(f)
    
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
    
    return monkey_rdms, original_data, llm_rdm

def compute_noise_ceiling(rdms):
    """
    计算noise ceiling
    
    Args:
        rdms: list of RDM matrices for the same arealabel
        
    Returns:
        noise_ceiling: 平均的session间RSA值
        individual_rsas: 每对session的RSA值
    """
    if len(rdms) < 2:
        return None, []
    
    individual_rsas = []
    
    # 计算所有session对之间的RSA
    for i in range(len(rdms)):
        for j in range(i+1, len(rdms)):
            rsa, _ = compute_rsa(rdms[i], rdms[j])
            individual_rsas.append(rsa)
    
    # noise ceiling是平均的session间RSA
    noise_ceiling = np.mean(individual_rsas)
    
    return noise_ceiling, individual_rsas

def compute_rsa(rdm1, rdm2):
    """计算RSA"""
    if rdm1.shape != rdm2.shape:
        min_size = min(rdm1.shape[0], rdm2.shape[0])
        rdm1 = rdm1[:min_size, :min_size]
        rdm2 = rdm2[:min_size, :min_size]
    
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
    
    print(f"\n分析 {arealabel} ({len(sessions_data)}个session):")
    
    # 提取RDM
    rdms = [session['rdm'] for session in sessions_data]
    session_nums = [session['session'] for session in sessions_data]
    
    # 计算noise ceiling
    noise_ceiling, individual_rsas = compute_noise_ceiling(rdms)
    
    if noise_ceiling is None:
        print(f"   ⚠️  只有1个session，无法计算noise ceiling")
        return None
    
    print(f"   Noise ceiling: {noise_ceiling:.4f}")
    print(f"   Session间RSA范围: {min(individual_rsas):.4f} - {max(individual_rsas):.4f}")
    
    # 计算每个session与LLM的RSA
    session_rsas = []
    corrected_rsas = []
    warnings = []
    
    for i, (session_num, rdm) in enumerate(zip(session_nums, rdms)):
        rsa, p_value = compute_rsa(rdm, llm_rdm)
        corrected_rsa, warning = compute_corrected_rsa(rsa, noise_ceiling)
        
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
        'noise_ceiling': noise_ceiling,
        'individual_rsas': individual_rsas,
        'session_rsas': session_rsas,
        'corrected_rsas': corrected_rsas,
        'warnings': warnings,
        'mean_rsa': mean_rsa,
        'std_rsa': std_rsa,
        'mean_corrected_rsa': mean_corrected_rsa,
        'std_corrected_rsa': std_corrected_rsa,
        'session_nums': session_nums
    }

def create_individual_arealabel_plots(results, save_dir='noise_ceiling_plots'):
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
        noise_ceiling = result['noise_ceiling']
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'RSA Analysis for {arealabel} (Noise Ceiling: {noise_ceiling:.4f})', 
                    fontsize=16, fontweight='bold')
        
        # 1. 原始RSA值条形图
        sessions = [f"Session {s}" for s in session_nums]
        bars1 = ax1.bar(sessions, session_rsas, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_title(f'Original RSA Values by Session\nMean: {np.mean(session_rsas):.4f} ± {np.std(session_rsas):.4f}')
        ax1.set_ylabel('Original RSA Value')
        ax1.set_xlabel('Session')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars1, session_rsas):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 添加平均线
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
        for bar, value in zip(bars2, corrected_rsas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 添加平均线
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
        for bar, value in zip(bars3_orig, session_rsas):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars3_corr, corrected_rsas):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 矫正比例和警告信息
        correction_ratios = [c/o for c, o in zip(corrected_rsas, session_rsas)]
        bars4 = ax4.bar(sessions, correction_ratios, color='orange', alpha=0.7, edgecolor='darkorange')
        ax4.set_title('Correction Ratio (Corrected/Original)')
        ax4.set_ylabel('Correction Ratio')
        ax4.set_xlabel('Session')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No correction')
        ax4.legend()
        
        # 添加数值标签和警告信息
        for i, (bar, ratio, warning) in enumerate(zip(bars4, correction_ratios, warnings)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 如果有警告，在图上标注
            if warning:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        '⚠️', ha='center', va='bottom', fontsize=12, color='red')
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = f'{save_dir}/rsa_{arealabel}_noise_ceiling.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {plot_file}")
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
                'Warning': warnings[i] if warnings[i] else ''
            })
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('Corrected_RSA', ascending=False)
        
        # 保存表格
        table_file = f'{save_dir}/rsa_table_{arealabel}_noise_ceiling.csv'
        df.to_csv(table_file, index=False)
        print(f"表格已保存: {table_file}")

def create_noise_ceiling_plots(results, save_dir='noise_ceiling_plots'):
    """创建noise ceiling相关的图表"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 过滤有效结果
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("没有有效的结果可以绘图")
        return
    
    arealabels = [r['arealabel'] for r in valid_results]
    noise_ceilings = [r['noise_ceiling'] for r in valid_results]
    mean_rsas = [r['mean_rsa'] for r in valid_results]
    mean_corrected_rsas = [r['mean_corrected_rsa'] for r in valid_results]
    n_sessions = [r['n_sessions'] for r in valid_results]
    
    # 创建2x2的子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RSA Analysis with Noise Ceiling Correction', fontsize=16, fontweight='bold')
    
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
    plot_file = f'{save_dir}/noise_ceiling_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_file}")
    plt.close()
    
    # 创建详细表格
    table_data = []
    for result in results:
        if result is not None:
            table_data.append({
                'Arealabel': result['arealabel'],
                'N_Sessions': result['n_sessions'],
                'Noise_Ceiling': result['noise_ceiling'],
                'Mean_Original_RSA': result['mean_rsa'],
                'Std_Original_RSA': result['std_rsa'],
                'Mean_Corrected_RSA': result['mean_corrected_rsa'],
                'Std_Corrected_RSA': result['std_corrected_rsa'],
                'Correction_Ratio': result['mean_corrected_rsa'] / result['mean_rsa']
            })
    
    import pandas as pd
    df = pd.DataFrame(table_data)
    df = df.sort_values('Mean_Corrected_RSA', ascending=False)
    
    # 保存表格
    table_file = f'{save_dir}/noise_ceiling_results.csv'
    df.to_csv(table_file, index=False)
    print(f"表格已保存: {table_file}")
    
    return df

def main():
    """主函数"""
    print("=== RSA分析with Noise Ceiling矫正 ===")
    
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
                arealabel_groups[arealabel].append({
                    'session': session_num,
                    'rdm': monkey_rdms[session_num]
                })
    
    print("   Arealabel分组结果:")
    for arealabel, sessions in arealabel_groups.items():
        print(f"     {arealabel}: {len(sessions)}个session")
    
    # 分析每个arealabel
    print("\n3. 分析每个arealabel...")
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
    print("\n=== 结果摘要 ===")
    print(df.to_string(index=False))
    
    # 保存完整结果
    with open('rsa_with_noise_ceiling_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n完整结果已保存到: rsa_with_noise_ceiling_results.pkl")
    
    print(f"\n✅ Noise ceiling分析完成！")

if __name__ == "__main__":
    main()
