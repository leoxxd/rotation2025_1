"""
修正的RSA分析with Noise Ceiling矫正
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

def compute_corrected_rsa(rsa_value, noise_ceiling, method='conservative'):
    """
    计算矫正后的RSA值
    
    Args:
        rsa_value: 原始RSA值
        noise_ceiling: noise ceiling值
        method: 矫正方法 ('conservative', 'proportional', 'sqrt')
        
    Returns:
        corrected_rsa: 矫正后的RSA值
    """
    if noise_ceiling is None or noise_ceiling <= 0:
        return rsa_value
    
    if method == 'conservative':
        # 保守方法：RSA / (1 + noise_ceiling)
        # 这样即使noise ceiling很小，矫正后的值也不会超过1
        corrected_rsa = rsa_value / (1 + noise_ceiling)
        
    elif method == 'proportional':
        # 比例方法：RSA / noise_ceiling，但限制在合理范围内
        corrected_rsa = rsa_value / noise_ceiling
        corrected_rsa = min(corrected_rsa, 1.0)  # 限制在1以内
        
    elif method == 'sqrt':
        # 平方根方法：RSA / sqrt(noise_ceiling)
        corrected_rsa = rsa_value / np.sqrt(noise_ceiling)
        corrected_rsa = min(corrected_rsa, 1.0)
        
    else:
        corrected_rsa = rsa_value
    
    # 确保矫正后的值在合理范围内
    corrected_rsa = max(0.0, min(corrected_rsa, 1.0))
    
    return corrected_rsa

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
    corrected_rsas_conservative = []
    corrected_rsas_proportional = []
    corrected_rsas_sqrt = []
    
    for i, (session_num, rdm) in enumerate(zip(session_nums, rdms)):
        rsa, p_value = compute_rsa(rdm, llm_rdm)
        
        # 尝试不同的矫正方法
        corrected_conservative = compute_corrected_rsa(rsa, noise_ceiling, 'conservative')
        corrected_proportional = compute_corrected_rsa(rsa, noise_ceiling, 'proportional')
        corrected_sqrt = compute_corrected_rsa(rsa, noise_ceiling, 'sqrt')
        
        session_rsas.append(rsa)
        corrected_rsas_conservative.append(corrected_conservative)
        corrected_rsas_proportional.append(corrected_proportional)
        corrected_rsas_sqrt.append(corrected_sqrt)
        
        print(f"     Session {session_num}:")
        print(f"       原始RSA: {rsa:.4f} (p={p_value:.4f})")
        print(f"       保守矫正: {corrected_conservative:.4f}")
        print(f"       比例矫正: {corrected_proportional:.4f}")
        print(f"       平方根矫正: {corrected_sqrt:.4f}")
    
    # 计算统计量
    mean_rsa = np.mean(session_rsas)
    std_rsa = np.std(session_rsas)
    
    mean_conservative = np.mean(corrected_rsas_conservative)
    mean_proportional = np.mean(corrected_rsas_proportional)
    mean_sqrt = np.mean(corrected_rsas_sqrt)
    
    print(f"   原始RSA: {mean_rsa:.4f} ± {std_rsa:.4f}")
    print(f"   保守矫正: {mean_conservative:.4f}")
    print(f"   比例矫正: {mean_proportional:.4f}")
    print(f"   平方根矫正: {mean_sqrt:.4f}")
    
    return {
        'arealabel': arealabel,
        'n_sessions': len(sessions_data),
        'noise_ceiling': noise_ceiling,
        'individual_rsas': individual_rsas,
        'session_rsas': session_rsas,
        'corrected_conservative': corrected_rsas_conservative,
        'corrected_proportional': corrected_rsas_proportional,
        'corrected_sqrt': corrected_rsas_sqrt,
        'mean_rsa': mean_rsa,
        'std_rsa': std_rsa,
        'mean_conservative': mean_conservative,
        'mean_proportional': mean_proportional,
        'mean_sqrt': mean_sqrt,
        'session_nums': session_nums
    }

def create_comparison_plot(results, save_dir='noise_ceiling_plots'):
    """创建不同矫正方法的比较图表"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 过滤有效结果
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("没有有效的结果可以绘图")
        return
    
    arealabels = [r['arealabel'] for r in valid_results]
    mean_rsas = [r['mean_rsa'] for r in valid_results]
    mean_conservative = [r['mean_conservative'] for r in valid_results]
    mean_proportional = [r['mean_proportional'] for r in valid_results]
    mean_sqrt = [r['mean_sqrt'] for r in valid_results]
    noise_ceilings = [r['noise_ceiling'] for r in valid_results]
    
    # 创建比较图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RSA Correction Methods Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(arealabels))
    width = 0.2
    
    # 1. 不同矫正方法的比较
    ax1.bar(x - 1.5*width, mean_rsas, width, label='Original RSA', alpha=0.7)
    ax1.bar(x - 0.5*width, mean_conservative, width, label='Conservative', alpha=0.7)
    ax1.bar(x + 0.5*width, mean_proportional, width, label='Proportional', alpha=0.7)
    ax1.bar(x + 1.5*width, mean_sqrt, width, label='Sqrt', alpha=0.7)
    
    ax1.set_title('RSA Values by Correction Method')
    ax1.set_ylabel('RSA Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(arealabels, rotation=45)
    ax1.legend()
    
    # 2. Noise ceiling分布
    ax2.bar(arealabels, noise_ceilings, color='lightblue', alpha=0.7)
    ax2.set_title('Noise Ceiling by Arealabel')
    ax2.set_ylabel('Noise Ceiling')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 矫正效果比较
    conservative_ratios = [c/o for c, o in zip(mean_conservative, mean_rsas)]
    proportional_ratios = [p/o for p, o in zip(mean_proportional, mean_rsas)]
    sqrt_ratios = [s/o for s, o in zip(mean_sqrt, mean_rsas)]
    
    ax3.bar(x - width, conservative_ratios, width, label='Conservative', alpha=0.7)
    ax3.bar(x, proportional_ratios, width, label='Proportional', alpha=0.7)
    ax3.bar(x + width, sqrt_ratios, width, label='Sqrt', alpha=0.7)
    
    ax3.set_title('Correction Ratios (Corrected/Original)')
    ax3.set_ylabel('Correction Ratio')
    ax3.set_xticks(x)
    ax3.set_xticklabels(arealabels, rotation=45)
    ax3.legend()
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # 4. 推荐方法（保守方法）
    ax4.bar(arealabels, mean_conservative, color='lightgreen', alpha=0.7)
    ax4.set_title('Recommended: Conservative Correction')
    ax4.set_ylabel('Corrected RSA Value')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = f'{save_dir}/rsa_correction_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_file}")
    plt.close()
    
    # 创建详细表格
    import pandas as pd
    table_data = []
    for result in valid_results:
        table_data.append({
            'Arealabel': result['arealabel'],
            'N_Sessions': result['n_sessions'],
            'Noise_Ceiling': result['noise_ceiling'],
            'Mean_Original_RSA': result['mean_rsa'],
            'Mean_Conservative': result['mean_conservative'],
            'Mean_Proportional': result['mean_proportional'],
            'Mean_Sqrt': result['mean_sqrt'],
            'Conservative_Ratio': result['mean_conservative'] / result['mean_rsa']
        })
    
    df = pd.DataFrame(table_data)
    df = df.sort_values('Mean_Conservative', ascending=False)
    
    # 保存表格
    table_file = f'{save_dir}/rsa_correction_comparison.csv'
    df.to_csv(table_file, index=False)
    print(f"表格已保存: {table_file}")
    
    return df

def main():
    """主函数"""
    print("=== 修正的RSA分析with Noise Ceiling矫正 ===")
    
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
    df = create_comparison_plot(results)
    
    # 显示结果摘要
    print("\n=== 结果摘要 ===")
    print(df.to_string(index=False))
    
    # 保存完整结果
    with open('rsa_noise_ceiling_fixed_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n完整结果已保存到: rsa_noise_ceiling_fixed_results.pkl")
    
    print(f"\n✅ 修正的Noise ceiling分析完成！")
    print("推荐使用保守矫正方法，因为它能确保矫正后的值在合理范围内。")

if __name__ == "__main__":
    main()
