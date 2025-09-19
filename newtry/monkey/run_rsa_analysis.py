"""
运行RSA分析

使用指定的LLM embedding文件路径
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def load_data():
    """加载数据"""
    # 加载猴子RDM数据
    with open('all_rdms.pkl', 'rb') as f:
        monkey_rdms = pickle.load(f)
    
    # 加载原始数据获取arealabel信息
    with open('extracted_monkey_responses.pkl', 'rb') as f:
        original_data = pickle.load(f)
    
    return monkey_rdms, original_data

def group_by_arealabel(monkey_rdms, original_data):
    """按arealabel分组"""
    arealabel_groups = {}
    
    for session_num, session_data in original_data['extracted_data'].items():
        if session_num in monkey_rdms:
            for roi_index, roi_data in session_data['rois'].items():
                arealabel = roi_data['arealabel']
                
                if arealabel not in arealabel_groups:
                    arealabel_groups[arealabel] = []
                
                arealabel_groups[arealabel].append({
                    'session': session_num,
                    'rdm': monkey_rdms[session_num],
                    'n_neurons': roi_data['n_neurons']
                })
    
    return arealabel_groups

def load_llm_embedding():
    """加载LLM embedding并计算RDM"""
    embedding_file = r"E:\lunzhuan1\visuo_llm-main\newtry\captions\embeddings_output\image_embeddings.npy"
    
    print(f"加载LLM embedding: {embedding_file}")
    
    # 加载embedding数据
    embeddings = np.load(embedding_file)
    print(f"  原始形状: {embeddings.shape}")
    
    # 检查数据形状并调整
    if len(embeddings.shape) == 2:
        if embeddings.shape[0] == 1000:
            # 形状正确 (n_features, n_images)
            pass
        elif embeddings.shape[1] == 1000:
            # 需要转置 (n_images, n_features) -> (n_features, n_images)
            embeddings = embeddings.T
            print(f"  转置后形状: {embeddings.shape}")
        else:
            print(f"  ⚠️  图片数量不是1000: {embeddings.shape}")
            # 如果图片数量不匹配，只取前1000张
            if embeddings.shape[0] > 1000:
                embeddings = embeddings[:1000, :]
                print(f"  截取前1000张图片: {embeddings.shape}")
            elif embeddings.shape[1] > 1000:
                embeddings = embeddings[:, :1000]
                print(f"  截取前1000张图片: {embeddings.shape}")
    else:
        print(f"  ❌ 数据维度错误: {embeddings.shape}")
        return None
    
    # 计算RDM
    llm_rdm = 1 - np.corrcoef(embeddings.T)
    print(f"  LLM RDM形状: {llm_rdm.shape}")
    
    return llm_rdm

def compute_rsa(rdm1, rdm2):
    """计算RSA"""
    # 确保两个RDM的维度匹配
    if rdm1.shape != rdm2.shape:
        min_size = min(rdm1.shape[0], rdm2.shape[0])
        rdm1 = rdm1[:min_size, :min_size]
        rdm2 = rdm2[:min_size, :min_size]
        print(f"    调整RDM大小到: {rdm1.shape}")
    
    mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
    rdm1_values = rdm1[mask]
    rdm2_values = rdm2[mask]
    correlation, p_value = spearmanr(rdm1_values, rdm2_values)
    return correlation, p_value

def analyze_arealabel_rsa(arealabel_groups, llm_rdm):
    """分析每个arealabel的RSA"""
    results = {}
    
    for arealabel, sessions in arealabel_groups.items():
        print(f"\n分析 {arealabel} ({len(sessions)}个session):")
        
        rsa_values = []
        session_info = []
        
        for session_data in sessions:
            session_num = session_data['session']
            rdm = session_data['rdm']
            n_neurons = session_data['n_neurons']
            
            rsa, p_value = compute_rsa(rdm, llm_rdm)
            rsa_values.append(rsa)
            session_info.append({
                'session': session_num,
                'rsa': rsa,
                'p_value': p_value,
                'n_neurons': n_neurons
            })
            
            print(f"   Session {session_num}: RSA = {rsa:.3f}, p = {p_value:.3f}")
        
        results[arealabel] = {
            'sessions': session_info,
            'mean_rsa': np.mean(rsa_values),
            'std_rsa': np.std(rsa_values),
            'rsa_values': np.array(rsa_values)
        }
        
        print(f"   平均RSA: {np.mean(rsa_values):.3f} ± {np.std(rsa_values):.3f}")
    
    return results

def plot_results(results):
    """绘制结果"""
    # 准备数据
    plot_data = []
    for arealabel, data in results.items():
        for session_info in data['sessions']:
            plot_data.append({
                'Arealabel': arealabel,
                'Session': session_info['session'],
                'RSA': session_info['rsa'],
                'N_Neurons': session_info['n_neurons']
            })
    
    df = pd.DataFrame(plot_data)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 箱线图
    import seaborn as sns
    sns.boxplot(data=df, x='Arealabel', y='RSA', ax=axes[0])
    axes[0].set_title('RSA by Arealabel')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # 平均RSA柱状图
    arealabel_means = df.groupby('Arealabel')['RSA'].agg(['mean', 'std']).reset_index()
    arealabel_means = arealabel_means.sort_values('mean', ascending=False)
    
    axes[1].bar(arealabel_means['Arealabel'], arealabel_means['mean'], 
                yerr=arealabel_means['std'], capsize=5, alpha=0.7)
    axes[1].set_title('Mean RSA by Arealabel')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rsa_by_arealabel.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== RSA分析 ===")
    
    # 加载数据
    monkey_rdms, original_data = load_data()
    print(f"加载了 {len(monkey_rdms)} 个session的RDM数据")
    
    # 按arealabel分组
    arealabel_groups = group_by_arealabel(monkey_rdms, original_data)
    print(f"按arealabel分组完成:")
    for arealabel, sessions in arealabel_groups.items():
        print(f"   {arealabel}: {len(sessions)}个session")
    
    # 加载LLM embedding
    llm_rdm = load_llm_embedding()
    if llm_rdm is None:
        print("❌ 无法加载LLM embedding数据")
        return
    
    # 进行RSA分析
    results = analyze_arealabel_rsa(arealabel_groups, llm_rdm)
    
    # 绘制结果
    plot_results(results)
    
    # 保存结果
    with open('rsa_by_arealabel_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ 结果已保存到: rsa_by_arealabel_results.pkl")
    
    # 打印摘要
    print(f"\n=== 结果摘要 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_rsa'], reverse=True)
    for arealabel, data in sorted_results:
        print(f"{arealabel}: {data['mean_rsa']:.3f} ± {data['std_rsa']:.3f} ({len(data['sessions'])}个session)")

if __name__ == "__main__":
    main()
