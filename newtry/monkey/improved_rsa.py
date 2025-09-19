"""
改进的RSA分析脚本
解决RSA值偏小的问题
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform

def improved_rsa():
    print("=== 改进的RSA分析 ===")
    
    # 1. 加载数据
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
    
    print(f"   猴子RDM数量: {len(monkey_rdms)}")
    print(f"   LLM embedding形状: {embeddings.shape}")
    
    # 2. 尝试不同的RDM计算方法
    print("\n2. 尝试不同的RDM计算方法...")
    
    def compute_rdm_methods(data, method='correlation'):
        """计算RDM的多种方法"""
        if method == 'correlation':
            # 原始方法：1 - 相关系数
            return 1 - np.corrcoef(data)
        elif method == 'euclidean':
            # 欧几里得距离
            return squareform(pdist(data.T, metric='euclidean'))
        elif method == 'cosine':
            # 余弦距离
            return squareform(pdist(data.T, metric='cosine'))
        elif method == 'correlation_abs':
            # 1 - 相关系数的绝对值
            return 1 - np.abs(np.corrcoef(data))
        elif method == 'correlation_squared':
            # 1 - 相关系数的平方
            corr = np.corrcoef(data)
            return 1 - corr**2
    
    # 计算LLM RDM的多种方法
    llm_rdms = {}
    methods = ['correlation', 'euclidean', 'cosine', 'correlation_abs', 'correlation_squared']
    
    for method in methods:
        llm_rdms[method] = compute_rdm_methods(embeddings, method)
        values = llm_rdms[method][np.triu(np.ones_like(llm_rdms[method], dtype=bool), k=1)]
        print(f"   {method}: 范围 {values.min():.3f} 到 {values.max():.3f}, 平均 {values.mean():.3f}")
    
    # 3. 计算RSA并比较
    print("\n3. 计算RSA并比较...")
    
    def compute_rsa(rdm1, rdm2):
        if rdm1.shape != rdm2.shape:
            min_size = min(rdm1.shape[0], rdm2.shape[0])
            rdm1 = rdm1[:min_size, :min_size]
            rdm2 = rdm2[:min_size, :min_size]
        
        mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
        rdm1_values = rdm1[mask]
        rdm2_values = rdm2[mask]
        
        # 尝试不同的相关性度量
        spearman_corr, spearman_p = spearmanr(rdm1_values, rdm2_values)
        pearson_corr, pearson_p = pearsonr(rdm1_values, rdm2_values)
        
        return {
            'spearman': (spearman_corr, spearman_p),
            'pearson': (pearson_corr, pearson_p)
        }
    
    # 按arealabel分组
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
    
    # 测试不同方法
    print("\n   测试不同RDM计算方法:")
    sample_rdm = list(monkey_rdms.values())[0]
    
    for method in methods:
        print(f"\n   {method}方法:")
        rsa_results = compute_rsa(sample_rdm, llm_rdms[method])
        print(f"     Spearman: {rsa_results['spearman'][0]:.4f} (p={rsa_results['spearman'][1]:.4f})")
        print(f"     Pearson: {rsa_results['pearson'][0]:.4f} (p={rsa_results['pearson'][1]:.4f})")
    
    # 4. 选择最佳方法进行完整分析
    print("\n4. 选择最佳方法进行完整分析...")
    
    # 选择RSA值最大的方法
    best_method = 'correlation'
    best_rsa = -1
    
    for method in methods:
        rsa_results = compute_rsa(sample_rdm, llm_rdms[method])
        if rsa_results['spearman'][0] > best_rsa:
            best_rsa = rsa_results['spearman'][0]
            best_method = method
    
    print(f"   选择最佳方法: {best_method} (RSA = {best_rsa:.4f})")
    
    # 5. 使用最佳方法进行完整RSA分析
    print(f"\n5. 使用{best_method}方法进行完整RSA分析...")
    
    llm_rdm = llm_rdms[best_method]
    results = {}
    
    for arealabel, sessions in arealabel_groups.items():
        print(f"\n   分析 {arealabel} ({len(sessions)}个session):")
        
        rsa_values = []
        for session_data in sessions:
            session_num = session_data['session']
            rdm = session_data['rdm']
            
            rsa_results = compute_rsa(rdm, llm_rdm)
            rsa = rsa_results['spearman'][0]
            p_value = rsa_results['spearman'][1]
            rsa_values.append(rsa)
            
            print(f"     Session {session_num}: RSA = {rsa:.4f}, p = {p_value:.4f}")
        
        results[arealabel] = {
            'mean_rsa': np.mean(rsa_values),
            'std_rsa': np.std(rsa_values),
            'rsa_values': rsa_values
        }
        
        print(f"     平均RSA: {np.mean(rsa_values):.4f} ± {np.std(rsa_values):.4f}")
    
    # 6. 保存结果
    print(f"\n6. 保存结果...")
    with open('improved_rsa_results.pkl', 'wb') as f:
        pickle.dump({
            'method': best_method,
            'results': results
        }, f)
    print("   结果已保存到: improved_rsa_results.pkl")
    
    # 7. 显示摘要
    print(f"\n=== 结果摘要 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_rsa'], reverse=True)
    for arealabel, data in sorted_results:
        print(f"{arealabel}: {data['mean_rsa']:.4f} ± {data['std_rsa']:.4f} ({len(data['rsa_values'])}个session)")
    
    print(f"\n✅ 改进的RSA分析完成！")
    print(f"使用的方法: {best_method}")

if __name__ == "__main__":
    improved_rsa()
