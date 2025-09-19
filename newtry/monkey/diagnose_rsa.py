"""
诊断RSA值偏小的问题
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def diagnose_rsa():
    print("=== RSA诊断分析 ===")
    
    # 1. 加载猴子RDM数据
    print("1. 加载猴子RDM数据...")
    with open('all_rdms.pkl', 'rb') as f:
        monkey_rdms = pickle.load(f)
    
    print(f"   加载了 {len(monkey_rdms)} 个session的RDM")
    
    # 2. 加载原始数据获取arealabel信息
    print("2. 加载原始数据...")
    with open('extracted_monkey_responses.pkl', 'rb') as f:
        original_data = pickle.load(f)
    
    # 3. 加载LLM embedding
    print("3. 加载LLM embedding...")
    embedding_file = r"E:\lunzhuan1\visuo_llm-main\newtry\captions\embeddings_output\image_embeddings.npy"
    
    try:
        embeddings = np.load(embedding_file)
        print(f"   原始形状: {embeddings.shape}")
        
        # 调整数据形状
        if embeddings.shape[0] == 1000:
            pass
        elif embeddings.shape[1] == 1000:
            embeddings = embeddings.T
            print(f"   转置后形状: {embeddings.shape}")
        else:
            print(f"   ⚠️  图片数量不匹配: {embeddings.shape}")
            if embeddings.shape[0] > 1000:
                embeddings = embeddings[:1000, :]
            elif embeddings.shape[1] > 1000:
                embeddings = embeddings[:, :1000]
            print(f"   截取后形状: {embeddings.shape}")
        
        # 计算LLM RDM
        llm_rdm = 1 - np.corrcoef(embeddings.T)
        print(f"   LLM RDM形状: {llm_rdm.shape}")
        
    except Exception as e:
        print(f"   ❌ 加载LLM embedding失败: {e}")
        return
    
    # 4. 分析RDM特征
    print("\n4. 分析RDM特征...")
    
    # 分析LLM RDM
    print("   LLM RDM分析:")
    llm_values = llm_rdm[np.triu(np.ones_like(llm_rdm, dtype=bool), k=1)]
    print(f"     值范围: {llm_values.min():.3f} 到 {llm_values.max():.3f}")
    print(f"     平均值: {llm_values.mean():.3f}")
    print(f"     标准差: {llm_values.std():.3f}")
    
    # 分析几个猴子RDM
    print("   猴子RDM分析:")
    for i, (session_num, rdm) in enumerate(list(monkey_rdms.items())[:3]):
        rdm_values = rdm[np.triu(np.ones_like(rdm, dtype=bool), k=1)]
        print(f"     Session {session_num}:")
        print(f"       形状: {rdm.shape}")
        print(f"       值范围: {rdm_values.min():.3f} 到 {rdm_values.max():.3f}")
        print(f"       平均值: {rdm_values.mean():.3f}")
        print(f"       标准差: {rdm_values.std():.3f}")
    
    # 5. 计算RSA并分析
    print("\n5. 计算RSA并分析...")
    
    def compute_rsa(rdm1, rdm2):
        if rdm1.shape != rdm2.shape:
            min_size = min(rdm1.shape[0], rdm2.shape[0])
            rdm1 = rdm1[:min_size, :min_size]
            rdm2 = rdm2[:min_size, :min_size]
        
        mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
        rdm1_values = rdm1[mask]
        rdm2_values = rdm2[mask]
        correlation, p_value = spearmanr(rdm1_values, rdm2_values)
        return correlation, p_value, rdm1_values, rdm2_values
    
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
    
    # 分析每个arealabel
    all_rsa_values = []
    for arealabel, sessions in arealabel_groups.items():
        print(f"\n   分析 {arealabel} ({len(sessions)}个session):")
        
        rsa_values = []
        for session_data in sessions:
            session_num = session_data['session']
            rdm = session_data['rdm']
            
            rsa, p_value, monkey_values, llm_values = compute_rsa(rdm, llm_rdm)
            rsa_values.append(rsa)
            all_rsa_values.append(rsa)
            
            print(f"     Session {session_num}: RSA = {rsa:.4f}, p = {p_value:.4f}")
        
        print(f"     平均RSA: {np.mean(rsa_values):.4f} ± {np.std(rsa_values):.4f}")
    
    # 6. 整体分析
    print(f"\n6. 整体RSA分析:")
    print(f"   所有RSA值范围: {np.min(all_rsa_values):.4f} 到 {np.max(all_rsa_values):.4f}")
    print(f"   平均RSA: {np.mean(all_rsa_values):.4f} ± {np.std(all_rsa_values):.4f}")
    
    # 7. 可能的问题分析
    print(f"\n7. 可能的问题分析:")
    
    # 检查RDM是否过于相似
    if np.std(all_rsa_values) < 0.01:
        print("   ⚠️  RSA值变化很小，可能RDM过于相似")
    
    # 检查RDM值范围
    if llm_values.max() < 0.1:
        print("   ⚠️  LLM RDM值范围很小，可能embedding过于相似")
    
    # 检查猴子RDM值范围
    sample_rdm = list(monkey_rdms.values())[0]
    sample_values = sample_rdm[np.triu(np.ones_like(sample_rdm, dtype=bool), k=1)]
    if sample_values.max() < 0.1:
        print("   ⚠️  猴子RDM值范围很小，可能response过于相似")
    
    # 8. 建议
    print(f"\n8. 改进建议:")
    print("   - 检查数据预处理是否正确")
    print("   - 尝试不同的RDM计算方法")
    print("   - 检查embedding质量")
    print("   - 考虑使用不同的相似性度量")

if __name__ == "__main__":
    diagnose_rsa()
