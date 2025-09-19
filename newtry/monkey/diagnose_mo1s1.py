"""
诊断MO1s1的RSA和noise ceiling问题
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def load_data():
    """加载数据"""
    with open('all_rdms.pkl', 'rb') as f:
        monkey_rdms = pickle.load(f)
    
    with open('extracted_monkey_responses.pkl', 'rb') as f:
        original_data = pickle.load(f)
    
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
    
    llm_rdm = 1 - np.corrcoef(embeddings)
    
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

def diagnose_mo1s1():
    """诊断MO1s1的问题"""
    print("=== 诊断MO1s1的RSA和noise ceiling问题 ===")
    
    # 加载数据
    monkey_rdms, original_data, llm_rdm = load_data()
    
    # 找到MO1s1的sessions
    mo1s1_sessions = []
    for session_num, session_data in original_data['extracted_data'].items():
        if session_num in monkey_rdms:
            for roi_index, roi_data in session_data['rois'].items():
                if roi_data['arealabel'] == 'MO1s1':
                    mo1s1_sessions.append({
                        'session': session_num,
                        'roi': roi_index,
                        'rdm': monkey_rdms[session_num],
                        'n_neurons': roi_data['n_neurons'],
                        'y1': roi_data['y1'],
                        'y2': roi_data['y2']
                    })
    
    print(f"MO1s1 sessions: {len(mo1s1_sessions)}")
    for session in mo1s1_sessions:
        print(f"  Session {session['session']}: {session['n_neurons']} neurons, y1={session['y1']}, y2={session['y2']}")
    
    if len(mo1s1_sessions) < 2:
        print("MO1s1只有1个session，无法计算noise ceiling")
        return
    
    # 计算session间RSA
    print(f"\n计算session间RSA:")
    session_rsas = []
    for i in range(len(mo1s1_sessions)):
        for j in range(i+1, len(mo1s1_sessions)):
            session1 = mo1s1_sessions[i]
            session2 = mo1s1_sessions[j]
            rsa, p_value = compute_rsa(session1['rdm'], session2['rdm'])
            session_rsas.append(rsa)
            print(f"  Session {session1['session']} vs Session {session2['session']}: RSA = {rsa:.4f} (p={p_value:.4f})")
    
    noise_ceiling = np.mean(session_rsas)
    print(f"\nNoise ceiling: {noise_ceiling:.4f}")
    
    # 计算每个session与LLM的RSA
    print(f"\n计算与LLM的RSA:")
    llm_rsas = []
    for session in mo1s1_sessions:
        rsa, p_value = compute_rsa(session['rdm'], llm_rdm)
        llm_rsas.append(rsa)
        print(f"  Session {session['session']}: RSA = {rsa:.4f} (p={p_value:.4f})")
    
    # 分析问题
    print(f"\n=== 问题分析 ===")
    print(f"Session间RSA范围: {min(session_rsas):.4f} - {max(session_rsas):.4f}")
    print(f"与LLM的RSA范围: {min(llm_rsas):.4f} - {max(llm_rsas):.4f}")
    print(f"Noise ceiling: {noise_ceiling:.4f}")
    print(f"平均与LLM的RSA: {np.mean(llm_rsas):.4f}")
    
    # 检查RDM特征
    print(f"\n=== RDM特征分析 ===")
    for i, session in enumerate(mo1s1_sessions):
        rdm = session['rdm']
        rdm_values = rdm[np.triu(np.ones_like(rdm, dtype=bool), k=1)]
        print(f"Session {session['session']} RDM:")
        print(f"  形状: {rdm.shape}")
        print(f"  值范围: {rdm_values.min():.4f} - {rdm_values.max():.4f}")
        print(f"  平均值: {rdm_values.mean():.4f}")
        print(f"  标准差: {rdm_values.std():.4f}")
    
    # 检查LLM RDM特征
    llm_values = llm_rdm[np.triu(np.ones_like(llm_rdm, dtype=bool), k=1)]
    print(f"\nLLM RDM:")
    print(f"  形状: {llm_rdm.shape}")
    print(f"  值范围: {llm_values.min():.4f} - {llm_values.max():.4f}")
    print(f"  平均值: {llm_values.mean():.4f}")
    print(f"  标准差: {llm_values.std():.4f}")
    
    # 可能的问题
    print(f"\n=== 可能的问题 ===")
    if noise_ceiling < 0.05:
        print("1. Noise ceiling很低，说明同一个arealabel的session间相似性很低")
    if np.mean(llm_rsas) > noise_ceiling * 2:
        print("2. 与LLM的RSA远大于noise ceiling，这在逻辑上不太合理")
        print("3. 可能的原因：")
        print("   - 数据质量问题")
        print("   - RDM计算有误")
        print("   - Session分组有误")
        print("   - LLM embedding与猴子数据不匹配")
    
    # 建议
    print(f"\n=== 建议 ===")
    print("1. 检查这些session是否真的应该归为同一个arealabel")
    print("2. 检查RDM计算是否正确")
    print("3. 考虑使用更保守的noise ceiling矫正方法")
    print("4. 或者直接使用原始RSA值，不进行noise ceiling矫正")

if __name__ == "__main__":
    diagnose_mo1s1()
