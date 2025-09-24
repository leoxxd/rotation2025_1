"""
计算所有session的RDM（Representational Dissimilarity Matrix）- z-score归一化版本

对每个session的每个arealabel的1000张图片的神经元响应进行z-score归一化，然后计算RDM，并保存结果
"""

import pickle
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def load_pkl_data(pkl_file="extracted_monkey_responses.pkl"):
    """加载pkl数据"""
    if not os.path.exists(pkl_file):
        print(f"❌ 文件不存在: {pkl_file}")
        return None
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 成功加载数据文件: {pkl_file}")
        return data
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None

def zscore_normalize_responses(responses):
    """
    对神经元响应进行z-score归一化
    
    Args:
        responses: 神经元响应数据 [n_neurons, n_images]
        
    Returns:
        normalized_responses: 归一化后的响应数据
    """
    if responses.shape[0] == 0:
        return responses
    
    # 对每个神经元（行）在1000张图片上进行z-score归一化
    mean = np.mean(responses, axis=1, keepdims=True)
    std = np.std(responses, axis=1, keepdims=True)
    
    # 避免除以零
    std[std == 0] = 1e-9
    
    normalized_responses = (responses - mean) / std
    
    return normalized_responses

def compute_rdm(responses, method='correlation'):
    """
    计算RDM
    
    Args:
        responses: response数据 (n_neurons, n_images)
        method: 距离计算方法 ('correlation', 'euclidean', 'cosine')
        
    Returns:
        rdm: RDM矩阵 (n_images, n_images)
    """
    if method == 'correlation':
        # 使用1 - 相关系数作为距离
        corr_matrix = np.corrcoef(responses.T)  # 转置以计算图片间的相关性
        rdm = 1 - corr_matrix
    elif method == 'euclidean':
        # 使用欧几里得距离
        distances = pdist(responses.T, metric='euclidean')
        rdm = squareform(distances)
    elif method == 'cosine':
        # 使用余弦距离
        distances = pdist(responses.T, metric='cosine')
        rdm = squareform(distances)
    else:
        raise ValueError(f"不支持的距离方法: {method}")
    
    return rdm

def compute_rdm_for_session_arealabel(data, session_num, arealabel, method='correlation'):
    """计算特定session特定arealabel的RDM（z-score归一化版本）"""
    print(f"  处理Session {session_num} - {arealabel}...")
    
    if session_num not in data['extracted_data']:
        print(f"    ❌ Session {session_num} 不存在")
        return None
    
    session_data = data['extracted_data'][session_num]
    
    # 找到对应的ROI数据
    roi_data = None
    for roi_index, roi_info in session_data['rois'].items():
        if roi_info['arealabel'] == arealabel:
            roi_data = roi_info
            break
    
    if roi_data is None:
        print(f"    ❌ Session {session_num} 中没有 {arealabel} 数据")
        return None
    
    responses = roi_data['responses']  # [n_neurons, 1000]
    print(f"   原始数据形状: {responses.shape}")
    
    # 进行z-score归一化
    normalized_responses = zscore_normalize_responses(responses)
    print(f"   归一化后形状: {normalized_responses.shape}")
    
    # 计算RDM
    rdm = compute_rdm(normalized_responses, method=method)
    
    print(f"   RDM形状: {rdm.shape}")
    print(f"   RDM范围: {rdm.min():.3f} 到 {rdm.max():.3f}")
    
    return {
        'session_num': session_num,
        'arealabel': arealabel,
        'rdm': rdm,
        'responses_shape': responses.shape,
        'normalized_responses_shape': normalized_responses.shape,
        'n_neurons': roi_data['n_neurons'],
        'method': method,
        'normalization': 'zscore_per_neuron_across_images'
    }

def save_rdm_data(rdm_data, output_file):
    """保存RDM数据"""
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(rdm_data, f)
        print(f"✅ RDM数据已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def compute_all_rdms_zscore(data, method='correlation', save_individual=True, save_combined=True):
    """计算所有session所有arealabel的RDM（z-score归一化版本）"""
    print(f"开始计算所有session所有arealabel的RDM（z-score归一化版本）...")
    print(f"使用距离方法: {method}")
    print(f"归一化方法: z-score per neuron across images")
    
    # 收集所有arealabel
    all_arealabels = set()
    for session_num, session_data in data['extracted_data'].items():
        for roi_index, roi_data in session_data['rois'].items():
            all_arealabels.add(roi_data['arealabel'])
    
    all_arealabels = sorted(list(all_arealabels))
    print(f"找到的arealabel: {all_arealabels}")
    
    all_rdms = {}
    session_info = []
    
    for session_num in data['extracted_data'].keys():
        session_rdms = {}
        for arealabel in all_arealabels:
            rdm_data = compute_rdm_for_session_arealabel(data, session_num, arealabel, method=method)
            if rdm_data is not None:
                session_rdms[arealabel] = rdm_data
                
                # 保存单个session单个arealabel的RDM
                if save_individual:
                    individual_file = f"rdm_session_{session_num}_{arealabel}_{method}_zscore.pkl"
                    save_rdm_data(rdm_data, individual_file)
        
        if session_rdms:
            all_rdms[session_num] = session_rdms
            session_info.append({
                'session_num': session_num,
                'n_arealabels': len(session_rdms),
                'arealabels': list(session_rdms.keys())
            })
    
    print(f"\n✅ 成功计算 {len(all_rdms)} 个session的RDM（z-score归一化版本）")
    
    # 保存所有RDM的汇总数据
    if save_combined:
        combined_data = {
            'all_rdms': all_rdms,
            'session_info': session_info,
            'method': method,
            'normalization': 'zscore_per_neuron_across_images',
            'arealabels': all_arealabels,
            'summary': {
                'n_sessions': len(all_rdms),
                'n_arealabels': len(all_arealabels),
                'method': method,
                'normalization': 'zscore_per_neuron_across_images'
            }
        }
        
        combined_file = f"all_rdms_{method}_zscore.pkl"
        save_rdm_data(combined_data, combined_file)
        
        # 保存session信息为CSV
        import pandas as pd
        df = pd.DataFrame(session_info)
        csv_file = f"session_info_{method}_zscore.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ Session信息已保存到: {csv_file}")
    
    return all_rdms, session_info

def main():
    """主函数"""
    print("=== 计算所有Session所有Arealabel的RDM（z-score归一化版本）===")
    
    # 加载数据
    data = load_pkl_data()
    if data is None:
        return
    
    # 设置参数
    method = 'correlation'  # 可选: 'correlation', 'euclidean', 'cosine'
    save_individual = False  # 是否保存单个session单个arealabel的RDM（建议False，节省空间）
    save_combined = True    # 是否保存汇总数据
    
    # 计算所有RDM
    all_rdms, session_info = compute_all_rdms_zscore(
        data, 
        method=method, 
        save_individual=save_individual, 
        save_combined=save_combined
    )
    
    # 显示使用说明
    print(f"\n=== 如何使用保存的RDM数据（z-score归一化版本）===")
    print(f"# 加载所有RDM")
    print(f"import pickle")
    print(f"with open('all_rdms_{method}_zscore.pkl', 'rb') as f:")
    print(f"    all_rdms_data = pickle.load(f)")
    print(f"all_rdms = all_rdms_data['all_rdms']")
    print(f"")
    print(f"# 访问特定session特定arealabel的RDM")
    print(f"session_1_mb1_rdm = all_rdms[1]['MB1']['rdm']  # 形状: (1000, 1000)")
    print(f"")
    print(f"# 归一化信息")
    print(f"normalization = all_rdms_data['normalization']  # 'zscore_per_neuron_across_images'")

if __name__ == "__main__":
    main()