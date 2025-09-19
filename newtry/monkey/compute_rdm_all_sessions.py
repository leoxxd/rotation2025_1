"""
计算所有session的RDM（Representational Dissimilarity Matrix）

对每个session的1000张图片计算RDM，并保存结果
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

def extract_session_responses(data, session_num):
    """提取特定session的response数据"""
    if session_num not in data['extracted_data']:
        print(f"❌ Session {session_num} 不存在")
        return None
    
    session_data = data['extracted_data'][session_num]
    
    # 收集所有ROI的response数据
    all_responses = []
    roi_info = []
    
    for roi_index, roi_data in session_data['rois'].items():
        all_responses.append(roi_data['responses'])
        roi_info.append({
            'roi_index': roi_index,
            'arealabel': roi_data['arealabel'],
            'n_neurons': roi_data['n_neurons'],
            'y1': roi_data['y1'],
            'y2': roi_data['y2']
        })
    
    # 合并所有ROI的response数据
    if all_responses:
        combined_responses = np.vstack(all_responses)
        return combined_responses, roi_info
    
    return None, None

def compute_rdm_for_session(data, session_num, method='correlation'):
    """计算特定session的RDM"""
    print(f"\n处理Session {session_num}...")
    
    # 提取response数据
    responses, roi_info = extract_session_responses(data, session_num)
    
    if responses is None:
        print(f"❌ Session {session_num} 没有有效数据")
        return None
    
    print(f"  数据形状: {responses.shape}")
    print(f"  ROI数量: {len(roi_info)}")
    
    # 计算RDM
    rdm = compute_rdm(responses, method=method)
    
    print(f"  RDM形状: {rdm.shape}")
    print(f"  RDM范围: {rdm.min():.3f} 到 {rdm.max():.3f}")
    
    return {
        'session_num': session_num,
        'rdm': rdm,
        'responses_shape': responses.shape,
        'roi_info': roi_info,
        'method': method
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

def plot_rdm(rdm, title="RDM", save_path=None):
    """绘制RDM"""
    plt.figure(figsize=(10, 8))
    plt.imshow(rdm, cmap='viridis', aspect='equal')
    plt.colorbar(label='Dissimilarity')
    plt.title(title)
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"RDM图已保存到: {save_path}")
    
    plt.show()

def compute_all_rdms(data, method='correlation', save_individual=True, save_combined=True):
    """计算所有session的RDM"""
    available_sessions = list(data['extracted_data'].keys())
    print(f"开始计算 {len(available_sessions)} 个session的RDM...")
    print(f"使用距离方法: {method}")
    
    all_rdms = {}
    session_info = []
    
    for session_num in available_sessions:
        # 计算单个session的RDM
        rdm_data = compute_rdm_for_session(data, session_num, method=method)
        
        if rdm_data is not None:
            all_rdms[session_num] = rdm_data
            session_info.append({
                'session_num': session_num,
                'rdm_shape': rdm_data['rdm'].shape,
                'responses_shape': rdm_data['responses_shape'],
                'n_rois': len(rdm_data['roi_info']),
                'total_neurons': rdm_data['responses_shape'][0]
            })
            
            # 保存单个session的RDM
            if save_individual:
                individual_file = f"rdm_session_{session_num}_{method}.pkl"
                save_rdm_data(rdm_data, individual_file)
    
    print(f"\n✅ 成功计算 {len(all_rdms)} 个session的RDM")
    
    # 保存所有RDM的汇总数据
    if save_combined:
        combined_data = {
            'all_rdms': all_rdms,
            'session_info': session_info,
            'method': method,
            'summary': {
                'n_sessions': len(all_rdms),
                'method': method,
                'rdm_shape': list(all_rdms.values())[0]['rdm'].shape if all_rdms else None
            }
        }
        
        combined_file = f"all_rdms_{method}.pkl"
        save_rdm_data(combined_data, combined_file)
        
        # 保存session信息为CSV
        import pandas as pd
        df = pd.DataFrame(session_info)
        csv_file = f"session_info_{method}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ Session信息已保存到: {csv_file}")
    
    return all_rdms, session_info

def analyze_rdm_similarity(all_rdms):
    """分析不同session间RDM的相似性"""
    print(f"\n=== RDM相似性分析 ===")
    
    sessions = list(all_rdms.keys())
    n_sessions = len(sessions)
    
    if n_sessions < 2:
        print("需要至少2个session才能进行相似性分析")
        return
    
    # 计算session间RDM的相关性
    similarity_matrix = np.zeros((n_sessions, n_sessions))
    
    for i, session_i in enumerate(sessions):
        for j, session_j in enumerate(sessions):
            if i <= j:  # 只计算上三角矩阵
                rdm_i = all_rdms[session_i]['rdm']
                rdm_j = all_rdms[session_j]['rdm']
                
                # 计算上三角矩阵的相关性（排除对角线）
                mask = np.triu(np.ones_like(rdm_i, dtype=bool), k=1)
                corr, _ = spearmanr(rdm_i[mask], rdm_j[mask])
                similarity_matrix[i, j] = corr
                similarity_matrix[j, i] = corr  # 对称矩阵
    
    print(f"Session间RDM相似性矩阵:")
    print(f"  平均相似性: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean():.3f}")
    print(f"  相似性范围: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].min():.3f} 到 {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].max():.3f}")
    
    # 绘制相似性矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Spearman Correlation')
    plt.title('RDM Similarity Between Sessions')
    plt.xlabel('Session')
    plt.ylabel('Session')
    
    # 添加数值标签
    for i in range(n_sessions):
        for j in range(n_sessions):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('rdm_similarity_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return similarity_matrix

def main():
    """主函数"""
    print("=== 计算所有Session的RDM ===")
    
    # 加载数据
    data = load_pkl_data()
    if data is None:
        return
    
    # 设置参数
    method = 'correlation'  # 可选: 'correlation', 'euclidean', 'cosine'
    save_individual = True  # 是否保存单个session的RDM
    save_combined = True    # 是否保存汇总数据
    
    # 计算所有RDM
    all_rdms, session_info = compute_all_rdms(
        data, 
        method=method, 
        save_individual=save_individual, 
        save_combined=save_combined
    )
    
    # 分析RDM相似性
    if len(all_rdms) > 1:
        similarity_matrix = analyze_rdm_similarity(all_rdms)
    
    # 显示使用说明
    print(f"\n=== 如何使用保存的RDM数据 ===")
    print(f"# 加载单个session的RDM")
    print(f"import pickle")
    print(f"with open('rdm_session_1_{method}.pkl', 'rb') as f:")
    print(f"    rdm_data = pickle.load(f)")
    print(f"rdm = rdm_data['rdm']  # 形状: {list(all_rdms.values())[0]['rdm'].shape if all_rdms else 'N/A'}")
    print(f"")
    print(f"# 加载所有RDM")
    print(f"with open('all_rdms_{method}.pkl', 'rb') as f:")
    print(f"    all_rdms_data = pickle.load(f)")
    print(f"all_rdms = all_rdms_data['all_rdms']")
    print(f"")
    print(f"# 访问特定session的RDM")
    print(f"session_1_rdm = all_rdms[1]['rdm']")

if __name__ == "__main__":
    main()
