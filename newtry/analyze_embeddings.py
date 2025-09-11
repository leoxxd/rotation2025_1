#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析生成的embedding文件
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def analyze_embeddings():
    """分析embedding文件"""
    
    # 1. 读取数据
    print("读取embedding数据...")
    embeddings = np.load('./embeddings_output/image_embeddings.npy')
    
    with open('./embeddings_output/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # 2. 基本信息
    print("\n" + "="*50)
    print("基本信息:")
    print(f"图像数量: {embeddings.shape[0]}")
    print(f"Embedding维度: {embeddings.shape[1]}")
    print(f"数据类型: {embeddings.dtype}")
    print(f"内存大小: {embeddings.nbytes / (1024*1024):.2f} MB")
    print("="*50)
    
    # 3. 统计信息
    print("\n统计信息:")
    print(f"最小值: {embeddings.min():.4f}")
    print(f"最大值: {embeddings.max():.4f}")
    print(f"平均值: {embeddings.mean():.4f}")
    print(f"标准差: {embeddings.std():.4f}")
    
    # 4. 查看前几张图像的信息
    print("\n前3张图像信息:")
    for i in range(min(3, len(metadata['image_ids']))):
        print(f"\n图像 {i+1}:")
        print(f"  图像ID: {metadata['image_ids'][i]}")
        print(f"  Captions:")
        for j, caption in enumerate(metadata['captions_per_image'][i]):
            print(f"    {j+1}. {caption}")
        print(f"  Embedding形状: {embeddings[i].shape}")
        print(f"  Embedding前5个值: {embeddings[i][:5]}")
    
    return embeddings, metadata

def visualize_embeddings(embeddings, metadata, n_samples=100):
    """可视化embeddings"""
    
    print(f"\n生成可视化图（使用前{n_samples}个样本）...")
    
    # 随机选择样本
    indices = np.random.choice(len(embeddings), min(n_samples, len(embeddings)), replace=False)
    sample_embeddings = embeddings[indices]
    sample_captions = [metadata['captions_per_image'][i][0] for i in indices]  # 取第一个caption
    
    # PCA降维
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(sample_embeddings)
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # 主图
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, c=range(len(pca_result)), cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'PCA Visualization (前{n_samples}个样本)')
    plt.xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    
    # 添加一些caption标签
    for i in range(0, len(sample_captions), max(1, len(sample_captions)//10)):
        plt.annotate(sample_captions[i][:30] + '...' if len(sample_captions[i]) > 30 else sample_captions[i], 
                    (pca_result[i, 0], pca_result[i, 1]), 
                    fontsize=8, alpha=0.8)
    
    # Embedding分布
    plt.subplot(2, 2, 2)
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7)
    plt.title('Embedding值分布')
    plt.xlabel('值')
    plt.ylabel('频次')
    
    # 每张图像embedding的L2范数
    plt.subplot(2, 2, 3)
    norms = np.linalg.norm(embeddings, axis=1)
    plt.hist(norms, bins=30, alpha=0.7)
    plt.title('每张图像embedding的L2范数分布')
    plt.xlabel('L2范数')
    plt.ylabel('频次')
    
    # 维度重要性（前20个维度）
    plt.subplot(2, 2, 4)
    dim_variance = np.var(embeddings, axis=0)
    top_dims = np.argsort(dim_variance)[-20:]
    plt.bar(range(20), dim_variance[top_dims])
    plt.title('前20个最重要维度的方差')
    plt.xlabel('维度排名')
    plt.ylabel('方差')
    
    plt.tight_layout()
    plt.savefig('./embeddings_output/embedding_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 可视化图已保存到: ./embeddings_output/embedding_analysis.png")

def find_similar_images(embeddings, metadata, query_idx=0, top_k=5):
    """找到相似的图像"""
    
    print(f"\n查找与图像 {query_idx} 最相似的图像...")
    
    # 计算余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_embedding = embeddings[query_idx:query_idx+1]
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # 找到最相似的图像（排除自己）
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    print(f"\n查询图像 {query_idx}:")
    print(f"图像ID: {metadata['image_ids'][query_idx]}")
    print("Captions:")
    for i, caption in enumerate(metadata['captions_per_image'][query_idx]):
        print(f"  {i+1}. {caption}")
    
    print(f"\n最相似的 {top_k} 张图像:")
    for i, idx in enumerate(similar_indices):
        similarity = similarities[idx]
        print(f"\n{i+1}. 图像 {idx} (相似度: {similarity:.4f})")
        print(f"   图像ID: {metadata['image_ids'][idx]}")
        print("   Captions:")
        for j, caption in enumerate(metadata['captions_per_image'][idx]):
            print(f"     {j+1}. {caption}")

def main():
    """主函数"""
    
    print("开始分析embedding文件...")
    
    # 检查文件是否存在
    if not os.path.exists('./embeddings_output/image_embeddings.npy'):
        print("❌ 找不到embedding文件，请先运行主脚本生成embeddings")
        return
    
    # 分析数据
    embeddings, metadata = analyze_embeddings()
    
    # 可视化
    visualize_embeddings(embeddings, metadata)
    
    # 查找相似图像
    find_similar_images(embeddings, metadata)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
