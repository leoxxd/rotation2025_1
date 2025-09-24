#!/usr/bin/env python3
"""
检查embedding文件的内容和质量
"""

import pickle
import numpy as np

def check_embeddings():
    print("=== 检查Embedding文件 ===")
    
    # 检查名词embeddings
    print("\n1. 名词Embeddings:")
    noun_emb = np.load('../captions/embeddings_output/noun_embeddings.npy')
    print(f"   形状: {noun_emb.shape}")
    print(f"   均值: {noun_emb.mean():.6f}")
    print(f"   标准差: {noun_emb.std():.6f}")
    print(f"   最小值: {noun_emb.min():.6f}")
    print(f"   最大值: {noun_emb.max():.6f}")
    
    # 检查动词embeddings
    print("\n2. 动词Embeddings:")
    verb_emb = np.load('../captions/embeddings_output/verb_embeddings.npy')
    print(f"   形状: {verb_emb.shape}")
    print(f"   均值: {verb_emb.mean():.6f}")
    print(f"   标准差: {verb_emb.std():.6f}")
    print(f"   最小值: {verb_emb.min():.6f}")
    print(f"   最大值: {verb_emb.max():.6f}")
    
    # 检查完整caption embeddings
    print("\n3. 完整Caption Embeddings:")
    image_emb = np.load('../captions/embeddings_output/image_embeddings.npy')
    print(f"   形状: {image_emb.shape}")
    print(f"   均值: {image_emb.mean():.6f}")
    print(f"   标准差: {image_emb.std():.6f}")
    print(f"   最小值: {image_emb.min():.6f}")
    print(f"   最大值: {image_emb.max():.6f}")
    
    # 检查名词列表
    print("\n4. 名词列表:")
    with open('../captions/embeddings_output/noun_lists.pkl', 'rb') as f:
        noun_lists = pickle.load(f)
    print(f"   总图像数: {len(noun_lists)}")
    print(f"   前3个图像的名词:")
    for i in range(3):
        print(f"     图像 {i}: {noun_lists[i][:10]}")
    print(f"   平均每图像名词数: {sum(len(n) for n in noun_lists) / len(noun_lists):.2f}")
    
    # 检查动词列表
    print("\n5. 动词列表:")
    with open('../captions/embeddings_output/verb_lists.pkl', 'rb') as f:
        verb_lists = pickle.load(f)
    print(f"   总图像数: {len(verb_lists)}")
    print(f"   前3个图像的动词:")
    for i in range(3):
        print(f"     图像 {i}: {verb_lists[i][:10]}")
    print(f"   平均每图像动词数: {sum(len(v) for v in verb_lists) / len(verb_lists):.2f}")
    
    # 检查是否有空列表
    print("\n6. 空列表统计:")
    empty_nouns = sum(1 for n in noun_lists if len(n) == 0)
    empty_verbs = sum(1 for v in verb_lists if len(v) == 0)
    print(f"   无名词的图像数: {empty_nouns}")
    print(f"   无动词的图像数: {empty_verbs}")
    
    # 检查embedding的相似性
    print("\n7. Embedding相似性:")
    # 计算名词和动词embedding的相关性
    noun_verb_corr = np.corrcoef(noun_emb.flatten(), verb_emb.flatten())[0, 1]
    print(f"   名词-动词embedding相关性: {noun_verb_corr:.6f}")
    
    # 计算与完整caption embedding的相关性
    noun_image_corr = np.corrcoef(noun_emb.flatten(), image_emb.flatten())[0, 1]
    verb_image_corr = np.corrcoef(verb_emb.flatten(), image_emb.flatten())[0, 1]
    print(f"   名词-完整caption相关性: {noun_image_corr:.6f}")
    print(f"   动词-完整caption相关性: {verb_image_corr:.6f}")

if __name__ == "__main__":
    check_embeddings()
