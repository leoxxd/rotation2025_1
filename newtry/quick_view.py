#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速查看embedding文件
"""

import numpy as np
import pickle
import os

def quick_view():
    """快速查看embedding文件"""
    
    if not os.path.exists('./embeddings_output/image_embeddings.npy'):
        print("❌ 找不到embedding文件")
        return
    
    # 读取数据
    embeddings = np.load('./embeddings_output/image_embeddings.npy')
    
    with open('./embeddings_output/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print("📊 Embedding文件信息:")
    print(f"   形状: {embeddings.shape}")
    print(f"   数据类型: {embeddings.dtype}")
    print(f"   内存大小: {embeddings.nbytes / (1024*1024):.2f} MB")
    print(f"   图像数量: {len(metadata['image_ids'])}")
    
    print("\n📝 前3张图像的caption:")
    for i in range(min(3, len(metadata['image_ids']))):
        print(f"\n图像 {i+1} (ID: {metadata['image_ids'][i]}):")
        for j, caption in enumerate(metadata['captions_per_image'][i]):
            print(f"  {j+1}. {caption}")
    
    print(f"\n🔍 第一张图像的embedding:")
    print(f"   形状: {embeddings[0].shape}")
    print(f"   前10个值: {embeddings[0][:10]}")
    print(f"   最小值: {embeddings[0].min():.4f}")
    print(f"   最大值: {embeddings[0].max():.4f}")
    print(f"   平均值: {embeddings[0].mean():.4f}")

if __name__ == "__main__":
    quick_view()
