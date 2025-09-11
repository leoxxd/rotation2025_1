#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看第一张图像的embedding
"""

import numpy as np
import pickle
import sys

def view_first_embedding():
    """查看第一张图像的embedding"""
    
    try:
        # 读取embeddings
        embeddings = np.load('./embeddings_output/image_embeddings.npy')
        
        # 读取元数据
        with open('./embeddings_output/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # 第一张图像的信息
        first_image_id = metadata['image_ids'][0]
        first_captions = metadata['captions_per_image'][0]
        first_embedding = embeddings[0]
        
        print("=" * 60)
        print("第一张图像的Embedding信息")
        print("=" * 60)
        
        print(f"图像ID: {first_image_id}")
        print(f"Caption数量: {len(first_captions)}")
        print("\n所有Captions:")
        for i, caption in enumerate(first_captions):
            print(f"  {i+1}. {caption}")
        
        print(f"\nEmbedding信息:")
        print(f"  形状: {first_embedding.shape}")
        print(f"  数据类型: {first_embedding.dtype}")
        print(f"  最小值: {first_embedding.min():.6f}")
        print(f"  最大值: {first_embedding.max():.6f}")
        print(f"  平均值: {first_embedding.mean():.6f}")
        print(f"  标准差: {first_embedding.std():.6f}")
        
        print(f"\n前20个embedding值:")
        for i in range(min(20, len(first_embedding))):
            print(f"  [{i:2d}]: {first_embedding[i]:8.6f}")
        
        if len(first_embedding) > 20:
            print(f"  ... (还有 {len(first_embedding) - 20} 个值)")
        
        print(f"\n后10个embedding值:")
        for i in range(max(0, len(first_embedding) - 10), len(first_embedding)):
            print(f"  [{i:2d}]: {first_embedding[i]:8.6f}")
        
        print("=" * 60)
        
    except FileNotFoundError:
        print("❌ 找不到embedding文件，请先运行主脚本生成embeddings")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    view_first_embedding()
