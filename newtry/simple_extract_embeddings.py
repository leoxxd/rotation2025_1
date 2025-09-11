#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版Caption Embedding提取脚本
专门处理Anno_Shared1000.txt文件
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    # 配置参数
    input_file = r"e:\lunzhuan1\rotation2025\Stim\Anno_Shared1000.txt"
    output_dir = "./embeddings_output"
    model_name = "all-mpnet-base-v2"
    
    print("开始处理Anno_Shared1000.txt文件...")
    
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("正在加载数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_captions = []
    image_ids = []
    caption_ids = []
    
    for line in lines:
        line = line.strip()
        if line:
            try:
                # 解析每行的数据
                image_group = eval(line)
                for item in image_group:
                    all_captions.append(item['caption'])
                    image_ids.append(item['image_id'])
                    caption_ids.append(item['id'])
            except Exception as e:
                print(f"解析行时出错: {e}")
                continue
    
    print(f"成功加载 {len(all_captions)} 个caption")
    
    # 3. 加载mpnet模型
    print(f"正在加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    
    # 4. 生成embeddings
    print("开始生成embeddings...")
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(all_captions), batch_size), desc="处理中"):
        batch = all_captions[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    
    # 合并所有embeddings
    embeddings = np.vstack(embeddings)
    
    # 5. 保存结果
    print("保存结果...")
    
    # 保存embeddings
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
    
    # 保存元数据
    metadata = {
        'captions': all_captions,
        'image_ids': image_ids,
        'caption_ids': caption_ids,
        'embedding_shape': embeddings.shape,
        'model_name': model_name
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # 6. 打印结果
    print("\n" + "="*50)
    print("处理完成！")
    print(f"总caption数量: {len(all_captions)}")
    print(f"Embedding维度: {embeddings.shape[1]}")
    print(f"唯一图像数量: {len(set(image_ids))}")
    print(f"输出目录: {output_dir}")
    print("="*50)
    
    # 7. 显示一些示例
    print("\n前5个caption示例:")
    for i in range(min(5, len(all_captions))):
        print(f"{i+1}. {all_captions[i]}")


if __name__ == "__main__":
    main()
