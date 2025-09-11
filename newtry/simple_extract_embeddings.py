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

# 设置Hugging Face缓存目录到当前项目目录
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像


def main():
    # 配置参数
    input_file = r"e:\lunzhuan1\rotation2025\Stim\Anno_Shared1000.txt"
    output_dir = "./embeddings_output"
    model_name = "all-mpnet-base-v2"
    
    print("开始处理Anno_Shared1000.txt文件...")
    print("处理方式：每张图像的5个caption分别生成embedding，然后平均得到图像embedding")
    
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据并按图像分组
    print("正在加载数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 按图像分组存储数据
    image_data = {}  # {image_id: [caption1, caption2, caption3, caption4, caption5]}
    
    for line in lines:
        line = line.strip()
        if line:
            try:
                # 解析每行的数据
                image_group = eval(line)
                for item in image_group:
                    image_id = item['image_id']
                    caption = item['caption']
                    
                    if image_id not in image_data:
                        image_data[image_id] = []
                    image_data[image_id].append(caption)
            except Exception as e:
                print(f"解析行时出错: {e}")
                continue
    
    print(f"成功加载 {len(image_data)} 张图像")
    print(f"每张图像平均有 {np.mean([len(captions) for captions in image_data.values()]):.1f} 个caption")
    
    # 3. 加载mpnet模型
    print(f"正在加载模型: {model_name}")
    
    # 尝试使用本地模型路径
    local_model_path = "./models/all-mpnet-base-v2"
    
    try:
        if os.path.exists(local_model_path):
            print(f"使用本地模型: {local_model_path}")
            model = SentenceTransformer(local_model_path)
        else:
            print("尝试从网络下载模型...")
            model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请尝试以下解决方案:")
        print("1. 设置环境变量: set HF_ENDPOINT=https://hf-mirror.com")
        print("2. 手动下载模型到 ./models/all-mpnet-base-v2/ 目录")
        print("3. 使用其他embedding模型")
        return
    
    # 4. 为每张图像生成平均embedding
    print("开始生成图像embeddings...")
    image_embeddings = []
    image_ids_list = []
    all_captions_per_image = []
    
    for image_id, captions in tqdm(image_data.items(), desc="处理图像"):
        # 为这张图像的5个caption生成embeddings
        caption_embeddings = model.encode(captions, convert_to_numpy=True)
        
        # 计算平均embedding
        mean_embedding = np.mean(caption_embeddings, axis=0)
        
        # 存储结果
        image_embeddings.append(mean_embedding)
        image_ids_list.append(image_id)
        all_captions_per_image.append(captions)
    
    # 转换为numpy数组
    image_embeddings = np.array(image_embeddings)
    
    # 5. 保存结果
    print("保存结果...")
    
    # 保存图像embeddings
    np.save(os.path.join(output_dir, 'image_embeddings.npy'), image_embeddings)
    
    # 保存元数据
    metadata = {
        'image_ids': image_ids_list,
        'captions_per_image': all_captions_per_image,  # 每张图像的所有caption
        'embedding_shape': image_embeddings.shape,
        'model_name': model_name,
        'processing_method': 'average_of_5_captions_per_image'
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # 6. 打印结果
    print("\n" + "="*60)
    print("处理完成！")
    print(f"图像数量: {len(image_data)}")
    print(f"每张图像embedding维度: {image_embeddings.shape[1]}")
    print(f"总embedding数组形状: {image_embeddings.shape}")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    # 7. 显示一些示例
    print("\n前3张图像的caption示例:")
    for i in range(min(3, len(image_ids_list))):
        print(f"\n图像ID: {image_ids_list[i]}")
        print("Captions:")
        for j, caption in enumerate(all_captions_per_image[i]):
            print(f"  {j+1}. {caption}")
        print(f"平均embedding形状: {image_embeddings[i].shape}")
        print(f"平均embedding前5个值: {image_embeddings[i][:5]}")


if __name__ == "__main__":
    main()
