#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用FastText生成名词embeddings
完全复制原始NSD项目的方法
"""

import os
import sys
import pickle
import numpy as np
import nltk
from collections import defaultdict

# 添加src路径以导入原始项目的函数
sys.path.append('../../src')
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import load_word_vectors, get_word_embedding
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import get_word_type_from_string

class FastTextNounEmbeddingGenerator:
    def __init__(self, fasttext_path=None, caption_file="../captions/nsd_captions.txt"):
        """
        初始化FastText名词embedding生成器
        
        Args:
            fasttext_path: fasttext模型文件路径
            caption_file: caption文件路径
        """
        self.caption_file = caption_file
        self.fasttext_path = fasttext_path or self._get_fasttext_path()
        self.embeddings = None
        self.EMBEDDING_TYPE = 'fasttext'
        
        # 加载fasttext模型
        self.load_fasttext_model()
    
    def _get_fasttext_path(self):
        """获取fasttext模型路径"""
        # 尝试多个可能的路径
        possible_paths = [
            "../../data/fasttext/crawl-300d-2M.vec",
            "../../data/fasttext/wiki-news-300d-1M.vec", 
            "../../data/fasttext/wiki-news-300d-1M.vec.zip",
            "fasttext/crawl-300d-2M.vec",
            "fasttext/wiki-news-300d-1M.vec"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果都不存在，返回默认路径
        return "fasttext/crawl-300d-2M.vec"
    
    def load_fasttext_model(self):
        """加载fasttext模型"""
        print(f"正在加载FastText模型: {self.fasttext_path}")
        
        if not os.path.exists(self.fasttext_path):
            print(f"FastText模型文件不存在: {self.fasttext_path}")
            print("请下载FastText模型:")
            print("wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip")
            print("unzip crawl-300d-2M.vec.zip")
            raise FileNotFoundError(f"FastText模型文件不存在: {self.fasttext_path}")
        
        try:
            self.embeddings = load_word_vectors(self.fasttext_path, 'fasttext')
            print(f"FastText模型加载成功，词汇量: {len(self.embeddings)}")
        except Exception as e:
            print(f"FastText模型加载失败: {e}")
            raise e
    
    def load_captions(self):
        """加载caption文件 - 与原始项目完全一致"""
        print(f"正在加载数据: {self.caption_file}")
        
        if not os.path.exists(self.caption_file):
            raise FileNotFoundError(f"Caption文件不存在: {self.caption_file}")
        
        # 按图像分组存储数据
        image_data = {}  # {image_id: [caption1, caption2, caption3, caption4, caption5]}
        
        with open(self.caption_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
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
        
        # 转换为列表格式
        captions = []
        for image_id in sorted(image_data.keys()):
            captions.append(image_data[image_id])
        
        print(f"成功加载 {len(captions)} 张图像")
        print(f"每张图像平均有 {np.mean([len(c) for c in captions]):.1f} 个caption")
        
        return captions
    
    def get_nouns_from_string(self, s):
        """
        从字符串中提取名词 - 使用原始项目的方法
        """
        return get_word_type_from_string(s, 'noun')
    
    def get_word_embedding(self, word):
        """
        获取单个单词的embedding - 使用原始项目的方法
        """
        return get_word_embedding(word, self.embeddings, self.EMBEDDING_TYPE)
    
    def generate_noun_embeddings(self, captions, normalize_embeddings=True):
        """
        生成名词embeddings - 完全复制原始项目的方法
        
        Args:
            captions: caption列表，每个元素是一个图片的多句caption列表
            normalize_embeddings: 是否对embedding进行z-score归一化
            
        Returns:
            noun_embeddings: 名词embeddings数组
            noun_lists: 每个图片的名词列表
        """
        print("开始生成名词embeddings...")
        
        n_images = len(captions)
        # 获取embedding维度
        sample_embedding = self.get_word_embedding("cat")
        embedding_dim = len(sample_embedding)
        
        # 初始化结果数组
        noun_embeddings = np.zeros((n_images, embedding_dim))
        noun_lists = []
        
        # 统计信息
        total_nouns = 0
        no_nouns_count = 0
        
        for i, image_captions in enumerate(captions):
            if i % 100 == 0:
                print(f"处理进度: {i}/{n_images} ({i/n_images*100:.1f}%)")
            
            # 收集该图片所有caption的名词
            all_nouns = []
            
            for j, caption in enumerate(image_captions):
                nouns = self.get_nouns_from_string(caption)
                all_nouns.extend(nouns)
            
            # 保留重复词：与原始代码保持一致，保留词频信息
            noun_lists.append(all_nouns)
            
            # 处理名词
            noun_word_embeddings = []
            for noun in all_nouns:
                try:
                    noun_emb = self.get_word_embedding(noun)
                    noun_word_embeddings.append(noun_emb)
                except Exception as e:
                    print(f"  ⚠️ 名词 '{noun}' 生成embedding失败: {e}")
                    continue
            
            if len(noun_word_embeddings) == 0:
                # 如果没有名词或所有名词embedding生成失败，使用"something"的embedding
                noun_embeddings[i] = self.get_word_embedding("something")
                no_nouns_count += 1
            else:
                noun_embeddings[i] = np.mean(noun_word_embeddings, axis=0)
                total_nouns += len(noun_word_embeddings)
        
        # 可选的embedding归一化
        if normalize_embeddings:
            print("\n对embeddings进行z-score归一化...")
            noun_mean = np.mean(noun_embeddings, axis=0)
            noun_std = np.std(noun_embeddings, axis=0)
            noun_embeddings = (noun_embeddings - noun_mean) / (noun_std + 1e-8)
            
            print(f"  名词embedding归一化: mean={np.mean(noun_embeddings):.6f}, std={np.std(noun_embeddings):.6f}")
        
        print(f"\n生成完成!")
        print(f"总名词数: {total_nouns}")
        print(f"无名词图片数: {no_nouns_count}")
        print(f"平均每张图片名词数: {total_nouns/n_images:.2f}")
        print(f"Embedding形状: {noun_embeddings.shape}")
        
        return noun_embeddings, noun_lists
    
    def run(self, normalize_embeddings=True):
        """运行完整的embedding生成流程"""
        print("=" * 60)
        print("FastText名词Embedding生成器")
        print("=" * 60)
        
        # 加载数据
        captions = self.load_captions()
        
        # 生成embeddings
        noun_embeddings, noun_lists = self.generate_noun_embeddings(captions, normalize_embeddings)
        
        # 保存结果
        output_dir = "embeddings_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存numpy数组
        np.save(os.path.join(output_dir, "noun_embeddings_fasttext.npy"), noun_embeddings)
        
        # 保存名词列表
        with open(os.path.join(output_dir, "noun_lists_fasttext.pkl"), 'wb') as f:
            pickle.dump(noun_lists, f)
        
        print(f"\n结果已保存到:")
        print(f"  - {output_dir}/noun_embeddings_fasttext.npy")
        print(f"  - {output_dir}/noun_lists_fasttext.pkl")
        
        return noun_embeddings, noun_lists

def main():
    """主函数"""
    # 检查NLTK数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("下载NLTK数据...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    
    # 创建生成器
    generator = FastTextNounEmbeddingGenerator()
    
    # 运行生成
    noun_embeddings, noun_lists = generator.run(normalize_embeddings=True)
    
    print("\n✅ FastText名词embedding生成完成!")

if __name__ == "__main__":
    main()

