#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试词性标注Embedding生成器
"""

import os
import pickle
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置Hugging Face缓存目录到当前项目目录
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def test_pos_extraction():
    """测试词性提取功能"""
    print("测试词性提取功能...")
    
    # 下载必要的NLTK数据
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("下载NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("下载NLTK averaged_perceptron_tagger...")
        nltk.download('averaged_perceptron_tagger')
    
    # 测试句子
    test_sentences = [
        "A person is walking on the street",
        "The dog is running in the park",
        "A beautiful sunset over the ocean"
    ]
    
    for sentence in test_sentences:
        print(f"\n测试句子: {sentence}")
        
        # 分词和词性标注
        tokens = nltk.word_tokenize(sentence.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        print(f"  分词结果: {tokens}")
        print(f"  词性标注: {pos_tags}")
        
        # 提取名词和动词
        nouns = []
        verbs = []
        
        for word, pos in pos_tags:
            if len(word) > 1 and word.isalpha():
                if pos.startswith('NN'):
                    nouns.append(word)
                elif pos.startswith('VB'):
                    verbs.append(word)
        
        print(f"  名词: {nouns}")
        print(f"  动词: {verbs}")

def test_embedding_generation():
    """测试embedding生成功能"""
    print("\n测试embedding生成功能...")
    
    # 加载模型
    print("加载模型...")
    local_model_path = "../models/all-mpnet-base-v2"
    
    try:
        if os.path.exists(local_model_path):
            print(f"使用本地模型: {local_model_path}")
            model = SentenceTransformer(local_model_path)
        else:
            print("尝试从网络下载模型...")
            model = SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    print(f"模型加载完成，embedding维度: {model.get_sentence_embedding_dimension()}")
    
    # 测试单词embedding
    test_words = ["person", "walking", "dog", "running", "sunset"]
    
    print(f"\n测试单词embedding:")
    for word in test_words:
        try:
            embedding = model.encode([word], convert_to_tensor=False)[0]
            print(f"  {word}: {embedding.shape}, 范围: {embedding.min():.3f} ~ {embedding.max():.3f}")
        except Exception as e:
            print(f"  {word}: 失败 - {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("词性标注Embedding测试")
    print("=" * 60)
    
    # 测试词性提取
    test_pos_extraction()
    
    # 测试embedding生成
    test_embedding_generation()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
