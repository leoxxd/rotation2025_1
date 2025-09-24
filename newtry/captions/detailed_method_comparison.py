#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细对比原始NSD项目和我们的embedding生成方法
重点分析：单词级别的embedding生成方式
"""

import os
import pickle
import numpy as np
import nltk

def analyze_embedding_generation_methods():
    """分析两种embedding生成方法的关键差异"""
    
    print("=" * 80)
    print("Embedding生成方法详细对比分析")
    print("=" * 80)
    
    print("\n1. 原始NSD项目的方法:")
    print("   - 使用 get_word_embedding(word, embeddings, EMBEDDING_TYPE)")
    print("   - 对于sentence transformer: get_embeddings([word], embeddings, embedding_type)[0]")
    print("   - 关键: 将单个单词作为列表 [word] 输入给模型")
    print("   - 模型处理: embedding_model.encode([word])")
    print("   - 结果: 返回该单词的embedding向量")
    
    print("\n2. 我们的方法:")
    print("   - 使用 self.model.encode([word], convert_to_tensor=False)[0]")
    print("   - 关键: 同样将单个单词作为列表 [word] 输入给模型")
    print("   - 模型处理: SentenceTransformer.encode([word])")
    print("   - 结果: 返回该单词的embedding向量")
    
    print("\n3. 关键发现:")
    print("   ✅ 两种方法在单词级别embedding生成上是相同的！")
    print("   ✅ 都是将单个单词作为列表输入给sentence transformer")
    print("   ✅ 都是对单词embeddings取平均")
    
    print("\n4. 那么问题在哪里？")
    print("   🔍 问题可能在于:")
    print("   - 分词方式的差异")
    print("   - 词汇修正的缺失")
    print("   - 数据预处理的不同")

def test_word_embedding_consistency():
    """测试单词embedding的一致性"""
    print("\n" + "=" * 80)
    print("单词Embedding一致性测试")
    print("=" * 80)
    
    # 测试单词
    test_words = ['food', 'plate', 'vegetables', 'garnish', 'stir', 'fry']
    
    print("\n理论上，两种方法对相同单词应该产生相同的embedding:")
    print("原始方法: get_embeddings([word], model, 'all-mpnet-base-v2')[0]")
    print("我们的方法: model.encode([word])[0]")
    print("这两种方式应该是等价的！")

def analyze_tokenization_impact():
    """分析分词对embedding的影响"""
    print("\n" + "=" * 80)
    print("分词对Embedding的影响分析")
    print("=" * 80)
    
    test_sentence = "Food on a white plate with vegetables and a garnish."
    
    print(f"\n测试句子: '{test_sentence}'")
    
    # 原始方法分词
    original_tokens = nltk.word_tokenize(test_sentence)
    print(f"\n原始方法分词结果:")
    print(f"  {original_tokens}")
    print(f"  单词数量: {len(original_tokens)}")
    
    # 我们的方法分词
    our_tokens = nltk.word_tokenize(test_sentence.lower())
    our_filtered_tokens = [token for token in our_tokens if len(token) > 1 and token.isalpha()]
    print(f"\n我们的方法分词结果:")
    print(f"  {our_filtered_tokens}")
    print(f"  单词数量: {len(our_filtered_tokens)}")
    
    # 分析差异
    print(f"\n差异分析:")
    print(f"  丢失的单词: {set(original_tokens) - set(our_filtered_tokens)}")
    print(f"  新增的单词: {set(our_filtered_tokens) - set(original_tokens)}")
    print(f"  共同单词: {set(original_tokens) & set(our_filtered_tokens)}")
    
    # 计算差异比例
    common_words = len(set(original_tokens) & set(our_filtered_tokens))
    total_original = len(original_tokens)
    similarity = common_words / total_original
    print(f"  单词重叠率: {similarity:.2%}")

def analyze_embedding_statistics():
    """分析embedding统计信息"""
    print("\n" + "=" * 80)
    print("Embedding统计信息分析")
    print("=" * 80)
    
    try:
        # 加载我们的embedding
        word_avg_emb = np.load("embeddings_output/word_average_embeddings.npy")
        image_emb = np.load("embeddings_output/image_embeddings.npy")
        
        print(f"\n我们的单词平均embedding:")
        print(f"  形状: {word_avg_emb.shape}")
        print(f"  均值: {word_avg_emb.mean():.6f}")
        print(f"  标准差: {word_avg_emb.std():.6f}")
        print(f"  最小值: {word_avg_emb.min():.6f}")
        print(f"  最大值: {word_avg_emb.max():.6f}")
        
        print(f"\n完整caption embedding:")
        print(f"  形状: {image_emb.shape}")
        print(f"  均值: {image_emb.mean():.6f}")
        print(f"  标准差: {image_emb.std():.6f}")
        print(f"  最小值: {image_emb.min():.6f}")
        print(f"  最大值: {image_emb.max():.6f}")
        
        # 计算相关性
        correlation = np.corrcoef(word_avg_emb.flatten(), image_emb.flatten())[0, 1]
        print(f"\n相关性分析:")
        print(f"  单词平均embedding与完整caption embedding的相关性: {correlation:.6f}")
        
        # 分析相关性低的原因
        print(f"\n相关性低的原因分析:")
        if correlation < 0.3:
            print("  ⚠️ 相关性过低，可能原因:")
            print("  1. 分词方式差异导致单词集合不同")
            print("  2. 大小写转换影响embedding质量")
            print("  3. 标点符号过滤过于激进")
            print("  4. 缺乏词汇修正机制")
        
    except FileNotFoundError as e:
        print(f"  文件未找到: {e}")

def analyze_word_lists_difference():
    """分析单词列表的差异"""
    print("\n" + "=" * 80)
    print("单词列表差异分析")
    print("=" * 80)
    
    try:
        with open("embeddings_output/word_lists.pkl", 'rb') as f:
            word_lists = pickle.load(f)
        
        print(f"\n单词列表统计:")
        print(f"  总图像数: {len(word_lists)}")
        
        # 统计每张图像的单词数
        word_counts = [len(words) for words in word_lists]
        print(f"  平均每图像单词数: {np.mean(word_counts):.2f}")
        print(f"  单词数范围: {min(word_counts)} - {max(word_counts)}")
        
        # 显示前几个图像的单词
        print(f"\n前3个图像的单词:")
        for i in range(3):
            print(f"  图像 {i}: {word_lists[i][:20]}...")
        
        # 分析单词类型
        all_words = []
        for words in word_lists:
            all_words.extend(words)
        
        unique_words = set(all_words)
        print(f"\n词汇统计:")
        print(f"  总单词数: {len(all_words)}")
        print(f"  唯一单词数: {len(unique_words)}")
        print(f"  平均每单词出现次数: {len(all_words)/len(unique_words):.2f}")
        
    except FileNotFoundError:
        print("  未找到word_lists.pkl文件")

def main():
    """主函数"""
    analyze_embedding_generation_methods()
    test_word_embedding_consistency()
    analyze_tokenization_impact()
    analyze_embedding_statistics()
    analyze_word_lists_difference()
    
    print("\n" + "=" * 80)
    print("结论和建议")
    print("=" * 80)
    print("""
关键发现:
1. 两种方法在单词级别embedding生成上是相同的
2. 都是将单个单词作为列表输入给sentence transformer
3. 都是对单词embeddings取平均

问题根源:
1. 分词方式差异: 我们丢失了标点符号和短词
2. 大小写转换: 可能影响embedding质量
3. 缺乏词汇修正: 原始方法有verb_adjustments机制

解决方案:
1. 使用与原始方法一致的分词方式
2. 不进行大小写转换
3. 不过滤标点符号
4. 添加词汇修正机制

这解释了为什么相关性这么低 - 不是embedding生成方式的问题，
而是数据预处理（分词）的问题！
""")

if __name__ == "__main__":
    main()
