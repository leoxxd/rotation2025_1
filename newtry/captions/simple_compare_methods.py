#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版方法对比分析（不依赖sentence_transformers）
"""

import os
import pickle
import numpy as np
import nltk

def compare_methods():
    """对比两种方法的关键差异"""
    
    print("=" * 80)
    print("单词平均Embedding生成方法对比分析")
    print("=" * 80)
    
    print("\n1. 数据格式差异:")
    print("   原始方法: 使用pickle格式的caption数据")
    print("   我们的方法: 使用文本文件格式的caption数据")
    
    print("\n2. 分词方式差异:")
    print("   原始方法: nltk.word_tokenize(this_sentence)")
    print("   我们的方法: nltk.word_tokenize(caption.lower()) + 过滤标点符号")
    
    print("\n3. 词汇修正差异:")
    print("   原始方法: 使用verb_adjustments字典修正拼写错误")
    print("   我们的方法: 没有词汇修正机制")
    
    print("\n4. 模型使用差异:")
    print("   原始方法: 支持多种embedding类型(fasttext, glove, sentence transformer)")
    print("   我们的方法: 只使用all-mpnet-base-v2")
    
    print("\n5. 错误处理差异:")
    print("   原始方法: 跳过没有embedding的单词，继续处理")
    print("   我们的方法: 跳过没有embedding的单词，继续处理")
    
    print("\n6. 平均计算差异:")
    print("   原始方法: np.mean(np.asarray(img_allWord_embeddings), axis=0)")
    print("   我们的方法: np.mean(word_embeddings, axis=0)")

def test_tokenization_differences():
    """测试分词差异"""
    print("\n" + "=" * 80)
    print("分词差异测试")
    print("=" * 80)
    
    # 测试句子
    test_sentences = [
        "Food on a white plate with vegetables and a garnish.",
        "A cooked stir fry dish arranged on a plate.",
        "A cup of coffee on a plate with a spoon.",
        "Women playing frisbee in a field."
    ]
    
    print("\n原始方法分词结果:")
    for sentence in test_sentences:
        tokens = nltk.word_tokenize(sentence)
        print(f"  '{sentence}'")
        print(f"  -> {tokens}")
    
    print("\n我们的方法分词结果:")
    for sentence in test_sentences:
        tokens = nltk.word_tokenize(sentence.lower())
        # 过滤标点符号和短词
        filtered_tokens = [token for token in tokens if len(token) > 1 and token.isalpha()]
        print(f"  '{sentence}'")
        print(f"  -> {filtered_tokens}")

def test_embedding_quality():
    """测试embedding质量"""
    print("\n" + "=" * 80)
    print("Embedding质量测试")
    print("=" * 80)
    
    # 加载我们的embedding
    try:
        word_avg_emb = np.load("embeddings_output/word_average_embeddings.npy")
        print(f"\n我们的单词平均embedding:")
        print(f"  形状: {word_avg_emb.shape}")
        print(f"  均值: {word_avg_emb.mean():.6f}")
        print(f"  标准差: {word_avg_emb.std():.6f}")
        print(f"  最小值: {word_avg_emb.min():.6f}")
        print(f"  最大值: {word_avg_emb.max():.6f}")
        
        # 检查是否有异常值
        if word_avg_emb.std() < 0.01:
            print("  ⚠️ 警告: 标准差过小，可能存在问题")
        if abs(word_avg_emb.mean()) > 0.1:
            print("  ⚠️ 警告: 均值偏离0过多，可能存在问题")
            
    except FileNotFoundError:
        print("  未找到word_average_embeddings.npy文件")
    
    # 加载完整caption embedding进行对比
    try:
        image_emb = np.load("embeddings_output/image_embeddings.npy")
        print(f"\n完整caption embedding:")
        print(f"  形状: {image_emb.shape}")
        print(f"  均值: {image_emb.mean():.6f}")
        print(f"  标准差: {image_emb.std():.6f}")
        print(f"  最小值: {image_emb.min():.6f}")
        print(f"  最大值: {image_emb.max():.6f}")
        
        # 计算相关性
        if 'word_avg_emb' in locals():
            correlation = np.corrcoef(word_avg_emb.flatten(), image_emb.flatten())[0, 1]
            print(f"\n单词平均embedding与完整caption embedding的相关性: {correlation:.6f}")
            
            if correlation < 0.3:
                print("  ⚠️ 警告: 相关性过低，可能存在问题")
                
    except FileNotFoundError:
        print("  未找到image_embeddings.npy文件")

def analyze_word_lists():
    """分析单词列表"""
    print("\n" + "=" * 80)
    print("单词列表分析")
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
            print(f"  图像 {i}: {word_lists[i][:15]}...")
        
        # 检查是否有空列表
        empty_count = sum(1 for words in word_lists if len(words) == 0)
        print(f"\n空单词列表数量: {empty_count}")
        
    except FileNotFoundError:
        print("  未找到word_lists.pkl文件")

def main():
    """主函数"""
    compare_methods()
    test_tokenization_differences()
    test_embedding_quality()
    analyze_word_lists()
    
    print("\n" + "=" * 80)
    print("总结和建议")
    print("=" * 80)
    print("""
主要差异:
1. 数据格式: 原始方法使用pickle，我们使用文本文件
2. 分词方式: 我们添加了大小写转换和标点符号过滤
3. 词汇修正: 原始方法有verb_adjustments修正机制，我们没有
4. 模型类型: 原始方法支持多种模型，我们只使用sentence transformer

可能的问题:
1. 大小写转换可能影响embedding质量
2. 标点符号过滤可能过于激进
3. 缺乏词汇修正机制
4. 模型选择可能不是最优的

建议:
1. 尝试不进行大小写转换
2. 减少标点符号过滤的激进程度
3. 添加词汇修正机制
4. 考虑使用与原始方法相同的模型
""")

if __name__ == "__main__":
    main()
