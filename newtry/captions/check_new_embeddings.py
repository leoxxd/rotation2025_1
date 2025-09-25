#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查新生成的embedding文件质量
"""

import pickle
import numpy as np

def check_new_embeddings():
    """检查新生成的embedding文件"""
    
    print("=" * 60)
    print("检查新生成的Embedding文件")
    print("=" * 60)
    
    # 检查单词平均embedding
    try:
        word_emb = np.load('embeddings_output/word_average_embeddings.npy')
        print(f"\n新的单词平均embedding:")
        print(f"  形状: {word_emb.shape}")
        print(f"  均值: {word_emb.mean():.6f}")
        print(f"  标准差: {word_emb.std():.6f}")
        print(f"  最小值: {word_emb.min():.6f}")
        print(f"  最大值: {word_emb.max():.6f}")
    except FileNotFoundError:
        print("  未找到word_average_embeddings.npy文件")
    
    # 检查完整caption embedding
    try:
        image_emb = np.load('embeddings_output/image_embeddings.npy')
        print(f"\n完整caption embedding:")
        print(f"  形状: {image_emb.shape}")
        print(f"  均值: {image_emb.mean():.6f}")
        print(f"  标准差: {image_emb.std():.6f}")
        print(f"  最小值: {image_emb.min():.6f}")
        print(f"  最大值: {image_emb.max():.6f}")
    except FileNotFoundError:
        print("  未找到image_embeddings.npy文件")
    
    # 计算相关性
    try:
        if 'word_emb' in locals() and 'image_emb' in locals():
            correlation = np.corrcoef(word_emb.flatten(), image_emb.flatten())[0, 1]
            print(f"\n相关性分析:")
            print(f"  单词平均embedding与完整caption embedding的相关性: {correlation:.6f}")
            
            if correlation < 0.3:
                print("  ⚠️ 相关性仍然较低，可能还有其他问题")
            elif correlation < 0.5:
                print("  ⚠️ 相关性中等，有改善但还可以更好")
            else:
                print("  ✅ 相关性良好！")
    except Exception as e:
        print(f"  计算相关性时出错: {e}")
    
    # 检查单词列表
    try:
        with open('embeddings_output/word_lists.pkl', 'rb') as f:
            word_lists = pickle.load(f)
        
        print(f"\n单词列表分析:")
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
        
        # 分析单词类型
        all_words = []
        for words in word_lists:
            all_words.extend(words)
        
        unique_words = set(all_words)
        print(f"\n词汇统计:")
        print(f"  总单词数: {len(all_words)}")
        print(f"  唯一单词数: {len(unique_words)}")
        print(f"  平均每单词出现次数: {len(all_words)/len(unique_words):.2f}")
        
        # 检查是否包含标点符号
        punctuation_words = [word for word in unique_words if not word.isalnum()]
        print(f"\n标点符号单词: {len(punctuation_words)}")
        if punctuation_words:
            print(f"  示例: {list(punctuation_words)[:10]}")
        
        # 检查是否包含短词
        short_words = [word for word in unique_words if len(word) == 1]
        print(f"\n单字符单词: {len(short_words)}")
        if short_words:
            print(f"  示例: {list(short_words)[:10]}")
            
    except FileNotFoundError:
        print("  未找到word_lists.pkl文件")
    except Exception as e:
        print(f"  分析单词列表时出错: {e}")

def main():
    """主函数"""
    check_new_embeddings()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
如果相关性仍然较低，可能的原因:
1. 模型选择问题: all-mpnet-base-v2可能不是最优选择
2. 数据预处理问题: 可能需要更仔细的文本清理
3. 词汇修正不完整: 可能需要更全面的词汇修正字典
4. 模型版本问题: 可能需要使用与原始项目相同的模型版本

建议:
1. 检查原始项目使用的具体模型版本
2. 对比原始项目的词汇修正字典
3. 考虑使用fasttext或glove等传统embedding方法
""")

if __name__ == "__main__":
    main()
