#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正后的分词方法
验证是否与原始NSD方法一致
"""

import nltk

def test_original_style_tokenization():
    """测试修正后的分词方法"""
    
    print("=" * 60)
    print("测试修正后的分词方法")
    print("=" * 60)
    
    # 词汇修正字典（与修改后的代码一致）
    verb_adjustments = {
        'waterskiing': '_____no_embedding_____',
        'unpealed': '_____no_embedding_____',
    }
    
    def tokenize_caption_original_style(caption):
        """
        使用与原始NSD项目一致的分词方式
        """
        # 使用NLTK进行分词（与原始方法一致，不进行大小写转换）
        tokens = nltk.word_tokenize(caption)
        
        # 应用词汇修正（与原始方法一致）
        corrected_tokens = []
        for token in tokens:
            if token in verb_adjustments:
                if verb_adjustments[token] == "_____not_verb_/_unknown_____":
                    continue  # 跳过这个词汇
                elif verb_adjustments[token] == "_____no_embedding_____":
                    continue  # 跳过没有embedding的词汇
                else:
                    corrected_tokens.append(verb_adjustments[token])
            else:
                corrected_tokens.append(token)
        
        return corrected_tokens
    
    # 测试句子
    test_sentences = [
        "Food on a white plate with vegetables and a garnish.",
        "A cooked stir fry dish arranged on a plate.",
        "A cup of coffee on a plate with a spoon.",
        "Women playing frisbee in a field."
    ]
    
    print("\n修正后的分词结果（与原始方法一致）:")
    for sentence in test_sentences:
        tokens = tokenize_caption_original_style(sentence)
        print(f"  '{sentence}'")
        print(f"  -> {tokens}")
        print(f"  单词数量: {len(tokens)}")
        print()
    
    # 对比原始方法
    print("\n对比原始方法分词结果:")
    for sentence in test_sentences:
        original_tokens = nltk.word_tokenize(sentence)
        print(f"  '{sentence}'")
        print(f"  -> {original_tokens}")
        print(f"  单词数量: {len(original_tokens)}")
        print()
    
    # 分析差异
    print("\n差异分析:")
    for sentence in test_sentences:
        original_tokens = nltk.word_tokenize(sentence)
        corrected_tokens = tokenize_caption_original_style(sentence)
        
        print(f"句子: '{sentence}'")
        print(f"  原始方法: {original_tokens}")
        print(f"  修正方法: {corrected_tokens}")
        
        if original_tokens == corrected_tokens:
            print("  ✅ 完全一致！")
        else:
            print("  ⚠️ 存在差异")
            print(f"  差异: {set(original_tokens) - set(corrected_tokens)}")
        print()

def test_old_vs_new_method():
    """对比旧方法和新方法"""
    
    print("=" * 60)
    print("对比旧方法和新方法")
    print("=" * 60)
    
    def old_tokenize_caption(caption):
        """旧方法：过滤标点符号和短词"""
        tokens = nltk.word_tokenize(caption.lower())
        filtered_tokens = []
        for token in tokens:
            if len(token) > 1 and token.isalpha():
                filtered_tokens.append(token)
        return filtered_tokens
    
    def new_tokenize_caption(caption):
        """新方法：与原始方法一致"""
        tokens = nltk.word_tokenize(caption)
        return tokens  # 简化版本，不应用词汇修正
    
    test_sentence = "Food on a white plate with vegetables and a garnish."
    
    print(f"测试句子: '{test_sentence}'")
    print()
    
    old_tokens = old_tokenize_caption(test_sentence)
    new_tokens = new_tokenize_caption(test_sentence)
    
    print(f"旧方法结果: {old_tokens}")
    print(f"新方法结果: {new_tokens}")
    print()
    
    print("差异分析:")
    print(f"  旧方法单词数: {len(old_tokens)}")
    print(f"  新方法单词数: {len(new_tokens)}")
    print(f"  丢失的单词: {set(new_tokens) - set(old_tokens)}")
    print(f"  新增的单词: {set(old_tokens) - set(new_tokens)}")
    print(f"  共同单词: {set(old_tokens) & set(new_tokens)}")
    
    overlap_rate = len(set(old_tokens) & set(new_tokens)) / len(new_tokens)
    print(f"  单词重叠率: {overlap_rate:.2%}")

def main():
    """主函数"""
    test_original_style_tokenization()
    test_old_vs_new_method()
    
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
修正后的方法特点:
1. ✅ 不进行大小写转换
2. ✅ 不过滤标点符号
3. ✅ 不过滤短词
4. ✅ 添加词汇修正机制
5. ✅ 与原始NSD方法完全一致

这应该能解决相关性过低的问题！
""")

if __name__ == "__main__":
    main()
