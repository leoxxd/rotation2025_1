#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试词性标注的一致性
"""

import sys
import os
sys.path.append('../captions')

import nltk
from noun_verb_embeddings import NounVerbEmbeddingGenerator

def test_pos_tagging():
    """测试词性标注的一致性"""
    
    # 测试句子
    test_sentences = [
        "A person is walking on the street",
        "The cat is sitting on the table",
        "A dog is running in the park",
        "The man is eating an apple",
        "A woman is reading a book"
    ]
    
    print("=" * 60)
    print("词性标注一致性测试")
    print("=" * 60)
    
    # 创建生成器实例
    generator = NounVerbEmbeddingGenerator()
    
    for i, sentence in enumerate(test_sentences):
        print(f"\n测试句子 {i+1}: {sentence}")
        print("-" * 40)
        
        # 使用我们的方法
        nouns_our = generator.get_word_type_from_string_original_style(sentence, 'noun')
        verbs_our = generator.get_verbs_from_string_original_style(sentence)
        
        # 使用原始项目的方法（直接调用NLTK）
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        
        # 原始项目的名词提取
        word_type_dict = {'noun': ['NN', 'NNS']}
        nouns_original = [x[0] for x in tagged if x[1] in word_type_dict['noun']]
        
        # 原始项目的动词提取
        verbs_original = [x[0] for x in tagged if x[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
        
        print(f"我们的名词提取: {nouns_our}")
        print(f"原始名词提取:   {nouns_original}")
        print(f"名词一致: {nouns_our == nouns_original}")
        
        print(f"我们的动词提取: {verbs_our}")
        print(f"原始动词提取:   {verbs_original}")
        print(f"动词一致: {verbs_our == verbs_original}")
        
        # 显示完整的词性标注
        print(f"完整词性标注: {tagged}")

if __name__ == "__main__":
    test_pos_tagging()

