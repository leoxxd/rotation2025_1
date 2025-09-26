#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试词性标注的一致性（简化版）
"""

import nltk

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
    
    for i, sentence in enumerate(test_sentences):
        print(f"\n测试句子 {i+1}: {sentence}")
        print("-" * 40)
        
        # 使用NLTK进行词性标注
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        
        # 原始项目的名词提取
        word_type_dict = {'noun': ['NN', 'NNS']}
        nouns_original = [x[0] for x in tagged if x[1] in word_type_dict['noun']]
        
        # 原始项目的动词提取
        verbs_original = [x[0] for x in tagged if x[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
        
        print(f"名词: {nouns_original}")
        print(f"动词: {verbs_original}")
        
        # 显示完整的词性标注
        print(f"完整词性标注: {tagged}")
        
        # 检查是否有我们遗漏的词性标签
        all_tags = set([tag for word, tag in tagged])
        noun_tags = set(['NN', 'NNS'])
        verb_tags = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        
        print(f"所有词性标签: {all_tags}")
        print(f"名词标签: {noun_tags}")
        print(f"动词标签: {verb_tags}")
        print(f"其他标签: {all_tags - noun_tags - verb_tags}")

if __name__ == "__main__":
    test_pos_tagging()

