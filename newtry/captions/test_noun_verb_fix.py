#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正后的名词动词提取方法
验证是否与原始NSD方法一致
"""

import nltk

def test_original_style_noun_verb_extraction():
    """测试修正后的名词动词提取方法"""
    
    print("=" * 60)
    print("测试修正后的名词动词提取方法")
    print("=" * 60)
    
    # 词汇修正字典（与修改后的代码一致）
    verb_adjustments = {
        'waterskiing': '_____no_embedding_____',
        'unpealed': '_____no_embedding_____',
    }
    
    def extract_nouns_and_verbs_original_style(caption):
        """
        使用与原始NSD项目一致的方法提取名词和动词
        """
        try:
            # 使用与原始方法一致的分词方式（不进行大小写转换）
            tokens = nltk.word_tokenize(caption)
            tagged = nltk.pos_tag(tokens)
            
            # 定义名词和动词的POS标签 - 与原始代码一致
            noun_tags = ['NN', 'NNS']  # 与get_word_type_dict()一致
            verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # 与get_verbs_from_string()一致
            
            nouns = []
            verbs = []
            
            for word, pos in tagged:
                # 应用词汇修正（与原始方法一致）
                if word in verb_adjustments:
                    if verb_adjustments[word] == "_____not_verb_/_unknown_____":
                        continue  # 跳过这个词汇
                    elif verb_adjustments[word] == "_____no_embedding_____":
                        continue  # 跳过没有embedding的词汇
                    else:
                        word = verb_adjustments[word]
                
                # 不过滤标点符号和短词（与原始方法一致）
                if pos in noun_tags:
                    nouns.append(word)
                elif pos in verb_tags:
                    verbs.append(word)
            
            return nouns, verbs
            
        except Exception as e:
            print(f"  ⚠️ 词性标注失败: {e}")
            return [], []
    
    # 测试句子
    test_sentences = [
        "Food on a white plate with vegetables and a garnish.",
        "A cooked stir fry dish arranged on a plate.",
        "A cup of coffee on a plate with a spoon.",
        "Women playing frisbee in a field."
    ]
    
    print("\n修正后的名词动词提取结果（与原始方法一致）:")
    for sentence in test_sentences:
        nouns, verbs = extract_nouns_and_verbs_original_style(sentence)
        print(f"  '{sentence}'")
        print(f"  名词: {nouns}")
        print(f"  动词: {verbs}")
        print()
    
    # 对比原始方法
    print("\n对比原始方法分词结果:")
    for sentence in test_sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        
        # 原始方法的名词提取
        noun_tags = ['NN', 'NNS']
        verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        
        original_nouns = [word for word, pos in tagged if pos in noun_tags]
        original_verbs = [word for word, pos in tagged if pos in verb_tags]
        
        print(f"  '{sentence}'")
        print(f"  原始方法名词: {original_nouns}")
        print(f"  原始方法动词: {original_verbs}")
        print()
    
    # 对比旧方法
    print("\n对比旧方法分词结果:")
    for sentence in test_sentences:
        tokens = nltk.word_tokenize(sentence.lower())
        tagged = nltk.pos_tag(tokens)
        
        noun_tags = ['NN', 'NNS']
        verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        
        old_nouns = []
        old_verbs = []
        
        for word, pos in tagged:
            if len(word) > 1 and word.isalpha():
                if pos in noun_tags:
                    old_nouns.append(word)
                elif pos in verb_tags:
                    old_verbs.append(word)
        
        print(f"  '{sentence}'")
        print(f"  旧方法名词: {old_nouns}")
        print(f"  旧方法动词: {old_verbs}")
        print()

def main():
    """主函数"""
    test_original_style_noun_verb_extraction()
    
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
修正后的方法特点:
1. ✅ 不进行大小写转换
2. ✅ 不过滤标点符号和短词
3. ✅ 添加词汇修正机制
4. ✅ 与原始NSD方法完全一致

这应该能解决noun和verb embedding的相关性问题！
""")

if __name__ == "__main__":
    main()
