#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名词动词分离Embedding生成器
将caption中的名词和动词分别提取，生成对应的embeddings
"""

import os
import pickle
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class NounVerbEmbeddingGenerator:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        初始化名词动词分离embedding生成器
        
        Args:
            model_name: 使用的sentence transformer模型名称
        """
        self.model_name = model_name
        self.model = None
        self.caption_file = "Anno_Shared1000.txt"
        self.output_dir = "embeddings_output"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("下载NLTK数据...")
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
    
    def load_model(self):
        """加载sentence transformer模型"""
        print(f"加载模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"模型加载完成，embedding维度: {self.model.get_sentence_embedding_dimension()}")
    
    def load_captions(self):
        """加载caption文件"""
        print(f"加载caption文件: {self.caption_file}")
        
        if not os.path.exists(self.caption_file):
            raise FileNotFoundError(f"Caption文件不存在: {self.caption_file}")
        
        # 支持多种格式
        if self.caption_file.endswith('.pkl'):
            # pickle格式：每个图片有多句caption
            with open(self.caption_file, 'rb') as f:
                captions = pickle.load(f)
            print(f"加载了 {len(captions)} 个图片的captions")
            print(f"每个图片平均有 {np.mean([len(c) for c in captions]):.1f} 句caption")
        elif self.caption_file.endswith('.txt'):
            # 文本格式：每行一个图片的多个caption（JSON格式）
            import json
            captions = []
            with open(self.caption_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            # 解析JSON格式的caption数据
                            caption_data = json.loads(line)
                            # 提取caption文本
                            image_captions = [item['caption'] for item in caption_data]
                            captions.append(image_captions)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ 解析JSON失败: {e}")
                            continue
            print(f"加载了 {len(captions)} 个图片的captions")
            print(f"每个图片平均有 {np.mean([len(c) for c in captions]):.1f} 句caption")
        else:
            # 其他格式：每行一句caption
            captions = []
            with open(self.caption_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        captions.append([line])  # 包装成列表格式以保持一致性
            print(f"加载了 {len(captions)} 个captions")
        
        return captions
    
    def extract_nouns_and_verbs(self, caption):
        """
        从caption中提取名词和动词
        
        Args:
            caption: 输入文本
            
        Returns:
            nouns: 名词列表
            verbs: 动词列表
        """
        # 分词和词性标注
        tokens = nltk.word_tokenize(caption.lower())
        tagged = nltk.pos_tag(tokens)
        
        # 定义名词和动词的POS标签
        noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']  # 名词
        verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # 动词
        
        nouns = []
        verbs = []
        
        for word, pos in tagged:
            # 过滤掉标点符号和短词
            if len(word) > 1 and word.isalpha():
                if pos in noun_tags:
                    nouns.append(word)
                elif pos in verb_tags:
                    verbs.append(word)
        
        return nouns, verbs
    
    def get_word_embedding(self, word):
        """
        获取单个单词的embedding
        
        Args:
            word: 输入单词
            
        Returns:
            embedding: 单词的embedding向量
        """
        # 使用sentence transformer对单个单词生成embedding
        embedding = self.model.encode([word], convert_to_tensor=False)[0]
        return embedding
    
    def generate_noun_verb_embeddings(self, captions):
        """
        生成名词和动词的embeddings
        
        Args:
            captions: caption列表，每个元素是一个图片的多句caption列表
            
        Returns:
            noun_embeddings: 名词embeddings数组
            verb_embeddings: 动词embeddings数组
            noun_lists: 每个图片的名词列表
            verb_lists: 每个图片的动词列表
        """
        print("开始生成名词动词embeddings...")
        
        n_images = len(captions)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # 初始化结果数组
        noun_embeddings = np.zeros((n_images, embedding_dim))
        verb_embeddings = np.zeros((n_images, embedding_dim))
        noun_lists = []
        verb_lists = []
        
        # 统计信息
        total_nouns = 0
        total_verbs = 0
        no_nouns_count = 0
        no_verbs_count = 0
        
        for i, image_captions in enumerate(captions):
            if i % 100 == 0:
                print(f"处理进度: {i}/{n_images} ({i/n_images*100:.1f}%)")
            
            # 收集该图片所有caption的名词和动词
            all_nouns = []
            all_verbs = []
            
            for j, caption in enumerate(image_captions):
                nouns, verbs = self.extract_nouns_and_verbs(caption)
                all_nouns.extend(nouns)
                all_verbs.extend(verbs)
            
            noun_lists.append(all_nouns)
            verb_lists.append(all_verbs)
            
            # 处理名词
            if len(all_nouns) == 0:
                # 如果没有名词，使用"object"的embedding
                noun_embeddings[i] = self.get_word_embedding("object")
                no_nouns_count += 1
            else:
                # 为每个名词生成embedding并取平均
                noun_word_embeddings = []
                for noun in all_nouns:
                    try:
                        noun_emb = self.get_word_embedding(noun)
                        noun_word_embeddings.append(noun_emb)
                    except Exception as e:
                        print(f"  ⚠️ 名词 '{noun}' 生成embedding失败: {e}")
                        continue
                
                if len(noun_word_embeddings) == 0:
                    noun_embeddings[i] = self.get_word_embedding("object")
                    no_nouns_count += 1
                else:
                    noun_embeddings[i] = np.mean(noun_word_embeddings, axis=0)
                    total_nouns += len(noun_word_embeddings)
            
            # 处理动词
            if len(all_verbs) == 0:
                # 如果没有动词，使用"is"的embedding
                verb_embeddings[i] = self.get_word_embedding("is")
                no_verbs_count += 1
            else:
                # 为每个动词生成embedding并取平均
                verb_word_embeddings = []
                for verb in all_verbs:
                    try:
                        verb_emb = self.get_word_embedding(verb)
                        verb_word_embeddings.append(verb_emb)
                    except Exception as e:
                        print(f"  ⚠️ 动词 '{verb}' 生成embedding失败: {e}")
                        continue
                
                if len(verb_word_embeddings) == 0:
                    verb_embeddings[i] = self.get_word_embedding("is")
                    no_verbs_count += 1
                else:
                    verb_embeddings[i] = np.mean(verb_word_embeddings, axis=0)
                    total_verbs += len(verb_word_embeddings)
        
        print(f"\n生成完成!")
        print(f"  总图片数: {n_images}")
        print(f"  总名词数: {total_nouns}")
        print(f"  总动词数: {total_verbs}")
        print(f"  平均每图片名词数: {total_nouns/n_images:.2f}")
        print(f"  平均每图片动词数: {total_verbs/n_images:.2f}")
        print(f"  无名词图片数: {no_nouns_count}")
        print(f"  无动词图片数: {no_verbs_count}")
        print(f"  Embedding维度: {embedding_dim}")
        
        return noun_embeddings, verb_embeddings, noun_lists, verb_lists
    
    def save_embeddings(self, noun_embeddings, verb_embeddings, noun_lists, verb_lists):
        """保存embeddings和元数据"""
        print("保存结果...")
        
        # 保存名词embeddings
        noun_embedding_file = os.path.join(self.output_dir, "noun_embeddings.npy")
        np.save(noun_embedding_file, noun_embeddings)
        print(f"  名词embeddings已保存: {noun_embedding_file}")
        
        # 保存动词embeddings
        verb_embedding_file = os.path.join(self.output_dir, "verb_embeddings.npy")
        np.save(verb_embedding_file, verb_embeddings)
        print(f"  动词embeddings已保存: {verb_embedding_file}")
        
        # 保存元数据
        metadata = {
            'model_name': self.model_name,
            'embedding_type': 'noun_verb_separated',
            'n_captions': len(noun_embeddings),
            'embedding_dim': noun_embeddings.shape[1],
            'caption_file': self.caption_file,
            'description': 'Noun and verb embeddings extracted separately from captions',
            'noun_embedding_file': noun_embedding_file,
            'verb_embedding_file': verb_embedding_file
        }
        
        metadata_file = os.path.join(self.output_dir, "noun_verb_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  元数据已保存: {metadata_file}")
        
        # 保存名词列表
        noun_lists_file = os.path.join(self.output_dir, "noun_lists.pkl")
        with open(noun_lists_file, 'wb') as f:
            pickle.dump(noun_lists, f)
        print(f"  名词列表已保存: {noun_lists_file}")
        
        # 保存动词列表
        verb_lists_file = os.path.join(self.output_dir, "verb_lists.pkl")
        with open(verb_lists_file, 'wb') as f:
            pickle.dump(verb_lists, f)
        print(f"  动词列表已保存: {verb_lists_file}")
        
        return noun_embedding_file, verb_embedding_file, metadata_file, noun_lists_file, verb_lists_file
    
    def run(self):
        """运行完整的embedding生成流程"""
        print("=" * 60)
        print("名词动词分离Embedding生成器")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 加载captions
        captions = self.load_captions()
        
        # 3. 生成embeddings
        noun_embeddings, verb_embeddings, noun_lists, verb_lists = self.generate_noun_verb_embeddings(captions)
        
        # 4. 保存结果
        noun_file, verb_file, metadata_file, noun_lists_file, verb_lists_file = self.save_embeddings(
            noun_embeddings, verb_embeddings, noun_lists, verb_lists
        )
        
        print("\n" + "=" * 60)
        print("生成完成!")
        print(f"名词embedding文件: {noun_file}")
        print(f"动词embedding文件: {verb_file}")
        print(f"元数据文件: {metadata_file}")
        print(f"名词列表文件: {noun_lists_file}")
        print(f"动词列表文件: {verb_lists_file}")
        print("=" * 60)
        
        return noun_embeddings, verb_embeddings, noun_lists, verb_lists

def main():
    """主函数"""
    # 创建生成器
    generator = NounVerbEmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    
    # 运行生成流程
    noun_embeddings, verb_embeddings, noun_lists, verb_lists = generator.run()
    
    # 显示一些统计信息
    print(f"\n名词Embedding统计信息:")
    print(f"  形状: {noun_embeddings.shape}")
    print(f"  均值: {noun_embeddings.mean():.4f}")
    print(f"  标准差: {noun_embeddings.std():.4f}")
    print(f"  最小值: {noun_embeddings.min():.4f}")
    print(f"  最大值: {noun_embeddings.max():.4f}")
    
    print(f"\n动词Embedding统计信息:")
    print(f"  形状: {verb_embeddings.shape}")
    print(f"  均值: {verb_embeddings.mean():.4f}")
    print(f"  标准差: {verb_embeddings.std():.4f}")
    print(f"  最小值: {verb_embeddings.min():.4f}")
    print(f"  最大值: {verb_embeddings.max():.4f}")

if __name__ == "__main__":
    main()
