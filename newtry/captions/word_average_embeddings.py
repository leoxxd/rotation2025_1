#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单词平均Embedding生成器
将caption分词后，对每个单词生成embedding，然后取平均
"""

import os
import pickle
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class WordAverageEmbeddingGenerator:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        初始化单词平均embedding生成器
        
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
        except LookupError:
            print("下载NLTK punkt tokenizer...")
            nltk.download('punkt')
    
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
    
    def tokenize_caption(self, caption):
        """
        对caption进行分词
        
        Args:
            caption: 输入文本
            
        Returns:
            tokens: 分词后的单词列表
        """
        # 使用NLTK进行分词
        tokens = nltk.word_tokenize(caption.lower())
        
        # 过滤掉标点符号和短词
        filtered_tokens = []
        for token in tokens:
            # 保留长度大于1且不是纯标点的词
            if len(token) > 1 and token.isalpha():
                filtered_tokens.append(token)
        
        return filtered_tokens
    
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
    
    def generate_word_average_embeddings(self, captions):
        """
        生成单词平均embeddings
        
        Args:
            captions: caption列表，每个元素是一个图片的多句caption列表
            
        Returns:
            embeddings: 单词平均embeddings数组
            word_lists: 每个图片的单词列表
        """
        print("开始生成单词平均embeddings...")
        
        n_images = len(captions)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # 初始化结果数组
        embeddings = np.zeros((n_images, embedding_dim))
        word_lists = []
        
        # 统计信息
        total_words = 0
        skipped_images = 0
        
        for i, image_captions in enumerate(captions):
            if i % 100 == 0:
                print(f"处理进度: {i}/{n_images} ({i/n_images*100:.1f}%)")
            
            # 收集该图片所有caption的单词
            all_words = []
            for j, caption in enumerate(image_captions):
                words = self.tokenize_caption(caption)
                all_words.extend(words)
            
            word_lists.append(all_words)
            
            if len(all_words) == 0:
                # 如果没有有效单词，使用空字符串的embedding
                print(f"  ⚠️ 图片 {i} 没有有效单词")
                embeddings[i] = self.get_word_embedding("")
                skipped_images += 1
                continue
            
            # 为每个单词生成embedding
            word_embeddings = []
            for word in all_words:
                try:
                    word_emb = self.get_word_embedding(word)
                    word_embeddings.append(word_emb)
                except Exception as e:
                    print(f"  ⚠️ 单词 '{word}' 生成embedding失败: {e}")
                    continue
            
            if len(word_embeddings) == 0:
                # 如果所有单词都失败，使用空字符串的embedding
                print(f"  ⚠️ 图片 {i} 所有单词embedding生成失败")
                embeddings[i] = self.get_word_embedding("")
                skipped_images += 1
            else:
                # 计算单词embeddings的平均值
                embeddings[i] = np.mean(word_embeddings, axis=0)
                total_words += len(word_embeddings)
        
        print(f"\n生成完成!")
        print(f"  总图片数: {n_images}")
        print(f"  总单词数: {total_words}")
        print(f"  平均每图片单词数: {total_words/n_images:.2f}")
        print(f"  跳过图片数: {skipped_images}")
        print(f"  Embedding维度: {embedding_dim}")
        
        return embeddings, word_lists
    
    def save_embeddings(self, embeddings, word_lists):
        """保存embeddings和元数据"""
        print("保存结果...")
        
        # 保存embeddings
        embedding_file = os.path.join(self.output_dir, "word_average_embeddings.npy")
        np.save(embedding_file, embeddings)
        print(f"  Embeddings已保存: {embedding_file}")
        
        # 保存元数据
        metadata = {
            'model_name': self.model_name,
            'embedding_type': 'word_average',
            'n_captions': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'caption_file': self.caption_file,
            'description': 'Word-level embeddings averaged across all words in each caption'
        }
        
        metadata_file = os.path.join(self.output_dir, "word_average_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  元数据已保存: {metadata_file}")
        
        # 保存单词列表
        word_lists_file = os.path.join(self.output_dir, "word_lists.pkl")
        with open(word_lists_file, 'wb') as f:
            pickle.dump(word_lists, f)
        print(f"  单词列表已保存: {word_lists_file}")
        
        return embedding_file, metadata_file, word_lists_file
    
    def run(self):
        """运行完整的embedding生成流程"""
        print("=" * 60)
        print("单词平均Embedding生成器")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 加载captions
        captions = self.load_captions()
        
        # 3. 生成embeddings
        embeddings, word_lists = self.generate_word_average_embeddings(captions)
        
        # 4. 保存结果
        embedding_file, metadata_file, word_lists_file = self.save_embeddings(embeddings, word_lists)
        
        print("\n" + "=" * 60)
        print("生成完成!")
        print(f"Embedding文件: {embedding_file}")
        print(f"元数据文件: {metadata_file}")
        print(f"单词列表文件: {word_lists_file}")
        print("=" * 60)
        
        return embeddings, word_lists

def main():
    """主函数"""
    # 创建生成器
    generator = WordAverageEmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    
    # 运行生成流程
    embeddings, word_lists = generator.run()
    
    # 显示一些统计信息
    print(f"\nEmbedding统计信息:")
    print(f"  形状: {embeddings.shape}")
    print(f"  均值: {embeddings.mean():.4f}")
    print(f"  标准差: {embeddings.std():.4f}")
    print(f"  最小值: {embeddings.min():.4f}")
    print(f"  最大值: {embeddings.max():.4f}")

if __name__ == "__main__":
    main()
