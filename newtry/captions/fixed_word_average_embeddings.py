#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版单词平均Embedding生成器
基于原始NSD项目的方法，修正分词和embedding生成问题
"""

import os
import pickle
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# 设置Hugging Face缓存目录到当前项目目录
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

class FixedWordAverageEmbeddingGenerator:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        初始化修正版单词平均embedding生成器
        
        Args:
            model_name: 使用的sentence transformer模型名称
        """
        self.model_name = model_name
        self.model = None
        self.caption_file = "Anno_Shared1000.txt"
        self.output_dir = "embeddings_output_fixed"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("正在下载 NLTK punkt 分词器...")
            nltk.download('punkt')
        
        # 词汇修正字典（基于原始项目的verb_adjustments）
        self.verb_adjustments = {
            # 常见的拼写错误和误分类
            'waterskiing': '_____no_embedding_____',
            'unpealed': '_____no_embedding_____',
            # 可以添加更多修正规则
        }
    
    def load_model(self):
        """加载sentence transformer模型"""
        print(f"正在加载模型: {self.model_name}")
        
        # 尝试使用本地模型路径
        local_model_path = "../models/all-mpnet-base-v2"
        
        try:
            if os.path.exists(local_model_path):
                print(f"使用本地模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path)
            else:
                print("尝试从网络下载模型...")
                self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e
        
        print(f"模型加载完成，embedding维度: {self.model.get_sentence_embedding_dimension()}")
    
    def load_captions(self):
        """加载caption文件"""
        print(f"正在加载数据: {self.caption_file}")
        
        if not os.path.exists(self.caption_file):
            raise FileNotFoundError(f"Caption文件不存在: {self.caption_file}")
        
        # 按图像分组存储数据
        image_data = {}  # {image_id: [caption1, caption2, caption3, caption4, caption5]}
        
        with open(self.caption_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                try:
                    # 解析每行的数据
                    image_group = eval(line)
                    for item in image_group:
                        image_id = item['image_id']
                        caption = item['caption']
                        
                        if image_id not in image_data:
                            image_data[image_id] = []
                        image_data[image_id].append(caption)
                except Exception as e:
                    print(f"解析行时出错: {e}")
                    continue
        
        # 转换为列表格式
        captions = []
        for image_id in sorted(image_data.keys()):
            captions.append(image_data[image_id])
        
        print(f"成功加载 {len(captions)} 张图像")
        print(f"每张图像平均有 {np.mean([len(c) for c in captions]):.1f} 个caption")
        
        return captions
    
    def tokenize_caption_original_style(self, caption):
        """
        使用与原始方法一致的分词方式
        
        Args:
            caption: 输入文本
            
        Returns:
            tokens: 分词后的单词列表
        """
        # 使用NLTK进行分词（与原始方法一致）
        tokens = nltk.word_tokenize(caption)
        
        # 应用词汇修正（与原始方法一致）
        corrected_tokens = []
        for token in tokens:
            if token in self.verb_adjustments:
                if self.verb_adjustments[token] == "_____not_verb_/_unknown_____":
                    continue  # 跳过这个词汇
                elif self.verb_adjustments[token] == "_____no_embedding_____":
                    continue  # 跳过没有embedding的词汇
                else:
                    corrected_tokens.append(self.verb_adjustments[token])
            else:
                corrected_tokens.append(token)
        
        return corrected_tokens
    
    def get_word_embedding(self, word):
        """获取单个单词的embedding"""
        try:
            # 使用sentence transformer对单个单词生成embedding
            embedding = self.model.encode([word], convert_to_tensor=False)[0]
            return embedding
        except Exception as e:
            print(f"获取单词 '{word}' 的embedding失败: {e}")
            return None
    
    def generate_fixed_word_average_embeddings(self, captions):
        """生成修正版的单词平均embeddings"""
        print("开始生成修正版单词平均embeddings...")
        
        n_images = len(captions)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # 初始化结果数组
        embeddings = np.zeros((n_images, embedding_dim))
        word_lists = []
        
        # 统计信息
        total_words = 0
        skipped_words = 0
        
        for i, image_captions in enumerate(captions):
            if i % 100 == 0:
                print(f"处理进度: {i}/{n_images} ({i/n_images*100:.1f}%)")
            
            # 收集该图片所有caption的单词
            all_words = []
            for j, caption in enumerate(image_captions):
                words = self.tokenize_caption_original_style(caption)
                all_words.extend(words)
            
            word_lists.append(all_words)
            
            # 为每个单词生成embedding
            word_embeddings = []
            for word in all_words:
                try:
                    word_emb = self.get_word_embedding(word)
                    if word_emb is not None:
                        word_embeddings.append(word_emb)
                        total_words += 1
                    else:
                        skipped_words += 1
                except Exception as e:
                    print(f"  ⚠️ 单词 '{word}' 生成embedding失败: {e}")
                    skipped_words += 1
            
            if len(word_embeddings) == 0:
                # 如果没有有效单词，使用空字符串的embedding
                print(f"  ⚠️ 图片 {i} 没有有效单词")
                embeddings[i] = self.get_word_embedding("")
            else:
                # 计算单词平均embedding（与原始方法一致）
                embeddings[i] = np.mean(np.asarray(word_embeddings), axis=0)
        
        print(f"\n生成完成!")
        print(f"  总图片数: {n_images}")
        print(f"  总单词数: {total_words}")
        print(f"  跳过单词数: {skipped_words}")
        print(f"  平均每图片单词数: {total_words/n_images:.2f}")
        print(f"  Embedding维度: {embedding_dim}")
        
        return embeddings, word_lists
    
    def save_embeddings(self, embeddings, word_lists):
        """保存embeddings和元数据"""
        print("保存结果...")
        
        # 保存单词平均embeddings
        embedding_file = os.path.join(self.output_dir, "word_average_embeddings_fixed.npy")
        np.save(embedding_file, embeddings)
        print(f"  单词平均embeddings已保存: {embedding_file}")
        
        # 保存元数据
        metadata = {
            'model_name': self.model_name,
            'embedding_type': 'fixed_word_average',
            'n_captions': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'caption_file': self.caption_file,
            'description': 'Fixed word average embeddings using original NSD method style',
            'embedding_file': embedding_file
        }
        
        metadata_file = os.path.join(self.output_dir, "fixed_word_average_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  元数据已保存: {metadata_file}")
        
        # 保存单词列表
        word_lists_file = os.path.join(self.output_dir, "word_lists_fixed.pkl")
        with open(word_lists_file, 'wb') as f:
            pickle.dump(word_lists, f)
        print(f"  单词列表已保存: {word_lists_file}")
        
        return embedding_file, metadata_file, word_lists_file
    
    def run(self):
        """运行完整的embedding生成流程"""
        print("=" * 60)
        print("修正版单词平均Embedding生成器")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 加载captions
        captions = self.load_captions()
        
        # 3. 生成embeddings
        embeddings, word_lists = self.generate_fixed_word_average_embeddings(captions)
        
        # 4. 保存结果
        files = self.save_embeddings(embeddings, word_lists)
        
        print("\n" + "=" * 60)
        print("生成完成!")
        for file in files:
            print(f"文件: {file}")
        print("=" * 60)
        
        return embeddings, word_lists

def main():
    """主函数"""
    # 配置参数
    input_file = "Anno_Shared1000.txt"
    output_dir = "./embeddings_output_fixed"
    model_name = "all-mpnet-base-v2"
    
    print("开始处理Anno_Shared1000.txt文件...")
    print("处理方式：修正版单词平均embedding生成")
    
    # 创建生成器
    generator = FixedWordAverageEmbeddingGenerator(model_name=model_name)
    generator.caption_file = input_file
    generator.output_dir = output_dir
    
    # 运行生成流程
    embeddings, word_lists = generator.run()
    
    # 显示一些统计信息
    print(f"\n单词平均Embedding统计信息:")
    print(f"  形状: {embeddings.shape}")
    print(f"  均值: {embeddings.mean():.4f}")
    print(f"  标准差: {embeddings.std():.4f}")
    print(f"  最小值: {embeddings.min():.4f}")
    print(f"  最大值: {embeddings.max():.4f}")

if __name__ == "__main__":
    main()
