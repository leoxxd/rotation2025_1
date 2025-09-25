#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版单词平均Embedding生成器
基于原始NSD项目的方法，修正分词和embedding生成问题
将caption分词后，对每个单词生成embedding，然后取平均
"""

import os
import pickle
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置Hugging Face缓存目录到当前项目目录
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

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
            print("正在下载 NLTK punkt 分词器...")
            nltk.download('punkt')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("正在下载 NLTK averaged_perceptron_tagger...")
            nltk.download('averaged_perceptron_tagger')
        
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
            print("请尝试以下解决方案:")
            print("1. 设置环境变量: set HF_ENDPOINT=https://hf-mirror.com")
            print("2. 手动下载模型到 ./models/all-mpnet-base-v2/ 目录")
            print("3. 使用其他embedding模型")
            raise e
        
        print(f"模型加载完成，embedding维度: {self.model.get_sentence_embedding_dimension()}")
    
    def load_captions(self):
        """加载caption文件 - 与simple_extract_embeddings.py完全一致"""
        print(f"正在加载数据: {self.caption_file}")
        
        if not os.path.exists(self.caption_file):
            raise FileNotFoundError(f"Caption文件不存在: {self.caption_file}")
        
        # 按图像分组存储数据
        image_data = {}  # {image_id: [caption1, caption2, caption3, caption4, caption5]}
        
        with open(self.caption_file, 'r', encoding='utf-8') as f:
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
        使用与原始NSD项目一致的分词方式
        
        Args:
            caption: 输入文本
            
        Returns:
            tokens: 分词后的单词列表
        """
        # 使用NLTK进行分词（与原始方法一致，不进行大小写转换）
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
    
    def generate_word_average_embeddings(self, captions, normalize_embeddings=True):
        """
        生成单词平均embeddings
        
        Args:
            captions: caption列表，每个元素是一个图片的多句caption列表
            normalize_embeddings: 是否对embedding进行z-score归一化
            
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
        
        for i, image_captions in enumerate(tqdm(captions, desc="处理图片")):
            
            # 收集该图片所有caption的单词
            all_words = []
            for j, caption in enumerate(image_captions):
                words = self.tokenize_caption_original_style(caption)
                all_words.extend(words)
            
            word_lists.append(all_words)
            
            if len(all_words) == 0:
                # 如果没有有效单词，使用空字符串的embedding
                print(f"  ⚠️ 图片 {i} 没有有效单词")
                embeddings[i] = self.get_word_embedding("")
                skipped_images += 1
                continue
            
            # 为每个单词生成embedding（与原始方法一致）
            word_embeddings = []
            for word in all_words:
                try:
                    # 使用与原始方法一致的embedding生成方式
                    word_emb = self.get_word_embedding(word)
                    if word_emb is not None:
                        word_embeddings.append(word_emb)
                        total_words += 1
                except Exception as e:
                    # 如果单词不存在embedding，跳过（与原始方法一致）
                    continue
            
            if len(word_embeddings) == 0:
                # 如果没有有效单词，使用空字符串的embedding
                print(f"  ⚠️ 图片 {i} 没有有效单词")
                embeddings[i] = self.get_word_embedding("")
                skipped_images += 1
            else:
                # 计算单词embeddings的平均值（与原始方法一致）
                embeddings[i] = np.mean(np.asarray(word_embeddings), axis=0)
        
        # 可选的embedding归一化
        if normalize_embeddings:
            print("\n对embeddings进行z-score归一化...")
            # 对embeddings进行z-score归一化
            embedding_mean = np.mean(embeddings, axis=0)
            embedding_std = np.std(embeddings, axis=0)
            embeddings = (embeddings - embedding_mean) / (embedding_std + 1e-8)  # 添加小常数避免除零
            
            print(f"  Embedding归一化: mean={np.mean(embeddings):.6f}, std={np.std(embeddings):.6f}")
        
        print(f"\n生成完成!")
        print(f"  总图片数: {n_images}")
        print(f"  总单词数: {total_words}")
        print(f"  平均每图片单词数: {total_words/n_images:.2f}")
        print(f"  跳过图片数: {skipped_images}")
        print(f"  Embedding维度: {embedding_dim}")
        print(f"  归一化: {'是' if normalize_embeddings else '否'}")
        
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
        
        # 3. 生成embeddings（默认启用归一化）
        embeddings, word_lists = self.generate_word_average_embeddings(captions, normalize_embeddings=True)
        
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
    """主函数 - 与simple_extract_embeddings.py完全一致"""
    # 配置参数
    input_file = "Anno_Shared1000.txt"
    output_dir = "./embeddings_output"
    model_name = "all-mpnet-base-v2"
    
    print("开始处理Anno_Shared1000.txt文件...")
    print("处理方式：修正版单词平均embedding生成（与原始NSD方法一致）")
    
    # 创建生成器
    generator = WordAverageEmbeddingGenerator(model_name=model_name)
    generator.caption_file = input_file
    generator.output_dir = output_dir
    
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
