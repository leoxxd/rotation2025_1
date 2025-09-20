#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词性标注Embedding生成器
将caption分词后，分别提取名词和动词，生成对应的embedding
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

class POSEmbeddingGenerator:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        初始化词性标注embedding生成器
        
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
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("下载NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("下载NLTK averaged_perceptron_tagger...")
            nltk.download('averaged_perceptron_tagger')
    
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
    
    def extract_pos_words(self, caption):
        """
        从caption中提取名词和动词
        
        Args:
            caption: 输入文本
            
        Returns:
            nouns: 名词列表
            verbs: 动词列表
        """
        # 使用NLTK进行分词和词性标注
        tokens = nltk.word_tokenize(caption.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        nouns = []
        verbs = []
        
        for word, pos in pos_tags:
            # 过滤掉标点符号和短词
            if len(word) > 1 and word.isalpha():
                # 名词标签 (NN, NNS, NNP, NNPS)
                if pos.startswith('NN'):
                    nouns.append(word)
                # 动词标签 (VB, VBD, VBG, VBN, VBP, VBZ)
                elif pos.startswith('VB'):
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
    
    def generate_pos_embeddings(self, captions):
        """
        生成词性标注embeddings
        
        Args:
            captions: caption列表，每个元素是一个图片的多句caption列表
            
        Returns:
            noun_embeddings: 名词embeddings数组
            verb_embeddings: 动词embeddings数组
            noun_lists: 每个图片的名词列表
            verb_lists: 每个图片的动词列表
        """
        print("开始生成词性标注embeddings...")
        
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
        skipped_noun_images = 0
        skipped_verb_images = 0
        
        for i, image_captions in enumerate(tqdm(captions, desc="处理图片")):
            
            # 收集该图片所有caption的名词和动词
            all_nouns = []
            all_verbs = []
            
            for j, caption in enumerate(image_captions):
                nouns, verbs = self.extract_pos_words(caption)
                all_nouns.extend(nouns)
                all_verbs.extend(verbs)
            
            noun_lists.append(all_nouns)
            verb_lists.append(all_verbs)
            
            # 处理名词embeddings
            if len(all_nouns) == 0:
                # 如果没有名词，使用空字符串的embedding
                print(f"  ⚠️ 图片 {i} 没有名词")
                noun_embeddings[i] = self.get_word_embedding("")
                skipped_noun_images += 1
            else:
                # 为每个名词生成embedding
                noun_word_embeddings = []
                for noun in all_nouns:
                    try:
                        noun_emb = self.get_word_embedding(noun)
                        noun_word_embeddings.append(noun_emb)
                    except Exception as e:
                        print(f"  ⚠️ 名词 '{noun}' 生成embedding失败: {e}")
                        continue
                
                if len(noun_word_embeddings) == 0:
                    print(f"  ⚠️ 图片 {i} 所有名词embedding生成失败")
                    noun_embeddings[i] = self.get_word_embedding("")
                    skipped_noun_images += 1
                else:
                    # 计算名词embeddings的平均值
                    noun_embeddings[i] = np.mean(noun_word_embeddings, axis=0)
                    total_nouns += len(noun_word_embeddings)
            
            # 处理动词embeddings
            if len(all_verbs) == 0:
                # 如果没有动词，使用空字符串的embedding
                print(f"  ⚠️ 图片 {i} 没有动词")
                verb_embeddings[i] = self.get_word_embedding("")
                skipped_verb_images += 1
            else:
                # 为每个动词生成embedding
                verb_word_embeddings = []
                for verb in all_verbs:
                    try:
                        verb_emb = self.get_word_embedding(verb)
                        verb_word_embeddings.append(verb_emb)
                    except Exception as e:
                        print(f"  ⚠️ 动词 '{verb}' 生成embedding失败: {e}")
                        continue
                
                if len(verb_word_embeddings) == 0:
                    print(f"  ⚠️ 图片 {i} 所有动词embedding生成失败")
                    verb_embeddings[i] = self.get_word_embedding("")
                    skipped_verb_images += 1
                else:
                    # 计算动词embeddings的平均值
                    verb_embeddings[i] = np.mean(verb_word_embeddings, axis=0)
                    total_verbs += len(verb_word_embeddings)
        
        print(f"\n生成完成!")
        print(f"  总图片数: {n_images}")
        print(f"  总名词数: {total_nouns}")
        print(f"  总动词数: {total_verbs}")
        print(f"  平均每图片名词数: {total_nouns/n_images:.2f}")
        print(f"  平均每图片动词数: {total_verbs/n_images:.2f}")
        print(f"  跳过名词图片数: {skipped_noun_images}")
        print(f"  跳过动词图片数: {skipped_verb_images}")
        print(f"  Embedding维度: {embedding_dim}")
        
        return noun_embeddings, verb_embeddings, noun_lists, verb_lists
    
    def save_embeddings(self, noun_embeddings, verb_embeddings, noun_lists, verb_lists):
        """保存embeddings和元数据"""
        print("保存结果...")
        
        # 保存名词embeddings
        noun_embedding_file = os.path.join(self.output_dir, "noun_embeddings.npy")
        np.save(noun_embedding_file, noun_embeddings)
        print(f"  名词Embeddings已保存: {noun_embedding_file}")
        
        # 保存动词embeddings
        verb_embedding_file = os.path.join(self.output_dir, "verb_embeddings.npy")
        np.save(verb_embedding_file, verb_embeddings)
        print(f"  动词Embeddings已保存: {verb_embedding_file}")
        
        # 保存名词元数据
        noun_metadata = {
            'model_name': self.model_name,
            'embedding_type': 'noun_average',
            'n_captions': len(noun_embeddings),
            'embedding_dim': noun_embeddings.shape[1],
            'caption_file': self.caption_file,
            'description': 'Noun-level embeddings averaged across all nouns in each caption'
        }
        
        noun_metadata_file = os.path.join(self.output_dir, "noun_metadata.pkl")
        with open(noun_metadata_file, 'wb') as f:
            pickle.dump(noun_metadata, f)
        print(f"  名词元数据已保存: {noun_metadata_file}")
        
        # 保存动词元数据
        verb_metadata = {
            'model_name': self.model_name,
            'embedding_type': 'verb_average',
            'n_captions': len(verb_embeddings),
            'embedding_dim': verb_embeddings.shape[1],
            'caption_file': self.caption_file,
            'description': 'Verb-level embeddings averaged across all verbs in each caption'
        }
        
        verb_metadata_file = os.path.join(self.output_dir, "verb_metadata.pkl")
        with open(verb_metadata_file, 'wb') as f:
            pickle.dump(verb_metadata, f)
        print(f"  动词元数据已保存: {verb_metadata_file}")
        
        # 保存词性列表
        pos_lists_file = os.path.join(self.output_dir, "pos_lists.pkl")
        pos_data = {
            'noun_lists': noun_lists,
            'verb_lists': verb_lists
        }
        with open(pos_lists_file, 'wb') as f:
            pickle.dump(pos_data, f)
        print(f"  词性列表已保存: {pos_lists_file}")
        
        return noun_embedding_file, verb_embedding_file, noun_metadata_file, verb_metadata_file, pos_lists_file
    
    def run(self):
        """运行完整的embedding生成流程"""
        print("=" * 60)
        print("词性标注Embedding生成器")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 加载captions
        captions = self.load_captions()
        
        # 3. 生成embeddings
        noun_embeddings, verb_embeddings, noun_lists, verb_lists = self.generate_pos_embeddings(captions)
        
        # 4. 保存结果
        files = self.save_embeddings(noun_embeddings, verb_embeddings, noun_lists, verb_lists)
        
        print("\n" + "=" * 60)
        print("生成完成!")
        for file in files:
            print(f"文件: {file}")
        print("=" * 60)
        
        return noun_embeddings, verb_embeddings, noun_lists, verb_lists

def main():
    """主函数"""
    # 配置参数
    input_file = "Anno_Shared1000.txt"
    output_dir = "./embeddings_output"
    model_name = "all-mpnet-base-v2"
    
    print("开始处理Anno_Shared1000.txt文件...")
    print("处理方式：每张图像的5个caption分词后，分别提取名词和动词，生成对应的embedding")
    
    # 创建生成器
    generator = POSEmbeddingGenerator(model_name=model_name)
    generator.caption_file = input_file
    generator.output_dir = output_dir
    
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
