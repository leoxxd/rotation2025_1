#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版名词动词分离Embedding生成器
基于原始NSD项目的方法，修正词性标注和embedding生成问题
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

class FixedNounVerbEmbeddingGenerator:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        初始化修正版名词动词分离embedding生成器
        
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
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("正在下载 NLTK averaged_perceptron_tagger...")
            nltk.download('averaged_perceptron_tagger')
        
        # 词汇修正字典（基于原始项目的verb_adjustments）
        self.verb_adjustments = {
            # 常见的拼写错误和误分类
            'stir': 'stir',  # 在'stir fry'中，stir应该是动词
            'fry': 'fry',    # 在'stir fry'中，fry应该是动词
            'waterskiing': '_____no_embedding_____',
            'unpealed': '_____no_embedding_____',
        }
        
        # 名词修正字典
        self.noun_adjustments = {
            'stir': '_____not_noun_____',  # 在'stir fry'中，stir不是名词
            'fry': '_____not_noun_____',   # 在'stir fry'中，fry不是名词
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
    
    def get_word_type_dict(self):
        """获取词性标签字典（与原始项目一致）"""
        return {
            'noun': ['NN', 'NNS'], 
            'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        }
    
    def get_nouns_from_string(self, sentence):
        """从句子中提取名词（与原始项目一致）"""
        word_type_dict = self.get_word_type_dict()
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        
        nouns = []
        for word, pos in tagged:
            if pos in word_type_dict['noun']:
                # 应用名词修正
                if word.lower() in self.noun_adjustments:
                    if self.noun_adjustments[word.lower()] == '_____not_noun_____':
                        continue  # 跳过这个"名词"
                    else:
                        word = self.noun_adjustments[word.lower()]
                nouns.append(word.lower())
        
        return nouns
    
    def get_verbs_from_string(self, sentence):
        """从句子中提取动词（与原始项目一致）"""
        word_type_dict = self.get_word_type_dict()
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        
        verbs = []
        for word, pos in tagged:
            if pos in word_type_dict['verb']:
                # 应用动词修正
                if word.lower() in self.verb_adjustments:
                    if self.verb_adjustments[word.lower()] == '_____not_verb_/_unknown_____':
                        continue  # 跳过这个"动词"
                    elif self.verb_adjustments[word.lower()] == '_____no_embedding_____':
                        continue  # 跳过没有embedding的词
                    else:
                        word = self.verb_adjustments[word.lower()]
                verbs.append(word.lower())
        
        return verbs
    
    def get_word_embedding(self, word):
        """获取单个单词的embedding"""
        try:
            # 使用sentence transformer对单个单词生成embedding
            embedding = self.model.encode([word], convert_to_tensor=False)[0]
            return embedding
        except Exception as e:
            print(f"获取单词 '{word}' 的embedding失败: {e}")
            return None
    
    def generate_fixed_embeddings(self, captions):
        """生成修正版的embeddings"""
        print("开始生成修正版embeddings...")
        
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
        skipped_nouns = []
        skipped_verbs = []
        
        for i, image_captions in enumerate(captions):
            if i % 100 == 0:
                print(f"处理进度: {i}/{n_images} ({i/n_images*100:.1f}%)")
            
            # 收集该图片所有caption的名词和动词
            all_nouns = []
            all_verbs = []
            
            for j, caption in enumerate(image_captions):
                nouns = self.get_nouns_from_string(caption)
                verbs = self.get_verbs_from_string(caption)
                all_nouns.extend(nouns)
                all_verbs.extend(verbs)
            
            noun_lists.append(all_nouns)
            verb_lists.append(all_verbs)
            
            # 处理名词embeddings
            noun_word_embeddings = []
            for noun in all_nouns:
                noun_emb = self.get_word_embedding(noun)
                if noun_emb is not None:
                    noun_word_embeddings.append(noun_emb)
                else:
                    skipped_nouns.append(noun)
            
            if len(noun_word_embeddings) == 0:
                # 如果没有名词，使用"something"的embedding
                noun_embeddings[i] = self.get_word_embedding("something")
                no_nouns_count += 1
            else:
                noun_embeddings[i] = np.mean(noun_word_embeddings, axis=0)
                total_nouns += len(noun_word_embeddings)
            
            # 处理动词embeddings
            verb_word_embeddings = []
            for verb in all_verbs:
                verb_emb = self.get_word_embedding(verb)
                if verb_emb is not None:
                    verb_word_embeddings.append(verb_emb)
                else:
                    skipped_verbs.append(verb)
            
            if len(verb_word_embeddings) == 0:
                # 如果没有动词，使用"is"的embedding
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
        print(f"  跳过的名词: {len(skipped_nouns)}")
        print(f"  跳过的动词: {len(skipped_verbs)}")
        print(f"  Embedding维度: {embedding_dim}")
        
        return noun_embeddings, verb_embeddings, noun_lists, verb_lists
    
    def save_embeddings(self, noun_embeddings, verb_embeddings, noun_lists, verb_lists):
        """保存embeddings和元数据"""
        print("保存结果...")
        
        # 保存名词embeddings
        noun_embedding_file = os.path.join(self.output_dir, "noun_embeddings_fixed.npy")
        np.save(noun_embedding_file, noun_embeddings)
        print(f"  名词embeddings已保存: {noun_embedding_file}")
        
        # 保存动词embeddings
        verb_embedding_file = os.path.join(self.output_dir, "verb_embeddings_fixed.npy")
        np.save(verb_embedding_file, verb_embeddings)
        print(f"  动词embeddings已保存: {verb_embedding_file}")
        
        # 保存元数据
        metadata = {
            'model_name': self.model_name,
            'embedding_type': 'fixed_noun_verb_separated',
            'n_captions': len(noun_embeddings),
            'embedding_dim': noun_embeddings.shape[1],
            'caption_file': self.caption_file,
            'description': 'Fixed noun and verb embeddings with proper POS tagging and word adjustments',
            'noun_embedding_file': noun_embedding_file,
            'verb_embedding_file': verb_embedding_file
        }
        
        metadata_file = os.path.join(self.output_dir, "fixed_noun_verb_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  元数据已保存: {metadata_file}")
        
        # 保存名词列表
        noun_lists_file = os.path.join(self.output_dir, "noun_lists_fixed.pkl")
        with open(noun_lists_file, 'wb') as f:
            pickle.dump(noun_lists, f)
        print(f"  名词列表已保存: {noun_lists_file}")
        
        # 保存动词列表
        verb_lists_file = os.path.join(self.output_dir, "verb_lists_fixed.pkl")
        with open(verb_lists_file, 'wb') as f:
            pickle.dump(verb_lists, f)
        print(f"  动词列表已保存: {verb_lists_file}")
        
        return noun_embedding_file, verb_embedding_file, metadata_file, noun_lists_file, verb_lists_file
    
    def run(self):
        """运行完整的embedding生成流程"""
        print("=" * 60)
        print("修正版名词动词分离Embedding生成器")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 加载captions
        captions = self.load_captions()
        
        # 3. 生成embeddings
        noun_embeddings, verb_embeddings, noun_lists, verb_lists = self.generate_fixed_embeddings(captions)
        
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
    output_dir = "./embeddings_output_fixed"
    model_name = "all-mpnet-base-v2"
    
    print("开始处理Anno_Shared1000.txt文件...")
    print("处理方式：修正版名词动词分离embedding生成")
    
    # 创建生成器
    generator = FixedNounVerbEmbeddingGenerator(model_name=model_name)
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
