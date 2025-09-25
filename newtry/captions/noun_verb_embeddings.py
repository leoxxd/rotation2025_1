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

# 设置Hugging Face缓存目录到当前项目目录
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

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
        
        nltk_data_path = 'C:/Users/Leo/AppData/Roaming/nltk_data'
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path = [nltk_data_path]  # 覆盖其他路径
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 下载必要的 NLTK 数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("正在下载 NLTK punkt 分词器...")
            nltk.download('punkt', download_dir=nltk_data_path, force=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("正在下载 NLTK averaged_perceptron_tagger...")
            nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, force=True)
        
        # 词汇修正字典（基于原始项目的verb_adjustments）
        self.verb_adjustments = {
            "unpealed": "_____no_embedding_____",
            "pirched": "_____no_embedding_____",
            "kitboarding": "kite-surfing",
            "suckingling": "sucking",
            "reahced": "reached",
            "ealking": "leaking",
            "bursking": "_____no_embedding_____",
            "d'oeuvre": "_____not_verb_/_unknown_____",
            "PIcked": "picked",
            "wlaks": "walks",
            "Parketing": "_____not_verb_/_unknown_____",
            "plowig": "plowing",
            "igrinding": "grinding",
            "hay.surrounded": "surrounded",
            "depics": "depicts",
            "oarked": "parked",
            "toothpicked": "_____no_embedding_____",
            "staanding": "standing",
            "Delapidated": "dilapidated",
            "foilaged": "foliaged",
            "parasurfing": "_____no_embedding_____",
            "giraffestanding": "standing",
            "deckered": "_____no_embedding_____",
            "expolsed": "_____not_verb_/_unknown_____",
            "attachced": "attached",
            "holdong": "holding",
            "Horsed": "_____not_verb_/_unknown_____",
            "dessed": "dressed",
            "hook'n": "_____not_verb_/_unknown_____",
            "traveleing": "travelling",
            "rididing": "riding",
            "caryring": "carrying",
            "rcieve": "receive",
            "fluies": "_____not_verb_/_unknown_____",
            "grabbbing": "grabbing",
            "ptiched": "pitched",
            "skatboarding": "skateboarding",
            "waterskiing": "_____no_embedding_____",
            "waterskiis": "_____no_embedding_____",
            "paraskiing": "_____no_embedding_____",
            "fasioned": "_____not_verb_/_unknown_____",
            "half-covering": "covering",
            "ecorated": "_____not_verb_/_unknown_____",
            "dipicting": "depicting",
            "silhoetted": "silhouetted",
            "stoppedon": "stopped",
            "hsome": "_____not_verb_/_unknown_____",
            "deocarated": "decorated",
            "elephant.at": "_____not_verb_/_unknown_____",
            "placining": "placing",
            "shrubbs": "_____not_verb_/_unknown_____",
            "standihng": "standing",
            "irding": "riding",
            "srufing": "surfing",
            "resembing": "resembling",
            "Aproned": "_____not_verb_/_unknown_____",
            "lfits": "lifts",
            "sittinng": "sitting",
            "t=with": "_____not_verb_/_unknown_____",
            "croched": "_____not_verb_/_unknown_____",
            "standig": "_____not_verb_/_unknown_____",
            "fly-hing": "flying",
            "ehating": "heating",
            "metling": "melting",
            "signs.On": "_____not_verb_/_unknown_____",
            "stetched": "_____not_verb_/_unknown_____",
            "cocered": "_____not_verb_/_unknown_____",
            "doecarted": "decorated",
            "leanding": "_____not_verb_/_unknown_____",
            "stanging": "standing",
            "buriesd": "buried",
            "payign": "paying",
            "widnshield": "_____not_verb_/_unknown_____",
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
    
    def get_word_type_from_string_original_style(self, s, word_type):
        """
        完全复制原始NSD项目的get_word_type_from_string函数
        
        Args:
            s: 输入句子
            word_type: 词性类型 ('noun' 或 'verb')
            
        Returns:
            words: 指定词性的单词列表
        """
        # 复制原始项目的get_word_type_dict函数
        word_type_dict = {
            'noun': ['NN', 'NNS'], 
            'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        }
        
        # 复制原始项目的get_sentence_tags函数
        tokens = nltk.word_tokenize(s)  # 不进行大小写转换
        tagged = nltk.pos_tag(tokens)
        
        # 注意：原始项目的get_word_type_from_string函数没有应用词汇修正
        # 词汇修正只在动词embedding生成时单独处理
        return [x[0] for x in tagged if x[1] in word_type_dict[word_type]]
    
    def get_verbs_from_string_original_style(self, s):
        """
        复制原始NSD项目的动词提取方法，包含词汇修正
        
        Args:
            s: 输入句子
            
        Returns:
            verbs: 动词列表
        """
        # 复制原始项目的get_verbs_from_string函数
        tokens = nltk.word_tokenize(s)
        tagged = nltk.pos_tag(tokens)
        verbs = [x[0] for x in tagged if x[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
        
        # 应用词汇修正（与原始方法一致）
        corrected_verbs = []
        for verb in verbs:
            if verb in self.verb_adjustments:
                if self.verb_adjustments[verb] == "_____not_verb_/_unknown_____":
                    continue  # 跳过这个词汇
                elif self.verb_adjustments[verb] == "_____no_embedding_____":
                    continue  # 跳过没有embedding的词汇
                else:
                    verb = self.verb_adjustments[verb]
            corrected_verbs.append(verb)
        
        return corrected_verbs

    def extract_nouns_and_verbs_original_style(self, caption):
        """
        使用与原始NSD项目一致的方法提取名词和动词
        
        Args:
            caption: 输入文本
            
        Returns:
            nouns: 名词列表
            verbs: 动词列表
        """
        try:
            # 名词：不应用词汇修正
            nouns = self.get_word_type_from_string_original_style(caption, 'noun')
            # 动词：应用词汇修正
            verbs = self.get_verbs_from_string_original_style(caption)
            
            return nouns, verbs
            
        except Exception as e:
            print(f"  ⚠️ 词性标注失败: {e}")
            # 如果NLTK失败，使用简单的分词作为备选
            tokens = nltk.word_tokenize(caption.lower())
            filtered_tokens = [w for w in tokens if len(w) > 1 and w.isalpha()]
            # 简单规则：假设所有词都是名词（保守估计）
            return filtered_tokens, []
    
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
    
    def generate_noun_verb_embeddings(self, captions, normalize_embeddings=True):
        """
        生成名词和动词的embeddings
        
        Args:
            captions: caption列表，每个元素是一个图片的多句caption列表
            normalize_embeddings: 是否对embedding进行z-score归一化
            
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
                nouns, verbs = self.extract_nouns_and_verbs_original_style(caption)
                all_nouns.extend(nouns)
                all_verbs.extend(verbs)
            
            # 保留重复词：与原始代码保持一致，保留词频信息
            noun_lists.append(all_nouns)
            verb_lists.append(all_verbs)
            
            # 处理名词
            noun_word_embeddings = []
            for noun in all_nouns:
                try:
                    noun_emb = self.get_word_embedding(noun)
                    noun_word_embeddings.append(noun_emb)
                except Exception as e:
                    print(f"  ⚠️ 名词 '{noun}' 生成embedding失败: {e}")
                    continue
            
            if len(noun_word_embeddings) == 0:
                # 如果没有名词或所有名词embedding生成失败，使用"something"的embedding
                noun_embeddings[i] = self.get_word_embedding("something")
                no_nouns_count += 1
            else:
                noun_embeddings[i] = np.mean(noun_word_embeddings, axis=0)
                total_nouns += len(noun_word_embeddings)
            
            # 处理动词
            verb_word_embeddings = []
            for verb in all_verbs:
                try:
                    verb_emb = self.get_word_embedding(verb)
                    verb_word_embeddings.append(verb_emb)
                except Exception as e:
                    print(f"  ⚠️ 动词 '{verb}' 生成embedding失败: {e}")
                    continue
            
            if len(verb_word_embeddings) == 0:
                # 如果没有动词或所有动词embedding生成失败，使用"is"的embedding
                verb_embeddings[i] = self.get_word_embedding("is")
                no_verbs_count += 1
            else:
                verb_embeddings[i] = np.mean(verb_word_embeddings, axis=0)
                total_verbs += len(verb_word_embeddings)
        
        # 可选的embedding归一化
        if normalize_embeddings:
            print("\n对embeddings进行z-score归一化...")
            # 对名词embeddings进行z-score归一化
            noun_mean = np.mean(noun_embeddings, axis=0)
            noun_std = np.std(noun_embeddings, axis=0)
            noun_embeddings = (noun_embeddings - noun_mean) / (noun_std + 1e-8)  # 添加小常数避免除零
            
            # 对动词embeddings进行z-score归一化
            verb_mean = np.mean(verb_embeddings, axis=0)
            verb_std = np.std(verb_embeddings, axis=0)
            verb_embeddings = (verb_embeddings - verb_mean) / (verb_std + 1e-8)  # 添加小常数避免除零
            
            print(f"  名词embedding归一化: mean={np.mean(noun_embeddings):.6f}, std={np.std(noun_embeddings):.6f}")
            print(f"  动词embedding归一化: mean={np.mean(verb_embeddings):.6f}, std={np.std(verb_embeddings):.6f}")
        
        print(f"\n生成完成!")
        print(f"  总图片数: {n_images}")
        print(f"  总名词数: {total_nouns}")
        print(f"  总动词数: {total_verbs}")
        print(f"  平均每图片名词数: {total_nouns/n_images:.2f}")
        print(f"  平均每图片动词数: {total_verbs/n_images:.2f}")
        print(f"  无名词图片数: {no_nouns_count}")
        print(f"  无动词图片数: {no_verbs_count}")
        print(f"  Embedding维度: {embedding_dim}")
        print(f"  归一化: {'是' if normalize_embeddings else '否'}")
        
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
        
        # 3. 生成embeddings（默认启用归一化）
        noun_embeddings, verb_embeddings, noun_lists, verb_lists = self.generate_noun_verb_embeddings(captions, normalize_embeddings=True)
        
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
    # 配置参数
    input_file = "Anno_Shared1000.txt"
    output_dir = "./embeddings_output"
    model_name = "all-mpnet-base-v2"
    
    print("开始处理Anno_Shared1000.txt文件...")
    print("处理方式：每张图像的5个caption分词后，分别提取名词和动词，生成对应的embedding")
    
    # 创建生成器
    generator = NounVerbEmbeddingGenerator(model_name=model_name)
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
