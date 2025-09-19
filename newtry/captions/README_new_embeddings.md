# 新的Caption Embedding方法

本目录包含两种新的caption embedding生成方法，基于`anno_shared_1000.txt`文件。

## 📁 文件说明

### 脚本文件
- `word_average_embeddings.py` - 单词平均embedding生成器
- `noun_verb_embeddings.py` - 名词动词分离embedding生成器

### 数据文件
- `anno_shared_1000.txt` - 1000个共享图片的caption文件
- `embeddings_output/` - 输出目录

## 🔧 方法1：单词平均Embedding

### 原理
1. 对每个图片的多句caption分别进行分词
2. 收集该图片所有caption的单词
3. 对每个单词单独生成embedding
4. 将所有单词的embedding取平均作为该图片的最终embedding

### 使用方法
```bash
cd E:\lunzhuan1\visuo_llm-main\newtry\captions
python word_average_embeddings.py
```

### 输出文件
- `embeddings_output/word_average_embeddings.npy` - 单词平均embeddings (1000, 768)
- `embeddings_output/word_average_metadata.pkl` - 元数据
- `embeddings_output/word_lists.pkl` - 每个图片的单词列表

### 特点
- ✅ 保留所有单词信息
- ✅ 通过平均减少噪声
- ✅ 适合词汇丰富的caption

## 🔧 方法2：名词动词分离Embedding

### 原理
1. 对每个图片的多句caption使用NLTK进行词性标注
2. 分别提取所有caption中的名词和动词
3. 为名词和动词分别生成embedding并取平均
4. 生成两个独立的embedding向量

### 使用方法
```bash
cd E:\lunzhuan1\visuo_llm-main\newtry\captions
python noun_verb_embeddings.py
```

### 输出文件
- `embeddings_output/noun_embeddings.npy` - 名词embeddings (1000, 768)
- `embeddings_output/verb_embeddings.npy` - 动词embeddings (1000, 768)
- `embeddings_output/noun_verb_metadata.pkl` - 元数据
- `embeddings_output/noun_lists.pkl` - 每个图片的名词列表
- `embeddings_output/verb_lists.pkl` - 每个图片的动词列表

### 特点
- ✅ 分离语义和动作信息
- ✅ 适合分析物体vs动作的神经表征
- ✅ 提供更细粒度的语义分析

## 📊 技术细节

### 多句Caption处理
- **支持格式**：pickle文件（多句caption）或文本文件（单句caption）
- **处理方式**：对每个图片的所有caption进行合并处理
- **单词收集**：收集该图片所有caption的单词，然后统一处理
- **与原始项目一致**：参考NSD项目的处理方式

### 模型
- 使用 `all-mpnet-base-v2` sentence transformer模型
- Embedding维度：768
- 支持中英文混合文本

### 词性标注
- 名词标签：`NN`, `NNS`, `NNP`, `NNPS`
- 动词标签：`VB`, `VBD`, `VBG`, `VBN`, `VBP`, `VBZ`

### 处理策略
- 自动过滤标点符号和短词
- 处理无名词/动词的情况（使用默认词）
- 错误处理和统计信息

## 🔄 与原始方法的对比

| 方法 | 输入 | 处理方式 | 输出 | 适用场景 |
|------|------|----------|------|----------|
| **原始方法** | 完整caption | 直接embedding | 1个向量 | 整体语义 |
| **单词平均** | 分词后单词 | 单词embedding平均 | 1个向量 | 词汇语义 |
| **名词动词** | 名词+动词 | 分别embedding | 2个向量 | 语义vs动作 |

## 🚀 使用建议

### 选择方法
- **整体语义分析** → 使用原始方法
- **词汇丰富度分析** → 使用单词平均方法
- **物体vs动作分析** → 使用名词动词分离方法

### 后续分析
1. 可以计算RDM进行RSA分析
2. 可以比较不同方法的神经相关性
3. 可以分析不同脑区对不同embedding类型的偏好

## 📝 示例代码

```python
import numpy as np
import pickle

# 加载单词平均embeddings
word_avg_embeddings = np.load('embeddings_output/word_average_embeddings.npy')

# 加载名词动词embeddings
noun_embeddings = np.load('embeddings_output/noun_embeddings.npy')
verb_embeddings = np.load('embeddings_output/verb_embeddings.npy')

# 加载元数据
with open('embeddings_output/word_average_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Embedding形状: {word_avg_embeddings.shape}")
print(f"模型: {metadata['model_name']}")
print(f"类型: {metadata['embedding_type']}")
```

## ⚠️ 注意事项

1. 确保已安装必要的依赖包
2. 首次运行会自动下载NLTK数据
3. 处理时间取决于caption数量和模型大小
4. 建议在GPU上运行以提高速度
