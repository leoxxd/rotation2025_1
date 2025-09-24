# 猴子神经元RSA分析 - 多Embedding类型支持

## 🎯 功能概述

`rsa_with_embedding_choice.py` 是一个支持多种embedding类型的猴子神经元RSA分析工具，可以方便地比较不同embedding方法与猴子神经活动的相关性。

## 🔧 支持的Embedding类型

| Embedding类型 | 文件名 | 描述 |
|---------------|--------|------|
| `image` | `image_embeddings.npy` | 完整Caption Embedding |
| `word_average` | `word_average_embeddings.npy` | 单词平均Embedding |
| `noun` | `noun_embeddings.npy` | 名词Embedding |
| `verb` | `verb_embeddings.npy` | 动词Embedding |

## 🚀 使用方法

### 基本用法

```bash
# 使用完整caption embedding
python rsa_with_embedding_choice.py --embedding_type image

# 使用单词平均embedding
python rsa_with_embedding_choice.py --embedding_type word_average

# 使用名词embedding
python rsa_with_embedding_choice.py --embedding_type noun

# 使用动词embedding
python rsa_with_embedding_choice.py --embedding_type verb
```

### 数据归一化选项

```bash
# 使用z-score归一化的猴子数据（默认）
python rsa_with_embedding_choice.py --embedding_type image --use_zscore

# 使用原始猴子数据
python rsa_with_embedding_choice.py --embedding_type image --no_zscore
```

### 完整示例

```bash
# 分析名词embedding与z-score归一化猴子数据的相关性
python rsa_with_embedding_choice.py --embedding_type noun --use_zscore

# 分析动词embedding与原始猴子数据的相关性
python rsa_with_embedding_choice.py --embedding_type verb --no_zscore
```

## 📁 输出文件结构

程序会根据选择的embedding类型和归一化方法自动创建相应的输出目录：

```
rsa_results_{embedding_type}_{normalization}/
├── rsa_{arealabel}_{embedding_type}_{normalization}.png    # 每个arealabel的详细分析图
├── rsa_table_{arealabel}_{embedding_type}_{normalization}.csv  # 每个arealabel的详细数据表
├── rsa_summary_{embedding_type}_{normalization}.png        # 汇总分析图
└── rsa_summary_{embedding_type}_{normalization}.csv        # 汇总数据表
```

### 输出文件示例

- `rsa_results_image_zscore/` - 完整caption embedding + z-score归一化
- `rsa_results_noun_original/` - 名词embedding + 原始数据
- `rsa_results_verb_zscore/` - 动词embedding + z-score归一化

## 📊 分析内容

### 1. 每个Arealabel的详细分析
- **原始RSA值**：猴子RDM与embedding RDM的直接相关性
- **矫正RSA值**：原始RSA除以噪声天花板
- **噪声天花板**：该arealabel内不同session之间的平均相关性
- **矫正比例**：矫正效果的可视化

### 2. 汇总分析
- **所有arealabel的RSA比较**
- **噪声天花板分布**
- **矫正效果对比**
- **排序后的结果**

## 🔍 结果解读

### 主要指标

1. **原始RSA**：直接的相关性，通常在0.1-0.3之间
2. **矫正RSA**：噪声天花板校正后的相关性，可能>1.0
3. **噪声天花板**：该脑区的最大可能相关性
4. **矫正比例**：矫正效果，>1表示矫正后相关性提高

### 不同Embedding类型的预期结果

- **完整Caption Embedding**：整体语义相关性
- **单词平均Embedding**：词汇层面的相关性
- **名词Embedding**：物体/概念层面的相关性
- **动词Embedding**：动作/行为层面的相关性

## ⚙️ 技术细节

### 数据流程
1. 加载选择的embedding文件
2. 加载猴子RDM数据（原始或z-score归一化）
3. 按arealabel分组分析
4. 计算噪声天花板
5. 进行RSA分析和矫正
6. 生成可视化结果

### 噪声天花板计算
- 每个session的RDM与其他session平均RDM的相关性
- 用于矫正个体差异和测量噪声

### 相关性计算
- 使用Spearman相关系数
- 基于RDM的上三角矩阵（排除对角线）

## 🎯 使用建议

### 选择Embedding类型
- **整体语义分析** → 使用 `image`
- **词汇丰富度分析** → 使用 `word_average`
- **物体vs动作分析** → 使用 `noun` 和 `verb`
- **比较不同语义层次** → 依次运行所有类型

### 选择归一化方法
- **消除个体差异** → 使用 `--use_zscore`
- **保持原始信号** → 使用 `--no_zscore`
- **对比分析** → 两种方法都运行

## 📋 依赖文件

### 必需文件
- `all_rdms_correlation.pkl` 或 `all_rdms_correlation_zscore.pkl`
- `extracted_monkey_responses.pkl`
- `embeddings_output/image_embeddings.npy`
- `embeddings_output/word_average_embeddings.npy`
- `embeddings_output/noun_embeddings.npy`
- `embeddings_output/verb_embeddings.npy`

### 可选文件
- 如果某些embedding文件不存在，程序会报错并提示

## 🚨 注意事项

1. **文件路径**：确保embedding文件路径正确
2. **数据完整性**：确保所有必需的.pkl文件存在
3. **内存使用**：处理大量数据时注意内存使用
4. **结果解释**：矫正后RSA > 1.0是正常现象

## 🔄 批量分析示例

```bash
# 分析所有embedding类型（z-score归一化）
for embedding in image word_average noun verb; do
    echo "分析 $embedding embedding..."
    python rsa_with_embedding_choice.py --embedding_type $embedding --use_zscore
done

# 分析所有embedding类型（原始数据）
for embedding in image word_average noun verb; do
    echo "分析 $embedding embedding（原始数据）..."
    python rsa_with_embedding_choice.py --embedding_type $embedding --no_zscore
done
```

## 📈 预期结果

运行完成后，您将得到：
- 每个arealabel的详细分析图表
- 汇总的对比分析图表
- 详细的CSV数据表
- 完整的分析结果pickle文件

这些结果可以帮助您：
- 比较不同embedding类型的神经相关性
- 识别对不同语义层次敏感的脑区
- 分析噪声天花板和矫正效果
- 为后续研究提供数据支持
