# Human fMRI RSA分析 - 多种Embedding类型选择

这个脚本支持对human fMRI数据进行RSA分析，可以选择不同的embedding类型进行比较。

## 支持的Embedding类型

1. **image_embeddings** - 完整caption embedding
2. **word_average_embeddings** - 单词平均embedding  
3. **noun_embeddings** - 名词embedding
4. **verb_embeddings** - 动词embedding

## 使用方法

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

### 高级选项

```bash
# 使用原始fMRI数据（不使用z-score归一化）
python rsa_with_embedding_choice.py --embedding_type image --no-use_zscore

# 使用z-score归一化的fMRI数据（默认）
python rsa_with_embedding_choice.py --embedding_type image --use_zscore
```

## 输出文件

每个分析会生成以下文件：

### 结果文件
- `rsa_results_{embedding_type}_{normalization}.csv` - 详细结果表格
- `rsa_results_{embedding_type}_{normalization}.mat` - MATLAB格式结果
- `rsa_results_{embedding_type}_{normalization}.pkl` - Python pickle格式结果

### 图形文件
- `rsa_analysis_{embedding_type}_{normalization}.png` - RSA分析结果图

## 输出目录结构

```
rsa_results_{embedding_type}_{normalization}/
├── rsa_results_{embedding_type}_{normalization}.csv
├── rsa_results_{embedding_type}_{normalization}.mat
├── rsa_results_{embedding_type}_{normalization}.pkl
└── rsa_analysis_{embedding_type}_{normalization}.png
```

## 结果解释

### CSV文件列说明
- **ROI**: ROI标签
- **Subject**: 被试ID
- **Correlation**: RSA相关性（Spearman相关系数）
- **P_Value**: 统计显著性p值
- **Embedding_Type**: 使用的embedding类型
- **Use_Zscore**: 是否使用z-score归一化

### 图形说明
- **左图**: ROI级别平均RSA相关性条形图（带误差线）
- **右图**: 所有被试RSA相关性分布直方图

## 数据要求

### Embedding文件
需要以下embedding文件存在于 `../captions/embeddings_output/` 目录：
- `image_embeddings.npy`
- `word_average_embeddings.npy`
- `noun_embeddings.npy`
- `verb_embeddings.npy`

### fMRI数据文件
需要以下文件存在于当前目录：
- `roi_rdm_results/all_subjects_roi_rdms.mat`

## 示例分析流程

```bash
# 1. 分析完整caption embedding
python rsa_with_embedding_choice.py --embedding_type image --use_zscore

# 2. 分析单词平均embedding
python rsa_with_embedding_choice.py --embedding_type word_average --use_zscore

# 3. 分析名词embedding
python rsa_with_embedding_choice.py --embedding_type noun --use_zscore

# 4. 分析动词embedding
python rsa_with_embedding_choice.py --embedding_type verb --use_zscore
```

## 注意事项

1. **数据路径**: 确保embedding文件路径正确
2. **内存使用**: 大型数据集可能需要较多内存
3. **计算时间**: 分析时间取决于ROI数量和被试数量
4. **结果解释**: 相关性越高表示模型与大脑表示越相似

## 故障排除

### 常见错误
1. **文件不存在**: 检查embedding文件和fMRI数据文件路径
2. **内存不足**: 尝试使用更小的数据集或增加系统内存
3. **权限错误**: 确保有写入输出目录的权限

### 调试模式
在脚本中添加调试信息：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 联系信息

如有问题，请检查：
1. 数据文件是否存在且格式正确
2. Python环境和依赖包是否完整
3. 文件路径是否正确
