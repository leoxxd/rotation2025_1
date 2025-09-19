# 简化版RSA分析器使用指南

## 🎯 功能概述

`rsa_analyzer_simple.py` 是一个简化版的RSA分析工具，具有以下特点：

### ✨ 主要特点
1. **直接使用1000张图片**：不做采样，使用全部图片计算RDM
2. **噪声天花板校正**：基于fMRI数据计算噪声天花板
3. **按被试和ROI分开保存**：结果组织清晰
4. **丰富的可视化**：生成多种图表
5. **简单直接**：无复杂的采样逻辑

## 🚀 使用方法

### 1. 运行分析
```bash
cd newtry
python rsa_analyzer_simple.py
```

### 2. 输出结构
```
rsa_results_simple/
├── rsa_results_simple.mat                    # 所有结果（MATLAB格式）
├── rsa_results_simple.csv                    # 所有结果（CSV格式）
├── simple_analysis.txt                       # 详细分析报告
├── by_subject/                               # 按被试分开
│   ├── s1_results.mat & s1_results.csv
│   ├── s2_results.mat & s2_results.csv
│   ├── s5_results.mat & s5_results.csv
│   └── s7_results.mat & s7_results.csv
├── by_roi/                                   # 按ROI分开
│   ├── roi_1_results.mat & roi_1_results.csv
│   ├── roi_2_results.mat & roi_2_results.csv
│   └── ...
└── plots/                                    # 可视化图表
    ├── correlation_comparison.png            # 相关性比较
    ├── roi_comparison.png                    # ROI比较
    ├── subject_comparison.png                # 被试比较
    ├── s1_roi_analysis.png                   # s1被试的ROI分析
    ├── s2_roi_analysis.png                   # s2被试的ROI分析
    ├── s5_roi_analysis.png                   # s5被试的ROI分析
    └── s7_roi_analysis.png                   # s7被试的ROI分析
```

## 📊 分析方法

### 1. **直接计算RDM**
- 使用全部1000张图片计算embedding RDM
- 使用全部1000张图片计算ROI RDM
- 直接计算两个RDM的相关性

### 2. **噪声天花板校正**
```python
# 噪声天花板 = 该被试fMRI RDM与其他被试平均fMRI RDM的相关性
noise_ceiling = corr_rdms(subject_rdm, mean_other_subjects_rdm)

# 校正后相关性
corrected_correlation = raw_correlation / noise_ceiling
```

### 3. **与传统方法对比**
- 同时计算Spearman相关性作为对比
- 提供多种相关性指标

## 🔍 结果解读

### 1. **原始相关性**
- 直接计算的相关性（无校正）
- 通常较低（0.1-0.3）

### 2. **校正后相关性**
- 噪声天花板校正后的相关性
- 可能 > 1.0（表示超过噪声天花板）
- 通常更高（0.3-0.5）

### 3. **噪声天花板**
- 该ROI在该被试中的最大可能相关性
- 通常在0.3-0.8之间

## 📈 可视化图表

### 1. **相关性比较图**
- 原始 vs 校正后相关性的散点图
- 显示校正效果

### 2. **ROI比较图**
- 按脑区分组的箱线图
- 比较不同脑区的表现

### 3. **被试比较图**
- 按被试分组的箱线图
- 比较不同被试的表现

### 4. **个别被试ROI分析图**
- 每个被试的详细ROI分析
- 4个子图：原始相关性、校正后相关性、噪声天花板、校正效果

## 🎯 优势

### 1. **简单直接**
- 无复杂采样逻辑
- 使用全部数据
- 计算速度快

### 2. **结果稳定**
- 无随机采样影响
- 结果可重复
- 数据利用率高

### 3. **易于理解**
- 逻辑清晰
- 结果直观
- 便于解释

## ⚠️ 注意事项

### 1. **计算资源**
- 使用全部1000张图片
- 内存需求较高
- 计算时间较长

### 2. **结果解释**
- 校正后相关性可能 > 1.0
- 这是正常现象，表示超过噪声天花板
- 不需要人为限制

### 3. **数据质量**
- 需要确保ROI和embedding数据质量
- 噪声天花板过低可能表明数据问题

## 🔧 技术细节

### 1. **相关性计算**
```python
def corr_rdms(X, Y):
    """原始项目的相关性计算函数"""
    X = X - X.mean(axis=1, keepdims=True)
    X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y /= np.sqrt(np.einsum("ij,ij->i", Y, Y))[:, None]
    return np.einsum("ik,jk", X, Y)
```

### 2. **RDM计算**
```python
# embedding RDM
embedding_rdm = pdist(embeddings, metric='correlation')

# ROI RDM (从.mat文件加载)
roi_rdm = roi_info['rdm'][0, 0].flatten()
```

### 3. **噪声天花板计算**
```python
# 该被试的fMRI RDM
subject_rdm = all_subject_rdms[subject][roi_key]

# 其他被试的平均fMRI RDM
other_subjects = [s for s in subjects if s != subject]
other_rdms = [all_subject_rdms[s][roi_key] for s in other_subjects]
mean_other_rdm = np.mean(other_rdms, axis=0)

# 噪声天花板
noise_ceiling = corr_rdms(subject_rdm.reshape(1, -1), mean_other_rdm.reshape(1, -1))[0, 0]
```

## 🎉 预期效果

使用简化版分析器，您应该能看到：

1. **更稳定的结果**：无采样随机性
2. **更高的相关性**：使用全部数据
3. **更快的计算**：无复杂采样逻辑
4. **更清晰的结果**：直接使用1000张图片

这个版本更适合快速验证和初步分析！🚀

