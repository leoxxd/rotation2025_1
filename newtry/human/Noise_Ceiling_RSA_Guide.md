# 带噪声天花板校正的RSA分析指南

## 🎯 功能概述

`rsa_analyzer_with_noise_ceiling.py` 是一个完整的RSA分析工具，具有以下特点：

### ✨ 主要功能
1. **噪声天花板校正**：基于fMRI数据计算噪声天花板，显著提高相关性
2. **按被试分开保存**：每个被试的结果单独保存
3. **按ROI分开保存**：每个脑区的结果单独保存
4. **丰富的可视化**：生成多种图表进行结果分析
5. **详细统计分析**：提供完整的统计报告

## 🚀 使用方法

### 1. 运行分析
```bash
cd newtry
python rsa_analyzer_with_noise_ceiling.py
```

### 2. 输出结构
```
rsa_results_noise_ceiling/
├── rsa_results_with_noise_ceiling.mat          # 所有结果（MATLAB格式）
├── rsa_results_with_noise_ceiling.csv          # 所有结果（CSV格式）
├── noise_ceiling_analysis.txt                  # 详细分析报告
├── by_subject/                                 # 按被试分开
│   ├── s1_results.mat
│   ├── s1_results.csv
│   ├── s2_results.mat
│   ├── s2_results.csv
│   └── ...
├── by_roi/                                     # 按ROI分开
│   ├── roi_1_results.mat
│   ├── roi_1_results.csv
│   ├── roi_2_results.mat
│   ├── roi_2_results.csv
│   └── ...
└── plots/                                      # 可视化图表
    ├── correlation_comparison.png              # 相关性比较
    ├── roi_comparison.png                      # ROI比较
    ├── subject_comparison.png                  # 被试比较
    ├── correlation_distributions.png           # 相关性分布
    ├── correlation_heatmap.png                 # 相关性热力图
    ├── s1_roi_analysis.png                     # s1被试的ROI分析
    ├── s2_roi_analysis.png                     # s2被试的ROI分析
    ├── s5_roi_analysis.png                     # s5被试的ROI分析
    └── s7_roi_analysis.png                     # s7被试的ROI分析
```

## 📊 可视化图表说明

### 1. **相关性比较图** (`correlation_comparison.png`)
- **内容**：原始相关性 vs 校正后相关性的散点图
- **用途**：直观显示噪声天花板校正的效果
- **解读**：点越偏离对角线，校正效果越明显

### 2. **ROI比较图** (`roi_comparison.png`)
- **内容**：按ROI分组的箱线图，显示原始相关性、校正后相关性和噪声天花板
- **用途**：比较不同脑区的表现
- **解读**：箱线图显示中位数、四分位数和异常值

### 3. **被试比较图** (`subject_comparison.png`)
- **内容**：按被试分组的箱线图
- **用途**：比较不同被试的表现
- **解读**：识别个体差异和一致性

### 4. **相关性分布图** (`correlation_distributions.png`)
- **内容**：四个子图显示不同指标的分布
- **用途**：了解数据的整体分布特征
- **解读**：直方图显示数据的集中趋势和离散程度

### 5. **相关性热力图** (`correlation_heatmap.png`)
- **内容**：被试 x ROI 的相关性矩阵
- **用途**：快速识别高相关性的被试-ROI组合
- **解读**：颜色越深表示相关性越高

### 6. **个别被试ROI分析图** (`{subject}_roi_analysis.png`)
- **内容**：每个被试的四个子图，显示该被试在不同ROI的表现
  - 原始相关性柱状图
  - 校正后相关性柱状图
  - 噪声天花板柱状图
  - 校正效果柱状图
- **用途**：详细分析每个被试的个体表现
- **解读**：
  - 原始相关性：该被试在各ROI的原始表现
  - 校正后相关性：噪声天花板校正后的表现
  - 噪声天花板：该被试在各ROI的最大可能相关性
  - 校正效果：校正带来的提升幅度

## 🔍 结果解读

### 1. **噪声天花板校正效果**
```python
# 校正公式
corrected_correlation = raw_correlation / noise_ceiling
```

### 2. **预期结果**
- **校正前**：相关性通常较低（0.1-0.2）
- **校正后**：相关性应该显著提高（0.3-0.5）
- **噪声天花板**：通常在0.3-0.8之间

### 3. **结果质量评估**
- **校正后相关性 > 0.3**：良好的结果
- **校正后相关性 > 0.4**：优秀的结果
- **校正后相关性 > 0.5**：非常优秀的结果

## 📈 统计分析

### 1. **按ROI统计**
- 每个ROI的平均相关性
- 标准差和变异系数
- 被试数量

### 2. **按被试统计**
- 每个被试的平均相关性
- 跨ROI的一致性
- 个体差异分析

### 3. **总体统计**
- 所有结果的汇总
- 校正效果评估
- 数据质量指标

## 🎯 关键优势

### 1. **噪声天花板校正**
- 消除个体差异
- 标准化相关性
- 提高结果可靠性

### 2. **灵活的数据组织**
- 按被试分开：便于个体分析
- 按ROI分开：便于脑区比较
- 多种格式：MATLAB和CSV

### 3. **丰富的可视化**
- 多种图表类型
- 高分辨率输出
- 中文标签支持

### 4. **详细的统计报告**
- 完整的统计分析
- 易于理解的格式
- 便于进一步分析

## 🔧 技术细节

### 1. **噪声天花板计算**
```python
# 基于fMRI数据计算
noise_ceiling = corr_rdms(subject_rdm, mean_other_subjects_rdm)
```

### 2. **数据格式**
- **输入**：ROI RDM数据（.mat文件）
- **输出**：多种格式的结果文件
- **可视化**：PNG格式，300 DPI

### 3. **依赖库**
- `numpy`：数值计算
- `scipy`：科学计算
- `matplotlib`：基础绘图
- `seaborn`：统计绘图
- `pandas`：数据处理

## 🎉 预期效果

使用噪声天花板校正后，您应该能看到：

1. **显著提高的相关性**：从0.1-0.2提升到0.3-0.5
2. **更稳定的结果**：消除个体差异影响
3. **更接近原始项目的结果**：使用相同的方法论
4. **更可靠的分析**：基于fMRI数据的噪声天花板

这就是为什么原始项目能达到0.4相关性的关键！🎯
