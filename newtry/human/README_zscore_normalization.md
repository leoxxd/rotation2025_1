# RSA分析中的z-score归一化

## 概述

本目录包含了在fMRI计算时对每个人的1000张图片beta进行z-score归一化的RSA分析方法。这种方法可以消除个体差异，提高不同被试之间的可比性。

## 文件说明

### 主要脚本

1. **`rsa_analyzer_simple.py`** - 原始RSA分析器（无归一化）
2. **`rsa_analyzer_with_zscore.py`** - z-score归一化版RSA分析器
3. **`compare_rsa_methods.py`** - 两种方法的对比分析器

### 数据文件

- `roi_rdm_results/all_subjects_roi_rdms.mat` - ROI RDM数据
- `../captions/embeddings_output/image_embeddings.npy` - caption embeddings

## z-score归一化方法

### 原理

在计算RDM之前，对每个体素在1000张图片上的响应进行z-score归一化：

```python
# 对每个体素进行z-score归一化
for voxel_idx in range(roi_data.shape[0]):
    voxel_responses = roi_data[voxel_idx, :]  # 该体素对1000张图片的响应
    mean_response = np.mean(voxel_responses)
    std_response = np.std(voxel_responses)
    
    if std_response > 0:
        normalized_data[voxel_idx, :] = (voxel_responses - mean_response) / std_response
    else:
        normalized_data[voxel_idx, :] = voxel_responses - mean_response
```

### 优势

1. **消除个体差异**：每个人的fMRI信号强度范围不同，z-score可以标准化到相同的分布
2. **保持相对模式**：z-score只改变数值范围，不改变相对关系，所以RDM计算仍然有效
3. **提高可比性**：不同被试之间的RSA结果更加可比

## 使用方法

### 1. 运行z-score归一化版RSA分析

```bash
cd newtry/human
python rsa_analyzer_with_zscore.py
```

### 2. 运行原始版RSA分析（用于对比）

```bash
cd newtry/human
python rsa_analyzer_simple.py
```

### 3. 对比两种方法

```bash
cd newtry/human
python compare_rsa_methods.py
```

## 输出文件

### z-score归一化版本

- `rsa_results_zscore/all_rsa_results_zscore.mat` - 所有结果
- `rsa_results_zscore/all_rsa_results_zscore.csv` - CSV格式结果
- `rsa_results_zscore/roi_lh_1/` - 按ROI分类的结果（包含分析图）
- `rsa_results_zscore/subject_s1/` - 按被试分类的结果（包含分析图）
- `rsa_results_zscore/rsa_analysis_zscore.png` - 总体分析图
- `rsa_results_zscore/rsa_heatmap_zscore.png` - 热力图
- `rsa_results_zscore/plots/` - 详细可视化图表目录
  - `correlation_comparison_zscore.png` - 相关性比较散点图
  - `roi_comparison_zscore.png` - ROI比较箱线图
  - `subject_comparison_zscore.png` - 被试比较箱线图
  - `s1_roi_analysis_zscore.png` - 被试s1的ROI分析图
  - `s2_roi_analysis_zscore.png` - 被试s2的ROI分析图
  - 等等...
- `rsa_results_zscore/zscore_analysis.txt` - 详细分析报告

### 对比分析

- `rsa_comparison/comparison_report.txt` - 对比报告
- `rsa_comparison/comparison_data.csv` - 对比数据
- `rsa_comparison/rsa_methods_comparison.png` - 方法对比图
- `rsa_comparison/rsa_correlation_scatter.png` - 相关性散点图
- `rsa_comparison/rsa_heatmap_comparison.png` - 热力图对比

## 结果解读

### 主要指标

1. **原始相关性**：fMRI RDM与embedding RDM的直接相关性
2. **校正后相关性**：原始相关性除以噪声天花板
3. **噪声天花板**：该被试与其他被试平均fMRI RDM的相关性

### 归一化效果

- **个体差异消除**：z-score归一化后，不同被试的fMRI信号分布更加一致
- **相对模式保持**：RDM的相对结构保持不变，只是数值范围标准化
- **相关性变化**：通常z-score归一化会略微改变相关性，但不会大幅改变相对排序

## 可视化功能

### 图表类型

1. **ROI分析图** (`roi_lh_1/roi_lh_1_rsa_analysis_zscore.png`)
   - 2x2子图布局
   - 原始相关性、校正后相关性、噪声天花板、校正效果
   - 按被试显示，包含数值标签

2. **被试分析图** (`subject_s1/s1_rsa_analysis_zscore.png`)
   - 2x2子图布局
   - 按ROI分组显示左右脑数据
   - 原始相关性、校正后相关性、噪声天花板、校正效果

3. **总体分析图** (`rsa_analysis_zscore.png`)
   - 原始相关性 vs 校正后相关性散点图
   - 包含y=x参考线

4. **热力图** (`rsa_heatmap_zscore.png`)
   - 被试 x ROI 原始相关性热力图
   - 被试 x ROI 校正后相关性热力图

5. **详细可视化图表** (`plots/` 目录)
   - 相关性比较散点图
   - ROI比较箱线图
   - 被试比较箱线图
   - 每个被试的ROI分析图

### 图表特点

- **中文字体支持**：使用SimHei、Arial Unicode MS等字体
- **高分辨率**：300 DPI，适合论文发表
- **数值标签**：柱状图上显示具体数值
- **网格线**：便于读取数值
- **颜色区分**：不同指标使用不同颜色
- **标题标注**：明确标注z-score归一化版本

## 技术细节

### 数据流程

1. 加载ROI掩码和fMRI数据
2. 提取ROI内的fMRI数据 `[n_roi_voxels, n_images]`
3. 对每个体素在1000张图片上进行z-score归一化
4. 计算归一化后的RDM
5. 与embedding RDM进行相关性分析
6. 生成完整的可视化图表

### 归一化公式

对于体素i在图像j上的响应值：
```
normalized_response[i,j] = (response[i,j] - mean(response[i,:])) / std(response[i,:])
```

其中：
- `mean(response[i,:])` 是体素i在所有1000张图片上的平均响应
- `std(response[i,:])` 是体素i在所有1000张图片上的标准差

## 注意事项

1. **数据完整性**：确保所有被试都有完整的1000张图片数据
2. **ROI选择**：确保ROI内有足够的体素进行统计分析
3. **噪声天花板**：z-score归一化可能影响噪声天花板的计算
4. **结果解释**：归一化后的结果需要谨慎解释，特别是与原始方法的对比

## 参考文献

- Kriegeskorte, N., et al. (2008). Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.
- Nili, H., et al. (2014). A toolbox for representational similarity analysis. *PLoS Computational Biology*, 10(7), e1003553.
