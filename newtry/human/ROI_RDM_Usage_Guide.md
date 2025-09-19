# ROI RDM处理器使用指南

## 🎯 功能概述

这个工具专门为您的数据结构设计，能够：

1. **读取ROI定义**：从`lh.streams.mgz`和`rh.streams.mgz`文件读取ROI标签
2. **加载fMRI数据**：从`S1_lh_Rsp.mat`等文件加载fMRI数据
3. **提取ROI数据**：根据ROI标签提取对应的fMRI数据
4. **计算RDM**：为每个ROI计算表示相似性矩阵
5. **保存结果**：保存所有ROI的RDM数据

## 📁 您的数据结构

```
E:\lunzhuan1\rotation2025\Human\
├── s1\
│   └── fs\label\
│       ├── lh.streams.mgz  # 左半球ROI标签
│       └── rh.streams.mgz  # 右半球ROI标签
├── s2\fs\label\
├── s5\fs\label\
├── s7\fs\label\
├── S1_lh_Rsp.mat          # 被试1左半球fMRI数据
├── S1_rh_Rsp.mat          # 被试1右半球fMRI数据
├── S2_lh_Rsp.mat
├── S2_rh_Rsp.mat
├── S5_lh_Rsp.mat
├── S5_rh_Rsp.mat
├── S7_lh_Rsp.mat
└── S7_rh_Rsp.mat
```

## 🔧 使用方法

### 步骤1: 安装依赖
```bash
pip install nibabel numpy scipy matplotlib
```

### 步骤2: 运行处理器
```bash
cd newtry
python roi_rdm_processor.py
```

### 步骤3: 查看结果
程序会在`roi_rdm_results/`目录下生成：
- `all_subjects_roi_rdms.mat` - 所有RDM数据
- `roi_rdm_summary.mat` - 汇总信息
- `roi_rdm_summary.txt` - 文本摘要
- `roi_rdm_analysis.png` - 可视化图表

## 📊 ROI标签对应关系

根据您的`streams.mgz.ctab`文件：

| 标签 | 名称 | 中文描述 | 主要脑区 |
|------|------|----------|----------|
| 0 | Unknown | 未知区域 | 背景 |
| 1 | early | 早期视觉 | V1, V2, V3, V4 |
| 2 | midventral | 中腹侧 | LOC, FFA, PPA, EBA |
| 3 | midlateral | 中外侧 | MT, MST, V5 |
| 4 | midparietal | 中顶叶 | SPL, IPL, LIP |
| 5 | ventral | 腹侧 | IT, ATL, FFA, PPA |
| 6 | lateral | 外侧 | MT+, MST, VIP |
| 7 | parietal | 顶叶 | SPL, IPL, SMG, AG |

**注意**：程序只处理标签1-7，跳过标签0（背景）

## 🔍 处理流程

### 1. 数据加载
```python
# 加载ROI掩码
roi_masks = load_roi_masks('s1')
# 结果: {'lh_1': mask_array, 'lh_2': mask_array, ...}

# 加载fMRI数据
fmri_data = load_fmri_data('s1')
# 结果: {'lh': [n_voxels, n_images], 'rh': [n_voxels, n_images]}
```

### 2. ROI数据提取
```python
# 提取特定ROI的数据
roi_data = extract_roi_data(fmri_data['lh'], roi_masks['lh_1'])
# 结果: [n_roi_voxels, n_images]
```

### 3. RDM计算
```python
# 计算RDM
rdm = compute_rdm(roi_data)
# 结果: 上三角距离向量
```

## 📈 输出结果

### MATLAB文件结构
```matlab
all_subjects_roi_rdms.mat:
├── s1/
│   ├── lh_1/  # 左半球ROI 1
│   │   ├── roi_data: [n_voxels, n_images]
│   │   └── rdm: [rdm_vector]
│   ├── lh_2/
│   ├── rh_1/
│   └── ...
├── s2/
├── s5/
└── s7/
```

### 汇总信息
```matlab
roi_rdm_summary.mat:
├── s1/
│   ├── lh_1/
│   │   ├── roi_label: 1
│   │   ├── roi_name: "early (早期视觉)"
│   │   ├── hemisphere: "lh"
│   │   ├── n_voxels: 1500
│   │   ├── n_images: 1000
│   │   └── rdm_length: 499500
│   └── ...
```

## 🎯 关键特性

### 1. 自动数据匹配
- 自动检查ROI掩码和fMRI数据的维度匹配
- 自动处理左右半球数据
- 自动跳过缺失的数据

### 2. 错误处理
- 检查文件是否存在
- 验证数据格式
- 处理异常情况

### 3. 质量控制
- 统计每个ROI的顶点数
- 检查数据完整性
- 生成详细报告

## 🔧 自定义选项

### 修改目标ROI
```python
# 在ROIRDMProcessor类中修改
self.target_rois = [1, 2, 5]  # 只处理特定ROI
```

### 修改距离度量
```python
# 在compute_rdm方法中修改
rdm = pdist(data_for_rdm, metric='euclidean')  # 使用欧几里得距离
```

### 修改数据目录
```python
# 在main函数中修改
processor = ROIRDMProcessor(data_dir="您的数据路径")
```

## 🚨 注意事项

### 1. 文件命名
- ROI文件：`{subject}/fs/label/{hemisphere}.streams.mgz`
- fMRI文件：`{SUBJECT}_{hemisphere}_Rsp.mat`

### 2. 数据格式
- ROI文件：FreeSurfer .mgz格式
- fMRI文件：MATLAB .mat格式

### 3. 内存使用
- 每个ROI的RDM会占用内存
- 建议确保有足够的内存

## 🎉 预期结果

运行成功后，您将获得：

1. **完整的ROI RDM数据**：每个被试每个ROI的RDM
2. **详细的统计信息**：ROI大小、数据完整性等
3. **可视化图表**：ROI分布、被试间比较等
4. **MATLAB兼容格式**：可直接在MATLAB中加载使用

## 🔗 后续分析

获得RDM后，您可以：

1. **比较不同ROI**：计算ROI间RDM的相关性
2. **比较不同被试**：分析个体差异
3. **比较不同模型**：与语言模型的RDM比较
4. **功能特异性分析**：分析ROI的功能特异性

## 📞 技术支持

如果遇到问题，请检查：

1. **文件路径**：确保数据目录正确
2. **文件格式**：确保文件格式正确
3. **依赖库**：确保所有依赖库已安装
4. **内存空间**：确保有足够的内存

程序会提供详细的错误信息和处理状态，帮助您诊断问题。

