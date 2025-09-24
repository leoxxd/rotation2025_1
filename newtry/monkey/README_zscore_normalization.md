# 猴子神经元信号RSA分析 - Z-score归一化版本

## 概述

本项目为猴子神经元信号实现了z-score归一化的RSA（Representational Similarity Analysis）分析，完全基于原始的`rsa_with_noise_ceiling.py`代码结构，只是在计算RDM之前对神经元响应进行z-score归一化。

## 文件结构

```
newtry/monkey/
├── compute_rdm_all_sessions_zscore.py          # 计算z-score归一化RDM
├── rsa_with_zscore_normalization.py            # z-score归一化RSA分析
├── compare_monkey_rsa_methods.py               # 比较原始和z-score版本
├── README_zscore_normalization.md              # 本文档
├── all_rdms_correlation_zscore.pkl             # z-score归一化RDM数据
├── session_info_correlation_zscore.csv         # session信息
├── rsa_with_zscore_results.pkl                 # z-score分析结果
├── rsa_comparison_monkey.csv                   # 比较结果
└── comparison_plots/                           # 比较图表目录
    ├── rsa_comparison_monkey.png
    └── correlation_analysis_monkey.png
```

## 技术实现

### 1. Z-score归一化

**归一化公式**：
```
z_ij = (x_ij - μ_i) / σ_i
```

其中：
- `x_ij`：第i个神经元对第j张图片的响应
- `μ_i`：第i个神经元在1000张图片上的平均响应
- `σ_i`：第i个神经元在1000张图片上的标准差
- `z_ij`：归一化后的响应

**归一化范围**：每个神经元在1000张图片上独立进行z-score归一化

### 2. 处理流程

1. **数据加载**：加载原始猴子神经元响应数据
2. **Z-score归一化**：对每个神经元的响应进行标准化
3. **RDM计算**：基于归一化后的响应计算表示性差异矩阵
4. **RSA分析**：与LLM embedding进行相关性分析
5. **噪声天花板计算**：使用原始项目的方法
6. **结果可视化**：生成详细的分析图表

## 使用方法

### 1. 计算Z-score归一化RDM

```bash
python compute_rdm_all_sessions_zscore.py
```

**输出文件**：
- `all_rdms_correlation_zscore.pkl`：所有session的z-score归一化RDM
- `session_info_correlation_zscore.csv`：session信息表

### 2. 运行Z-score归一化RSA分析

```bash
python rsa_with_zscore_normalization.py
```

**输出文件**：
- `rsa_with_zscore_results.pkl`：完整分析结果
- `noise_ceiling_plots_zscore/`：详细分析图表目录
  - `rsa_*_zscore.png`：各区域详细分析图
  - `rsa_table_*_zscore.csv`：各区域数据表
  - `noise_ceiling_analysis_zscore.png`：汇总分析图
  - `noise_ceiling_results_zscore.csv`：汇总数据表

### 3. 比较原始版本和Z-score版本

```bash
python compare_monkey_rsa_methods.py
```

**输出文件**：
- `rsa_comparison_monkey.csv`：比较结果数据
- `comparison_plots/`：比较图表目录
  - `rsa_comparison_monkey.png`：方法比较图
  - `correlation_analysis_monkey.png`：相关性分析图

## 分析结果

### 主要发现

1. **Z-score归一化效果**：
   - 平均RSA差异：0.0335 ± 0.0243
   - 平均矫正RSA差异：-0.0099 ± 0.1157
   - 平均噪声天花板差异：0.0763 ± 0.0411

2. **相关性分析**：
   - 原始RSA vs Z-score RSA：r = 0.556, p = 0.039
   - 矫正RSA vs Z-score矫正RSA：r = 0.890, p = 0.000
   - 噪声天花板 vs Z-score噪声天花板：r = 0.965, p = 0.000

3. **区域排名**（按矫正后RSA排序）：
   - MO1s1：0.870（最高）
   - AMC3：0.742
   - MO1s2：0.628
   - MO2：0.498
   - PITP4：0.476
   - LPP4：0.460
   - AB3：0.442
   - MB1：0.423
   - MB3：0.367
   - MB2：0.361
   - MF3：0.338
   - AB1：0.313
   - CLC3：0.299
   - MF1：0.231（最低）

### 技术特点

1. **完全基于原始代码**：保持了原始的分析流程和画图风格
2. **只添加z-score归一化**：在RDM计算前对神经元响应进行标准化
3. **保持相对模式**：归一化后RDM的相对结构保持不变
4. **完整可视化**：包含区域分析图和汇总图表

## 与原始版本的对比

| 特征 | 原始版本 | Z-score版本 |
|------|----------|-------------|
| 数据预处理 | 无 | Z-score归一化 |
| RDM计算 | 基于原始响应 | 基于归一化响应 |
| 噪声天花板 | 原始方法 | 原始方法 |
| RSA分析 | 原始方法 | 原始方法 |
| 可视化 | 原始风格 | 原始风格+归一化标识 |

## 技术细节

### 1. 归一化实现

```python
def zscore_normalize_responses(responses):
    """
    对神经元响应进行z-score归一化
    
    Args:
        responses: 神经元响应数据 [n_neurons, n_images]
        
    Returns:
        normalized_responses: 归一化后的响应数据
    """
    if responses.shape[0] == 0:
        return responses
    
    # 对每个神经元（行）在1000张图片上进行z-score归一化
    mean = np.mean(responses, axis=1, keepdims=True)
    std = np.std(responses, axis=1, keepdims=True)
    
    # 避免除以零
    std[std == 0] = 1e-9
    
    normalized_responses = (responses - mean) / std
    
    return normalized_responses
```

### 2. RDM计算

```python
def compute_rdm(responses, method='correlation'):
    """
    计算RDM
    
    Args:
        responses: response数据 (n_neurons, n_images)
        method: 距离计算方法 ('correlation', 'euclidean', 'cosine')
        
    Returns:
        rdm: RDM矩阵 (n_images, n_images)
    """
    if method == 'correlation':
        # 使用1 - 相关系数作为距离
        corr_matrix = np.corrcoef(responses.T)  # 转置以计算图片间的相关性
        rdm = 1 - corr_matrix
    # ... 其他方法
    
    return rdm
```

## 输出文件说明

### 1. RDM数据文件

- **`all_rdms_correlation_zscore.pkl`**：包含所有session的z-score归一化RDM数据
- **`session_info_correlation_zscore.csv`**：session信息表，包含每个session的统计信息

### 2. 分析结果文件

- **`rsa_with_zscore_results.pkl`**：完整的RSA分析结果
- **`noise_ceiling_plots_zscore/`**：详细分析图表目录
  - 各区域的详细分析图（4个子图）
  - 各区域的数据表
  - 汇总分析图
  - 汇总数据表

### 3. 比较结果文件

- **`rsa_comparison_monkey.csv`**：原始版本和z-score版本的详细比较
- **`comparison_plots/`**：比较图表目录
  - 方法比较图（4个子图）
  - 相关性分析图（4个子图）

## 注意事项

1. **数据一致性**：确保使用相同的原始数据文件
2. **内存使用**：RDM计算可能需要较多内存
3. **计算时间**：z-score归一化会增加一定的计算时间
4. **结果解释**：归一化后的结果需要结合原始结果进行解释

## 依赖项

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Pickle

## 作者

基于原始`rsa_with_noise_ceiling.py`代码，添加z-score归一化功能。

## 更新日志

- **v1.0**：初始版本，实现z-score归一化RSA分析
- 完全基于原始代码结构
- 添加z-score归一化功能
- 保持原始分析流程和可视化风格