# 猴子数据Response提取工具

这个文件夹包含了从猴子数据中提取response的完整工具集。

## 文件说明

### 1. `extract_monkey_responses.py`
主要的数据提取脚本，功能包括：
- 从`exclude_area.xls`读取ROI信息
- 根据位置范围(y1, y2)和可靠性阈值筛选神经元
- 提取前1000张图片的response数据
- 跳过arealabel为"Unknown"的ROI
- 保存提取的数据为.mat和.pkl格式

### 2. `extract_by_session.py`
按session提取数据的脚本，功能包括：
- 提取特定session的数据
- 验证数据格式
- 保存单个session的数据

### 3. `compute_rdm_all_sessions.py`
计算所有session的RDM脚本，功能包括：
- 计算每个session的RDM矩阵
- 分析RDM相似性
- 保存RDM数据

### 4. `run_rsa_analysis.py`
RSA分析脚本，功能包括：
- 按arealabel对session进行分组
- 与LLM embedding的RDM进行RSA分析
- 生成详细的分析结果和可视化

### 5. `extracted_monkey_responses.mat` 和 `extracted_monkey_responses.pkl`
提取的数据文件，包含：

**pickle文件**（完整数据）：
- `extracted_data`: 按session组织的response数据
- `exclude_area_info`: ROI信息
- `summary`: 数据提取摘要

**mat文件**（简化数据）：
- `combined_responses`: 合并的response数据 (12581, 1000)
- `neuron_mapping`: 神经元映射，记录每个神经元属于哪个session和ROI
- `session_info`: 每个ROI的元信息
- `summary`: 数据提取摘要
- `total_neurons`: 总神经元数
- `n_images`: 图片数量

## 数据提取流程

1. **ROI筛选**: 根据`exclude_area.xls`中的session、arealabel、y1、y2信息
2. **神经元筛选**: 
   - 位置筛选：pos在[y1, y2]范围内
   - 可靠性筛选：reliability_best > 0.4
3. **Response提取**: 提取前1000张图片的response数据
4. **数据整合**: 按session和ROI组织数据

## 提取结果摘要

- **总session数**: 48个
- **总ROI数**: 48个  
- **总神经元数**: 12,581个
- **数据形状**: (n_neurons, 1000) - 每个ROI包含不同数量的神经元，但都是1000张图片

## 使用方法

### 提取数据
```bash
cd monkey
python extract_monkey_responses.py
```

### 按session提取数据
```bash
python extract_by_session.py
```

### 计算RDM
```bash
python compute_rdm_all_sessions.py
```

### 进行RSA分析
```bash
python run_rsa_analysis.py
```

### 使用数据
```python
# 方法1：从pickle文件访问完整数据
import pickle
with open('extracted_monkey_responses.pkl', 'rb') as f:
    data = pickle.load(f)
session_data = data['extracted_data'][1]  # 获取session 1的数据
roi_data = session_data['rois'][3]  # 获取ROI 3的数据
responses = roi_data['responses']  # 获取response数据

# 方法2：从mat文件使用映射访问数据
import scipy.io as sio
mat_data = sio.loadmat('extracted_monkey_responses.mat')
combined_responses = mat_data['combined_responses']  # (12581, 1000)
neuron_mapping = mat_data['neuron_mapping']  # 神经元映射

# 提取session 1的神经元
session_mask = neuron_mapping['session'] == 1
session_neurons = combined_responses[session_mask, :]

# 提取特定ROI的数据
roi_mask = neuron_mapping['roi'] == 3
roi_neurons = combined_responses[roi_mask, :]

# 按arealabel提取数据
arealabel_mask = neuron_mapping['arealabel'] == 'MB1'
mb1_neurons = combined_responses[arealabel_mask, :]
```

## 数据结构

提取的数据按以下结构组织：

```
extracted_data/
├── session_1/
│   └── roi_3/
│       ├── arealabel: "MB1"
│       ├── y1, y2: 位置范围
│       ├── neuron_indices: 符合条件的神经元索引
│       ├── responses: (n_neurons, 1000) response数据
│       └── n_neurons: 神经元数量
├── session_2/
│   └── ...
└── ...
```

## 注意事项

1. 脚本会自动跳过arealabel为"Unknown"的ROI
2. 可靠性阈值默认设置为0.4，可在代码中修改
3. 只提取前1000张图片的response数据
4. 某些session可能没有对应的ROI信息，会被自动跳过
5. 数据同时保存为.mat和.pkl格式，推荐使用.pkl格式进行后续分析

## 依赖包

- pandas
- scipy
- numpy
- xlrd (用于读取Excel文件)
- matplotlib (用于绘图)
- pickle (用于数据保存)
