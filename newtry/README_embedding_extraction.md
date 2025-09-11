# Caption Embedding 提取工具

这个工具专门用于处理 `Anno_Shared1000.txt` 文件，提取其中所有caption的embedding向量。

## 文件说明

1. **`simple_extract_embeddings.py`** - 简化版脚本，推荐使用
2. **`extract_embeddings_shared1000.py`** - 完整版脚本，支持更多功能

## 安装依赖

```bash
pip install sentence-transformers numpy tqdm
```

## 使用方法

### 方法1：使用简化版脚本（推荐）

```bash
python simple_extract_embeddings.py
```

### 方法2：使用完整版脚本

```bash
# 基本使用
python extract_embeddings_shared1000.py

# 自定义参数
python extract_embeddings_shared1000.py --input_file "your_file_path.txt" --output_dir "./my_output" --visualize
```

## 输出文件

脚本会在 `./embeddings_output/` 目录下生成以下文件：

- `embeddings.npy` - 所有caption的embedding向量（numpy数组）
- `metadata.pkl` - 元数据（包含caption文本、图像ID等）
- `metadata.json` - 元数据的JSON格式（便于查看）
- `embeddings_pca.png` - 可视化图（如果使用--visualize参数）

## 数据格式

### 输入格式
文件每行包含一个图像组的数据，格式如下：
```
[{'image_id': 262145, 'id': 694, 'caption': 'People shopping in an open market for vegetables.'}, ...]
```

### 输出格式
- **embeddings.npy**: 形状为 (N, 768) 的numpy数组，其中N是caption总数，768是mpnet模型的embedding维度
- **metadata.pkl**: 包含以下字段的字典：
  - `captions`: 所有caption文本的列表
  - `image_ids`: 对应的图像ID列表
  - `caption_ids`: 对应的caption ID列表
  - `embedding_shape`: embedding数组的形状
  - `model_name`: 使用的模型名称

## 使用示例

```python
import numpy as np
import pickle

# 加载embeddings
embeddings = np.load('./embeddings_output/embeddings.npy')

# 加载元数据
with open('./embeddings_output/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# 查看第一个caption的embedding
print(f"第一个caption: {metadata['captions'][0]}")
print(f"对应的embedding形状: {embeddings[0].shape}")
print(f"embedding向量: {embeddings[0][:10]}...")  # 显示前10个维度
```

## 注意事项

1. 确保输入文件路径正确
2. 首次运行时会自动下载mpnet模型（约400MB）
3. 处理1000个图像组可能需要几分钟时间
4. 生成的embedding向量是768维的浮点数数组

## 故障排除

如果遇到问题，请检查：
1. 文件路径是否正确
2. 是否安装了所有依赖包
3. 是否有足够的磁盘空间
4. 网络连接是否正常（用于下载模型）
