# Caption Embedding 提取工具

这个工具专门用于处理 `Anno_Shared1000.txt` 文件，**按图像分组**提取embedding向量。

## 处理方式

- **输入**：1000张图像，每张图像有5个caption
- **处理**：每张图像的5个caption分别生成embedding，然后计算平均embedding
- **输出**：1000个图像embedding向量（每个向量是5个caption embedding的平均值）

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

- `image_embeddings.npy` - 1000个图像的平均embedding向量（形状：1000 × 768）
- `metadata.pkl` - 元数据（包含每张图像的所有caption、图像ID等）
- `metadata.json` - 元数据的JSON格式（便于查看）
- `embeddings_pca.png` - 可视化图（如果使用--visualize参数）

## 数据格式

### 输入格式
文件每行包含一个图像组的数据，格式如下：
```
[{'image_id': 262145, 'id': 694, 'caption': 'People shopping in an open market for vegetables.'}, ...]
```

### 输出格式
- **image_embeddings.npy**: 形状为 (1000, 768) 的numpy数组，其中1000是图像数量，768是mpnet模型的embedding维度
- **metadata.pkl**: 包含以下字段的字典：
  - `image_ids`: 图像ID列表（1000个）
  - `captions_per_image`: 每张图像的所有caption（1000个列表，每个列表包含5个caption）
  - `embedding_shape`: embedding数组的形状 (1000, 768)
  - `model_name`: 使用的模型名称
  - `processing_method`: 处理方法说明

## 使用示例

```python
import numpy as np
import pickle

# 加载图像embeddings
image_embeddings = np.load('./embeddings_output/image_embeddings.npy')

# 加载元数据
with open('./embeddings_output/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# 查看第一张图像的信息
print(f"第一张图像ID: {metadata['image_ids'][0]}")
print(f"第一张图像的5个caption:")
for i, caption in enumerate(metadata['captions_per_image'][0]):
    print(f"  {i+1}. {caption}")
print(f"对应的平均embedding形状: {image_embeddings[0].shape}")
print(f"平均embedding向量前10个值: {image_embeddings[0][:10]}")

# 查看整体统计
print(f"总图像数量: {len(metadata['image_ids'])}")
print(f"Embedding数组形状: {image_embeddings.shape}")
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
