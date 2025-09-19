import pickle
import numpy as np

# 快速检查RSA值偏小的原因
print("=== 快速RSA诊断 ===")

# 1. 检查猴子RDM
print("1. 检查猴子RDM...")
with open('all_rdms.pkl', 'rb') as f:
    monkey_rdms = pickle.load(f)

sample_rdm = list(monkey_rdms.values())[0]
sample_values = sample_rdm[np.triu(np.ones_like(sample_rdm, dtype=bool), k=1)]
print(f"   猴子RDM值范围: {sample_values.min():.3f} 到 {sample_values.max():.3f}")
print(f"   平均值: {sample_values.mean():.3f}")

# 2. 检查LLM RDM
print("2. 检查LLM RDM...")
embedding_file = r"E:\lunzhuan1\visuo_llm-main\newtry\captions\embeddings_output\image_embeddings.npy"
embeddings = np.load(embedding_file)

if embeddings.shape[0] == 1000:
    pass
elif embeddings.shape[1] == 1000:
    embeddings = embeddings.T
else:
    if embeddings.shape[0] > 1000:
        embeddings = embeddings[:1000, :]
    elif embeddings.shape[1] > 1000:
        embeddings = embeddings[:, :1000]

llm_rdm = 1 - np.corrcoef(embeddings.T)
llm_values = llm_rdm[np.triu(np.ones_like(llm_rdm, dtype=bool), k=1)]
print(f"   LLM RDM值范围: {llm_values.min():.3f} 到 {llm_values.max():.3f}")
print(f"   平均值: {llm_values.mean():.3f}")

# 3. 计算一个RSA示例
print("3. 计算RSA示例...")
from scipy.stats import spearmanr

if sample_rdm.shape != llm_rdm.shape:
    min_size = min(sample_rdm.shape[0], llm_rdm.shape[0])
    sample_rdm = sample_rdm[:min_size, :min_size]
    llm_rdm = llm_rdm[:min_size, :min_size]

mask = np.triu(np.ones_like(sample_rdm, dtype=bool), k=1)
monkey_values = sample_rdm[mask]
llm_values = llm_rdm[mask]

rsa, p_value = spearmanr(monkey_values, llm_values)
print(f"   示例RSA: {rsa:.4f}")
print(f"   p值: {p_value:.4f}")

# 4. 分析问题
print("4. 问题分析:")
if rsa < 0.1:
    print("   ❌ RSA值确实很小！")
    if sample_values.max() < 0.1:
        print("   - 猴子RDM值范围很小，可能response过于相似")
    if llm_values.max() < 0.1:
        print("   - LLM RDM值范围很小，可能embedding过于相似")
    print("   - 建议检查数据预处理和RDM计算方法")
else:
    print("   ✅ RSA值正常")
