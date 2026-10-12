# ============================================================
# CLIP 图文相似度实验
# 参考：OpenAI CLIP 官方仓库
# 功能：
#   1. 加载 RN50 模型
#   2. 读 4 张图 + 4 句正确描述
#   3. 提取图像/文本特征，打印形状，L2 归一化
#   4. 算 4x4 余弦相似度矩阵，画热力图
#   5. 看每张图有没有匹配到正确描述
#   6. 加几句“很像但错了”的描述，看 CLIP 能不能分出来
# ============================================================

import os
import clip
import torch
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------- 1. 选设备、加载模型 ----------------
# 有 GPU 就用 GPU，没有就用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用的设备是：", device)

# RN50 比较小，先拿它试,如果跑得动可以换 ViT-B/32
model, preprocess = clip.load("RN50", device=device)
# 加载CLIP的RN50（ResNet50）图像编码器预训练模型  preprocess 是配套图片预处理工具
model.eval()  # 只做推理，不训练
print("模型加载完成！")

# ---------------- 2. 图片路径和标签 ----------------
# 注意：图片要提前放在和代码同一个文件夹下
img_paths = ["./img0.jpg", "./img1.jpg", "./img2.jpg", "./img3.jpg"]

# 这几个标签只是画图的时候当 y 轴用，方便看
img_labels = ["img0 mouse", "img1 cat", "img2 maple leaf", "img3 sunset cloud"]

# ---------------- 3. 四条正确的英文描述 ----------------
# 图片与文字一一对应
texts = [
    "A light gray Logitech wireless mouse",
    "A calico cat sitting on the asphalt road",
    "A red maple leaf on dark wet asphalt",
    "Golden sunset clouds over green forest hills"
]

# ---------------- 4. 读图片 + 预处理 ----------------
# CLIP 的 preprocess 会自动把图片 resize 到 224x224 并归一化
images = []
for path in img_paths:
    if not os.path.exists(path):
        print("找不到图片：", path)
        continue
    img = Image.open(path).convert("RGB")   # 转成 RGB，防止有些图是 RGBA
    img = preprocess(img).unsqueeze(0).to(device)  # 加一维 batch
    images.append(img)

# 把 4 张图拼成一个 batch，形状是 (4, 3, 224, 224)
image_batch = torch.cat(images, dim=0)
print("图片 batch 形状：", image_batch.shape)

# ---------------- 5. 把文字变成 token ----------------
text_tokens = clip.tokenize(texts).to(device)
print("文字 token 形状：", text_tokens.shape)

# ---------------- 6. 提取特征 + L2 归一化 ----------------
#算相似度之前要 L2 归一化
# 这样余弦相似度就直接等于点积，代码简单很多
with torch.no_grad():   # 推理不需要梯度，省内存
    image_features = model.encode_image(image_batch)
    text_features = model.encode_text(text_tokens)

# L2 归一化：每个向量除以自己的长度
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 打印形状，作业要求
print("\n===== 特征形状 =====")
print("图像特征 shape:", image_features.shape)   # 应该是 (4, 1024)
print("文本特征 shape:", text_features.shape)   # 应该是 (4, 1024)

# ---------------- 7. 算 4x4 相似度矩阵 ----------------
# 归一化之后，直接矩阵乘法就是余弦相似度
similarity_matrix = image_features @ text_features.T
similarity_matrix = similarity_matrix.cpu().numpy()   # 转 numpy 方便画图

print("\n===== 4x4 相似度矩阵 =====")
print(np.round(similarity_matrix, 4))   # 保留 4 位小数，看着舒服

# ---------------- 8. 画热力图 ----------------
# 颜色越深说明越相似，对角线应该最亮（如果模型分对了的话）
plt.figure(figsize=(8, 5))
sns.heatmap(
    similarity_matrix,
    annot=True,          # 格子里显示数值
    fmt=".3f",           # 保留 3 位小数
    cmap="Blues",        # 蓝色系，越深越像
    xticklabels=texts,
    yticklabels=img_labels
)
plt.title("CLIP Image-Text Cosine Similarity Heatmap")
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("heatmap_correct.png", dpi=150)  
plt.show()

# ---------------- 9. 检查每张图有没有匹配对 ----------------
print("\n===== 配对结果检查 =====")
correct_num = 0
for i in range(len(img_labels)):
    best_idx = similarity_matrix[i].argmax()   # 这一行最大的那个位置
    best_text = texts[best_idx]
    best_sim = similarity_matrix[i, best_idx]

    if best_idx == i:
        print(f"{img_labels[i]} -> 最匹配：'{best_text}' (相似度 {best_sim:.4f}) 正确")
        correct_num += 1
    else:
        print(f"{img_labels[i]} -> 最匹配：'{best_text}' (相似度 {best_sim:.4f}) 错了")
        print(f"    正确应该是：'{texts[i]}' (相似度 {similarity_matrix[i, i]:.4f})")

print(f"一共对了 {correct_num} / {len(img_labels)} 张")

# ---------------- 10. 故意加几句“很像但错了”的描述 ----------------
# 以 img0（鼠标）为例，正确是 light gray（浅灰）
# 我故意改成黑色、白色、有线、两个，看 CLIP 能不能分出来
wrong_texts = [
    "A black Logitech wireless mouse",        # 颜色错
    "A white Logitech wireless mouse",        # 颜色错
    "A light gray Logitech wired mouse",      # 有线/无线错
    "Two light gray Logitech wireless mice"   # 数量错
]

wrong_tokens = clip.tokenize(wrong_texts).to(device)
with torch.no_grad():
    wrong_features = model.encode_text(wrong_tokens)
wrong_features = wrong_features / wrong_features.norm(dim=-1, keepdim=True)

# 拿 img0 这张鼠标图，分别和正确描述、错误描述算相似度
mouse_feat = image_features[0]   # 第 0 张图的特征

sim_correct = (mouse_feat @ text_features[0]).item()   # 和正确描述
sim_wrong = (mouse_feat @ wrong_features.T).cpu().numpy()  # 和 4 句错误描述

print("\n===== 错误描述对比实验(img0 鼠标）=====")
print(f"正确描述：'{texts[0]}'")
print(f"相似度：{sim_correct:.4f}")
print()

for wtext, wsim in zip(wrong_texts, sim_wrong):
    diff = sim_correct - wsim   # 正数说明正确描述更高，CLIP 分对了
    if diff > 0:
        print(f"错误描述：'{wtext}'")
        print(f"相似度：{wsim:.4f}，比正确描述低 {diff:.4f}  CLIP 分出来了")
    else:
        print(f"错误描述：'{wtext}'")
        print(f"相似度：{wsim:.4f}，居然比正确描述还高 {-diff:.4f}  CLIP 被骗了")
    print()

# ---------------- 11. 把对比也画成小热力图 ----------------
# 只有一行，5 列：1 个正确 + 4 个错误
compare = np.concatenate([[sim_correct], sim_wrong]).reshape(1, -1)
compare_labels = ["正确: light gray"] + [
    "错: black", "错: white", "错: wired", "错: two"
]

plt.figure(figsize=(9, 2.5))
sns.heatmap(
    compare,
    annot=True,
    fmt=".3f",
    cmap="Reds",   # 用红色系，表示这是“找茬”实验
    xticklabels=compare_labels,
    yticklabels=["img0 mouse"]
)
plt.title("img0 mouse:正确描述 vs 错误描述")
plt.xticks(rotation=20, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("heatmap_wrong.png", dpi=150)
plt.show()

