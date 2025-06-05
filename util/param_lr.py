import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
lr_1e_5 = [0.84, 0.66, 0.61, 0.64]
lr_3e_5 = [0.90, 0.79, 0.77, 0.78]
lr_3e_4 = [0.74, 0.37, 0.36, 0.36]

# x轴位置
x = np.arange(len(labels))
width = 0.25

# 绘图
fig, ax = plt.subplots(figsize=(9, 6))
bar1 = ax.bar(x - width, lr_1e_5, width, label='lr=1e-5')
bar2 = ax.bar(x, lr_3e_5, width, label='lr=3e-5')
bar3 = ax.bar(x + width, lr_3e_4, width, label='lr=3e-4')

# 设置标题和标签
ax.set_ylabel('Score')
ax.set_xlabel('Evaluation Metric')
ax.set_title('Performance Comparison at Different Learning Rates')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 显示柱顶数值
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

# 将图例放在图外右上角
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

plt.tight_layout()
plt.show()
