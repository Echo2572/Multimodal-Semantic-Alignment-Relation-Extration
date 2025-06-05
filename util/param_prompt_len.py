import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
prompt_6 = [0.80, 0.70, 0.72, 0.71]
prompt_12 = [0.90, 0.79, 0.77, 0.78]
prompt_24 = [0.88, 0.75, 0.72, 0.73]

# x轴位置
x = np.arange(len(labels))
width = 0.25

# 绘图
fig, ax = plt.subplots(figsize=(9, 6))
bar1 = ax.bar(x - width, prompt_6, width, label='Prompt Length = 6')
bar2 = ax.bar(x, prompt_12, width, label='Prompt Length = 12')
bar3 = ax.bar(x + width, prompt_24, width, label='Prompt Length = 24')

# 设置标签和标题
ax.set_ylabel('Score')
ax.set_xlabel('Evaluation Metric')
ax.set_title('Effect of Visual Prompt Length on Evaluation Metrics')
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
