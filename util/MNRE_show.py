import ast
import matplotlib.pyplot as plt
from collections import Counter

# 定义关系类型字典
rel2id = {
    "None": 0, "/per/per/parent": 1, "/per/per/siblings": 2, "/per/per/couple": 3,
    "/per/per/neighbor": 4, "/per/per/peer": 5, "/per/per/charges": 6, "/per/per/alumi": 7,
    "/per/per/alternate_names": 8, "/per/org/member_of": 9, "/per/loc/place_of_residence": 10,
    "/per/loc/place_of_birth": 11, "/org/org/alternate_names": 12, "/org/org/subsidiary": 13,
    "/org/loc/locate_at": 14, "/loc/loc/contain": 15, "/per/misc/present_in": 16,
    "/per/misc/awarded": 17, "/per/misc/race": 18, "/per/misc/religion": 19,
    "/per/misc/nationality": 20, "/misc/misc/part_of": 21, "/misc/loc/held_on": 22
}


# 读取数据集并统计关系分布
def get_relation_distribution(file_path):
    relation_counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = ast.literal_eval(line.strip())
            relation = data['relation']
            relation_counts[relation] += 1

    # 确保所有 23 种关系都有计数（即使为 0）
    for rel in rel2id.keys():
        if rel not in relation_counts:
            relation_counts[rel] = 0

    return relation_counts


# 可视化函数
def plot_relation_distribution(relation_counts):
    # 按数量从高到低排序
    sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
    relations = [rel for rel, count in sorted_relations]
    counts = [count for rel, count in sorted_relations]

    # 创建柱状图
    plt.figure(figsize=(16, 10))  # 调整图像大小
    bars = plt.bar(relations, counts, color='skyblue')

    # 添加数据标签
    for bar in bars:
        yval = bar.get_height()
        # 动态调整标签位置，避免超出范围
        label_offset = 10  # 设置标签距离顶部的最小间距
        if yval + label_offset > plt.gca().get_ylim()[1]:  # 如果标签超出图像上边界
            label_offset = -10  # 如果超出，放置在柱子下方
        plt.text(bar.get_x() + bar.get_width() / 2, yval + label_offset, int(yval), ha='center', va='bottom' if label_offset > 0 else 'top', fontsize=8)

    # 设置标题和标签
    plt.xlabel('Relation Type', fontsize=14)
    plt.ylabel('Count(Item)', fontsize=14)

    # 旋转 x 轴标签并增加间距
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # 调整布局以避免标签被截断
    plt.tight_layout(pad=5.0)  # 增加间距

    # 保存图像
    plt.savefig('MNRE_Train.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'MNRE_Train.png'")

    # 显示图像
    plt.show()


# 主程序
if __name__ == "__main__":
    file_path = "G:\\dataset\\MNRE_data\\txt\\ours_train.txt"
    # file_path = "G:\\dataset\\MNRE_data\\txt\\ours_test.txt"
    relation_counts = get_relation_distribution(file_path)
    print("Relation counts:", dict(relation_counts))
    plot_relation_distribution(relation_counts)
