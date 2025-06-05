# Multimodal-Semantic-Alignment-Relation-Extration
This is a repository about my bachelor's degree project.

## 环境

Python 3.8 + torch 1.8 + torchvision 0.9.0 + transformer 4.11.3 + numpy 1.24.4

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xxx
```

## 模型结构

主要参考文献: 

```
Chen X, Zhang N, Li L, et al. Good visual guidance makes a better extractor: Hierarchical visual prefix for multimodal entity and relation extraction[J]. arXiv preprint arXiv:2205.03521, 2022.

Dai Y, Gao F, Zeng D. An Alignment and Matching Network with Hierarchical Visual Features for Multimodal Named Entity and Relation Extraction[C]//International Conference on Neural Information Processing. Singapore: Springer Nature Singapore, 2023: 298-310.
```

- BERT提取文本特征
- CLIP提取图像特征
- 视觉前缀引导机制--->进一步图像编码，只提取最高层语义特征
- Multihead-Attention --->设置两层注意力模块，qkv自定义输入
- Softmax概率输出

## 总体性能

|    Epoch    | Accuracy | Precision |  Recall  |    F1    |
| :---------: | :------: | :-------: | :------: | :------: |
|    Run1     |   0.90   |   0.78    |   0.76   |   0.77   |
|    Run2     |   0.91   |   0.80    |   0.78   |   0.79   |
|    Run3     |   0.91   |   0.80    |   0.78   |   0.79   |
| **Average** | **0.90** | **0.79**  | **0.77** | **0.78** |

