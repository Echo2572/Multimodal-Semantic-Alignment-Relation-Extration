import json
import os
import warnings
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel
from processor.dataset import MMREProcessor, MMREDataset
from torchvision import transforms

warnings.filterwarnings("ignore")

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
DATA_PATH = {
    'train': 'G:/dataset/MNRE_data/txt/ours_train.txt',
    'dev': 'G:/dataset/MNRE_data/txt/ours_val.txt',
    'test': 'G:/dataset/MNRE_data/txt/ours_test.txt',
    'train_auximgs': 'G:/dataset/MNRE_data/txt/mre_train_dict.pth',
    'dev_auximgs': 'G:/dataset/MNRE_data/txt/mre_dev_dict.pth',
    'test_auximgs': 'G:/dataset/MNRE_data/txt/mre_test_dict.pth',
    're_path': 'G:/dataset/MNRE_data/ours_rel2id.json'
}

IMG_PATH = {
    'train': 'G:/dataset/MNRE_data/img_org/train/',
    'dev': 'G:/dataset/MNRE_data/img_org/val/',
    'test': 'G:/dataset/MNRE_data/img_org/test'
}

AUX_PATH = {
    'train': 'G:/dataset/MNRE_data/img_vg/train/crops',
    'dev': 'G:/dataset/MNRE_data/img_vg/val/crops',
    'test': 'G:/dataset/MNRE_data/img_vg/test/crops'
}

# 文本单模态模型（基于预训练BERT）
class TextOnlyModel(nn.Module):
    def __init__(self, num_labels, tokenizer):
        super(TextOnlyModel, self).__init__()
        self.bert = BertModel.from_pretrained("H:/Pretrained Model/bert-base-uncased")
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.device = device
        self.to(self.device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        bsz = input_ids.size(0)
        # 文本特征提取
        text_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_last_hidden_state = text_output.last_hidden_state  # (bsz, text_seq_len, 768)

        # 提取头尾实体表示
        entity_hidden_state = torch.Tensor(bsz, 2 * 768).to(self.device)
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = text_last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = text_last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)

        entity_hidden_state = self.dropout(entity_hidden_state)
        logits = self.classifier(entity_hidden_state)
        probs = torch.softmax(logits, dim=-1)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.view(-1))
            return loss, probs
        return probs

# 训练函数
def train_model(model, train_loader, num_epochs=10, lr=2e-5, model_name="TextOnly"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # 解包7个值，仅使用文本相关项
            input_ids, token_type_ids, attention_mask, relations, _, _, _ = batch
            input_ids, attention_mask, token_type_ids, relations = (
                input_ids.to(device),
                attention_mask.to(device),
                token_type_ids.to(device),
                relations.to(device),
            )
            optimizer.zero_grad()
            try:
                loss, probs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=relations,
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = probs.argmax(-1)
                correct += (preds == relations).sum().item()
                total += relations.size(0)

                print(f"{model_name} Batch {batch_idx},", end="\r")
            except RuntimeError as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                continue

        print(
            f"{model_name} Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {total_loss / len(train_loader):.4f}, "
            f"Accuracy: {correct / total:.4f}"
        )

    # 保存模型
    save_dir = "../save"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "baseline_bert.pth")
    torch.save(model.state_dict(), save_path)
    print(f"{model_name} 模型保存至 {save_path}")

# 测试函数
def test_model(model, test_loader, rel2id, model_name="TextOnly"):
    model.eval()
    all_preds = []
    all_labels = []
    id2rel = {v: k for k, v in rel2id.items()}
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 解包7个值，仅使用文本相关项
            input_ids, token_type_ids, attention_mask, relations, _, _, _ = batch
            input_ids, attention_mask, token_type_ids, relations = (
                input_ids.to(device),
                attention_mask.to(device),
                token_type_ids.to(device),
                relations.to(device),
            )
            _, probs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=relations,
            )

            preds = probs.argmax(-1).cpu().tolist()
            batch_size = relations.size(0)

            # 打印每个样本的结果（无实体名称）
            for j in range(batch_size):
                print(f"\n{model_name} Sample {i * batch_size + j + 1}:")
                print(f"True Relation: {id2rel[relations[j].item()]} (ID: {relations[j].item()})")
                print(f"Predicted Relation: {id2rel[preds[j]]} (ID: {preds[j]})")
                prob_list = [f"{prob:.4f}" for prob in probs[j].tolist()]
                print(f"Probabilities: {prob_list}")

            all_preds.extend(preds)
            all_labels.extend(relations.cpu().tolist())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n\033[33m{model_name} Test Results:\033[0m")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# 主函数
def main():
    # 初始化数据处理器
    processor = MMREProcessor(DATA_PATH)
    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    tokenizer = processor.tokenizer

    # 占位变换（文本模态不使用）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 训练数据集
    train_dataset = MMREDataset(
        processor,
        transform,
        IMG_PATH,
        AUX_PATH,
        sample_ratio=0.01,
        mode="train"
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    # 测试数据集
    test_dataset = MMREDataset(
        processor,
        transform,
        IMG_PATH,
        AUX_PATH,
        sample_ratio=1,
        mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    # 初始化模型
    model = TextOnlyModel(num_labels, tokenizer)
    print("加载预训练 BERT 模型")

    # 训练
    # print("\n开始训练 TextOnly 模型...")
    # train_model(model, train_loader, num_epochs=10, lr=2e-5, model_name="TextOnly")

    # 测试
    print("\n开始测试 TextOnly 模型...")
    with open(DATA_PATH['re_path'], 'r') as f:
        rel2id = json.load(f)
    results = test_model(model, test_loader, rel2id, model_name="TextOnly")

    # 保存结果
    save_dir = "../save"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "baseline_bert_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"测试结果已保存至 {os.path.join(save_dir, 'baseline_bert_results.json')}")

if __name__ == '__main__':
    main()