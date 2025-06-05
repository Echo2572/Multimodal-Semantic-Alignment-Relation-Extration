import json
import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timm
from processor.dataset import MMREProcessor, MMREDataset

warnings.filterwarnings("ignore")

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


# ViT模型
class ViTOnlyModel(nn.Module):
    def __init__(self, num_labels):
        super(ViTOnlyModel, self).__init__()
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )
        state_dict = torch.load("H:/Pretrained Model/vit-base-patch16-224/pytorch_model.bin")
        self.vit.load_state_dict(state_dict, strict=False)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 * 2, num_labels)  # head + tail 拼接

    def forward(self, images, labels=None):
        feats = self.vit(images)
        combined = torch.cat([feats, feats], dim=-1)
        out = self.dropout(combined)
        logits = self.classifier(out)
        probs = torch.softmax(logits, dim=-1)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.view(-1))
            return loss, probs
        return probs


# 训练函数
def train_model(model, train_loader, num_epochs=10, lr=2e-5, model_name="ViTOnly"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            print(batch_idx,end=',')
            images = batch[4].to(device)
            relations = batch[3].to(device)

            optimizer.zero_grad()
            try:
                loss, probs = model(images=images, labels=relations)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = probs.argmax(-1)
                correct += (preds == relations).sum().item()
                total += relations.size(0)

                print(f"{model_name} Batch {batch_idx}", end="\r")
            except RuntimeError as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                continue

        print(
            f"{model_name} Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {total_loss / len(train_loader):.4f}, "
            f"Accuracy: {correct / total:.4f}"
        )

    os.makedirs("../save", exist_ok=True)
    save_path = os.path.join("../save", "baseline_vit.pth")
    torch.save(model.state_dict(), save_path)
    print(f"{model_name} 模型已保存至 {save_path}")


# 测试函数（含详细打印）
def test_model(model, test_loader, rel2id, model_name="ViTOnly"):
    model.eval()
    all_preds = []
    all_labels = []
    id2rel = {v: k for k, v in rel2id.items()}
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch[4].to(device)
            relations = batch[3].to(device)

            probs = model(images=images)
            preds = probs.argmax(-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(relations.cpu().tolist())

            # 打印每个样本的预测细节
            batch_size = images.size(0)
            for j in range(batch_size):
                true_id = relations[j].item()
                pred_id = preds[j]
                print(f"\n{model_name} Sample {i * batch_size + j + 1}:")
                print(f"True Relation: {id2rel[true_id]} (ID: {true_id})")
                print(f"Predicted Relation: {id2rel[pred_id]} (ID: {pred_id})")
                prob_list = [f"{prob:.4f}" for prob in probs[j].tolist()]
                print(f"Probabilities: {prob_list}")

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n\033[36m{model_name} Test Results:\033[0m")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# 主函数
def main():
    processor = MMREProcessor(DATA_PATH)
    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = MMREDataset(
        processor,
        transform,
        IMG_PATH,
        AUX_PATH,
        sample_ratio=0.5,
        mode="train"
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    test_dataset = MMREDataset(
        processor,
        transform,
        IMG_PATH,
        AUX_PATH,
        sample_ratio=1.0,
        mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    model = ViTOnlyModel(num_labels)
    print("加载本地预训练 ViT 模型")

    # print("\n开始训练 ViTOnly 模型...")
    # train_model(model, train_loader, num_epochs=10, lr=2e-5, model_name="ViTOnly")

    print("\n开始测试 ViTOnly 模型...")
    with open(DATA_PATH['re_path'], 'r') as f:
        rel2id = json.load(f)
    results = test_model(model, test_loader, rel2id, model_name="ViTOnly")

    save_dir = "../save"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "baseline_vit_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"测试结果保存至 {os.path.join(save_dir, 'baseline_vit_results.json')}")


if __name__ == '__main__':
    main()
