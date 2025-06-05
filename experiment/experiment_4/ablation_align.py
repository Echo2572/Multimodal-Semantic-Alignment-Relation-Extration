import json
import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import ast
import random

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


# 数据处理器（与原始 MMREProcessor 一致）
class MMREProcessor(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.re_path = data_path['re_path']

    def load_from_file(self, mode="train", sample_ratio=1.0):
        load_file = self.data_path[mode]
        print("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h'])
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == len(imgids)

        aux_path = self.data_path[mode + "_auximgs"]
        aux_imgs = torch.load(aux_path)

        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(words))), k=int(len(words) * sample_ratio))
            sample_words = [words[idx] for idx in sample_indexes]
            sample_relations = [relations[idx] for idx in sample_indexes]
            sample_heads = [heads[idx] for idx in sample_indexes]
            sample_tails = [tails[idx] for idx in sample_indexes]
            sample_imgids = [imgids[idx] for idx in sample_indexes]
            sample_dataid = [dataid[idx] for idx in sample_indexes]
            return {'words': sample_words, 'relations': sample_relations, 'heads': sample_heads, 'tails': sample_tails,
                    'imgids': sample_imgids, 'dataid': sample_dataid, 'aux_imgs': aux_imgs}

        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict


# 数据集类（适配 BERT 和 CLIP）
class CLIPMMREDataset(Dataset):
    def __init__(self, processor, bert_tokenizer, clip_processor, img_path=None, aux_img_path=None, max_seq=40,
                 sample_ratio=1.0, mode="train"):
        self.processor = processor
        self.bert_tokenizer = bert_tokenizer
        self.clip_processor = clip_processor
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.mode = mode

        self.data_dict = self.processor.load_from_file(mode, sample_ratio)
        self.re_dict = self.processor.get_relation_dict()

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = (
            self.data_dict['words'][idx],
            self.data_dict['relations'][idx],
            self.data_dict['heads'][idx],
            self.data_dict['tails'][idx],
            self.data_dict['imgids'][idx]
        )
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        head_name, tail_name = head_d['name'], tail_d['name']

        # 构造文本，插入 <s> 和 <o> 标记
        extend_word_list = []
        for i in range(len(word_list)):
            if i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        text = " ".join(extend_word_list)

        # 使用 BERT tokenizer 处理文本
        encoded = self.bert_tokenizer(
            text,
            max_length=self.max_seq,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)  # [max_seq]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [max_seq]
        token_type_ids = encoded.get('token_type_ids', torch.zeros_like(input_ids)).squeeze(0)  # [max_seq]

        # 加载图像
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), color='black')  # 占位图像

        # 使用 CLIPProcessor 处理图像
        processed = self.clip_processor(
            images=image,
            return_tensors="pt",
            padding=True
        )
        pixel_values = processed["pixel_values"].squeeze(0)  # [3, 224, 224]

        # 关系标签
        re_label = self.re_dict[relation]

        return input_ids, attention_mask, token_type_ids, pixel_values, torch.tensor(re_label), head_name, tail_name


# BERT 和 CLIP 融合模型
class BERTCLIPFusionModel(nn.Module):
    def __init__(self, num_labels, bert_path, clip_path, bert_tokenizer):
        super(BERTCLIPFusionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.clip = CLIPModel.from_pretrained(clip_path)
        self.vision_model = self.clip.vision_model  # 仅使用图像编码器
        self.bert_tokenizer = bert_tokenizer
        self.dropout = nn.Dropout(0.3)
        # BERT 实体特征 (768) + CLIP 图像特征 (768) = 1536 维
        self.classifier = nn.Linear(768 + 768, num_labels)
        self.device = device
        self.to(self.device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None, labels=None):
        # 提取 BERT 文本特征
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden_states = bert_outputs.last_hidden_state  # (bsz, seq_len, 768)

        # 提取头尾实体特征（基于 <s> 和 <o> 标记）
        bsz = input_ids.size(0)
        head_features = torch.zeros(bsz, 768, device=self.device)
        tail_features = torch.zeros(bsz, 768, device=self.device)
        head_token_id = self.bert_tokenizer.convert_tokens_to_ids('<s>')
        tail_token_id = self.bert_tokenizer.convert_tokens_to_ids('<o>')
        for i in range(bsz):
            head_indices = input_ids[i].eq(head_token_id).nonzero(as_tuple=True)[0]
            tail_indices = input_ids[i].eq(tail_token_id).nonzero(as_tuple=True)[0]
            head_idx = head_indices[0] if head_indices.numel() > 0 else torch.tensor(0, device=self.device)  # 回退到 [CLS]
            tail_idx = tail_indices[0] if tail_indices.numel() > 0 else torch.tensor(0, device=self.device)  # 回退到 [CLS]
            head_features[i] = hidden_states[i, head_idx, :]  # (768,)
            tail_features[i] = hidden_states[i, tail_idx, :]  # (768,)
        text_features = (head_features + tail_features) / 2  # 平均头尾特征 (bsz, 768)

        # 提取 CLIP 图像特征
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.pooler_output  # (bsz, 768)

        # 简单拼接特征
        combined_features = torch.cat([text_features, image_features], dim=-1)  # (bsz, 1536)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        probs = torch.softmax(logits, dim=-1)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.view(-1))
            return loss, probs
        return probs


# 训练函数
def train_model(model, train_loader, num_epochs=10, lr=2e-5, model_name="BERTCLIPFusion"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            print(batch_idx,end=',')
            # 解包7个值
            input_ids, attention_mask, token_type_ids, pixel_values, labels, _, _ = batch
            input_ids, attention_mask, token_type_ids, pixel_values, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                token_type_ids.to(device),
                pixel_values.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            try:
                loss, probs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    pixel_values=pixel_values,
                    labels=labels,
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = probs.argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

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
    save_path = os.path.join(save_dir, "baseline_bert_clip_fusion.pth")
    torch.save(model.state_dict(), save_path)
    print(f"{model_name} 模型保存至 {save_path}")


# 测试函数
def test_model(model, test_loader, rel2id, model_name="BERTCLIPFusion"):
    model.eval()
    all_preds = []
    all_labels = []
    id2rel = {v: k for k, v in rel2id.items()}
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 解包7个值
            input_ids, attention_mask, token_type_ids, pixel_values, labels, head_name, tail_name = batch
            input_ids, attention_mask, token_type_ids, pixel_values, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                token_type_ids.to(device),
                pixel_values.to(device),
                labels.to(device),
            )
            _, probs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                labels=labels,
            )

            preds = probs.argmax(-1).cpu().tolist()
            batch_size = labels.size(0)

            # 打印每个样本的结果
            for j in range(batch_size):
                print(f"\n{model_name} Sample {i * batch_size + j + 1}:")
                print(f"Head Entity: {head_name[j]}")
                print(f"Tail Entity: {tail_name[j]}")
                print(f"True Relation: {id2rel[labels[j].item()]} (ID: {labels[j].item()})")
                print(f"Predicted Relation: {id2rel[preds[j]]} (ID: {preds[j]})")
                prob_list = [f"{prob:.4f}" for prob in probs[j].tolist()]
                print(f"Probabilities: {prob_list}")

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

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

    # BERT 和 CLIP 处理器
    bert_path = "H:/Pretrained Model/bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    # 添加特殊标记
    special_tokens = {'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']}
    bert_tokenizer.add_special_tokens(special_tokens)
    clip_processor = CLIPProcessor.from_pretrained("H:/Pretrained Model/clip-vit-base-patch32")

    # 训练数据集
    train_dataset = CLIPMMREDataset(
        processor,
        bert_tokenizer,
        clip_processor,
        img_path=IMG_PATH,
        aux_img_path=AUX_PATH,
        sample_ratio=0.5,
        mode="train"
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    # 测试数据集
    test_dataset = CLIPMMREDataset(
        processor,
        bert_tokenizer,
        clip_processor,
        img_path=IMG_PATH,
        aux_img_path=AUX_PATH,
        sample_ratio=1.0,
        mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    # 初始化模型
    clip_path = "H:/Pretrained Model/clip-vit-base-patch32"
    model = BERTCLIPFusionModel(num_labels, bert_path, clip_path, bert_tokenizer)
    # 调整 BERT 嵌入层以适应新词汇表
    model.bert.resize_token_embeddings(len(bert_tokenizer))
    print("加载本地预训练 BERT 和 CLIP 模型")

    # 训练
    # print("\n开始训练 BERTCLIPFusion 模型...")
    # train_model(model, train_loader, num_epochs=10, lr=2e-5, model_name="BERTCLIPFusion")

    # 测试
    print("\n开始测试 BERTCLIPFusion 模型...")
    with open(DATA_PATH['re_path'], 'r') as f:
        rel2id = json.load(f)
    results = test_model(model, test_loader, rel2id, model_name="BERTCLIPFusion")

    # 保存结果
    save_dir = "../save"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "baseline_bert_clip_fusion_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"测试结果已保存至 {os.path.join(save_dir, 'baseline_bert_clip_fusion_results.json')}")


if __name__ == '__main__':
    main()
