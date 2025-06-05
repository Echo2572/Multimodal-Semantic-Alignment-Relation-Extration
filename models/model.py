import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, CLIPVisionModel


# 图像编码器
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained("H:/Pretrained Model/clip-vit-base-patch32", local_files_only=True)

    def forward(self, imgs, aux_imgs=True):
        prompt_guids = self.get_vision_prompt(imgs)
        if aux_imgs is not None:  # 输入aux_imgs:(bsz,3[num],3,224,224)
            aux_prompt_guids = []
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # 维度变换:(3[num],bsz,3,224,224)

            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_vision_prompt(aux_imgs[i])
                aux_prompt_guids.append(aux_prompt_guid)
                # prompt_guids:(13,bsz,50,768) aux_prompt_guids:(3[num],13,bsz,50,768)
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    # 输入x:(bsz,3,224,224)
    def get_vision_prompt(self, x):
        prompt_guids = list(self.clip(x, output_hidden_states=True).hidden_states)
        return prompt_guids  # (13,bsz,50,768)


# Multimodal Semantic Alignment Relation Extraction Model
class MSAREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(MSAREModel, self).__init__()
        self.bert = BertModel.from_pretrained('H:/Pretrained Model/bert-base-uncased')
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.dropout = nn.Dropout(0.5)  # 正则化

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.bert.config.hidden_size * 4, num_labels)
        )

        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        self.init_params = []

        if self.args.use_prompt:
            self.image_model = ImageModel()

            self.feature_adapter = nn.Linear(768, 64)  # 特征压缩

            # 图像特征-->视觉前缀引导
            self.encoder_conv = nn.Sequential(
                nn.Linear(in_features=64 * 50, out_features=2000),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(in_features=2000, out_features=768 * 2 * self.args.prompt_len)  # 输出 prompt_len 个 key-value 对
            )

        # 跨模态注意力融合机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=8,
            dropout=0.1
        )

        # 额外注意力层: 精炼融合后的特征
        self.additional_attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=8,
            dropout=0.1
        )

        # 对比学习,计算I-T对之间的对比损失
        if self.args.use_contrastive:
            self.temp = nn.Parameter(torch.ones([]) * self.args.temp)

            # 投影到指定维度的嵌入空间
            self.vision_proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 4 * self.bert.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4 * self.bert.config.hidden_size, self.args.embed_dim)
            )
            self.text_proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 4 * self.bert.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4 * self.bert.config.hidden_size, self.args.embed_dim)
            )

        # 匹配层,2分类判断I-T对是否匹配
        if self.args.use_matching:
            self.itm_head = nn.Sequential(
                nn.Linear(768, 768 * 4),
                nn.Tanh(),
                nn.Linear(768 * 4, 2)
            )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, images=None, aux_imgs=None):
        bsz = input_ids.size(0)

        with torch.no_grad():
            # (13,bsz,50,768),(3[num],13,bsz,50,768)
            images, aux_images = self.image_model(images, aux_imgs)

        image_features = images[-1]  # (bsz, 50, 768)

        if aux_images is not None:
            aux_features = [aux_image[-1] for aux_image in aux_images]  # (3,bsz,50,768)
            aux_features = torch.stack(aux_features).mean(dim=0)  # (bsz, 50, 768)
            image_features = (image_features + aux_features) / 2

        # 图像注意力掩码 (bsz, 50)
        image_atts = torch.ones((bsz, image_features.shape[1])).to(self.args.device)

        # 文本编码器
        text_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_last_hidden_state = text_output.last_hidden_state  # (bsz,seq_len,768)

        # 文本序列长度,图像序列长度
        text_seq_len = text_last_hidden_state.size(1)
        image_seq_len = image_features.size(1)

        combined_features = torch.cat((text_last_hidden_state, image_features), dim=1)  # (bsz,text_seq_len+image_seq_len,768)
        combined_att_mask = torch.cat((attention_mask, image_atts), dim=1)  # (bsz,text_seq_len+image_seq_len)

        if self.args.use_prompt:
            compressed_features = self.feature_adapter(image_features)  # (bsz,50,64)
            flattened_features = compressed_features.reshape(bsz, -1)  # (bsz,50*64)

            prompt_embeds = self.encoder_conv(flattened_features)  # (bsz,768*2*prompt_len)
            prompt_embeds = prompt_embeds.view(bsz, 2 * self.args.prompt_len, 768)  # (bsz,2*prompt_len,768)
            prompt_key, prompt_value = prompt_embeds.chunk(2, dim=1)  # (bsz, prompt_len, 768)

            # (bsz,text_seq_len+image_seq_len+prompt_len,768)
            key = torch.cat((combined_features, prompt_key), dim=1)
            value = torch.cat((combined_features, prompt_value), dim=1)

            prompt_att_mask = torch.ones((bsz, self.args.prompt_len)).to(self.args.device)  # (bsz,prompt_len)
            key_att_mask = torch.cat((combined_att_mask, prompt_att_mask), dim=1)  # (bsz,text_seq_len+image_seq_len+prompt_len)
            attn_mask = (key_att_mask == 0)
        else:
            key = combined_features
            value = combined_features
            attn_mask = (combined_att_mask == 0)

        combined_features = combined_features.transpose(0, 1)  # (text_seq_len+image_seq_len,bsz,768)

        # (text_seq_len+image_seq_len+prompt_len,bsz,768)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        # (text_seq_len+image_seq_len,bsz,768)
        fused_features, _ = self.cross_attention(
            query=combined_features,
            key=key,
            value=value,
            key_padding_mask=attn_mask
        )

        fused_feat=ures = fused_features + combined_features  # 融合后的特征

        src_len = fused_features.size(0)
        if attn_mask.size(1) != src_len:
            attn_mask = attn_mask[:, :src_len]

        # 精炼融合特征
        additional_fused_features, _ = self.additional_attention(
            query=fused_features,
            key=fused_features,
            value=fused_features,
            key_padding_mask=attn_mask
        )
        additional_fused_features = additional_fused_features + fused_features
        additional_fused_features = additional_fused_features.transpose(0, 1)  # (bsz,text_seq_len+image_seq_len,768)
        fusion_last_hidden_state = additional_fused_features[:, :text_seq_len, :]  # (bsz,text_seq_len,768)
        entity_hidden_state = torch.zeros(bsz, 2 * self.bert.config.hidden_size, device=self.args.device)

        # 获取头尾实体特征
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = fusion_last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = fusion_last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)

        logits = self.classifier(entity_hidden_state)  # 分类

        # 计算损失函数
        if labels is not None and self.args.use_contrastive:
            loss_fn = nn.CrossEntropyLoss()
            text_feats = self.text_proj(text_last_hidden_state[:, 0, :])
            image_feats = self.vision_proj(image_features[:, 0, :])
            cl_loss = self.get_contrastive_loss(text_feats, image_feats)

            if self.args.use_matching:
                neg_output, itm_label = self.get_matching_loss(
                    image_features, image_atts, image_feats,
                    text_last_hidden_state, attention_mask, text_feats
                )
                matching_loss = loss_fn(neg_output, itm_label)
                main_loss = loss_fn(logits, labels.view(-1))
                loss = 0.4 * main_loss + 0.2 * matching_loss + 0.4 * cl_loss
            else:
                main_loss = loss_fn(logits, labels.view(-1))
                loss = 0.6 * main_loss + 0.4 * cl_loss
            return loss, logits

        elif labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            main_loss = loss_fn(logits, labels.view(-1))
            return main_loss, logits

        return logits

    # 获取匹配损失
    def get_matching_loss(self, image_features, image_atts, image_feat, text_embeds, text_atts, text_feat):
        bsz = text_embeds.shape[0]

        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bsz):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item() if torch.sum(weights_t2i[b]) > 0 else 0
            image_embeds_neg.append(image_features[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bsz):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item() if torch.sum(weights_i2t[b]) > 0 else 0
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_all = torch.cat([torch.stack(text_embeds_neg), text_embeds], dim=0)
        text_atts_all = torch.cat([torch.stack(text_atts_neg), text_atts], dim=0)
        image_embeds_all = torch.cat([torch.stack(image_embeds_neg), image_features], dim=0)
        image_atts_all = torch.cat([torch.stack(image_atts_neg), image_atts], dim=0)

        combined_features = torch.cat([text_embeds_all, image_embeds_all], dim=1)

        compressed_features = self.feature_adapter(image_embeds_all)
        flattened_features = compressed_features.reshape(2 * bsz, -1)

        prompt_embeds = self.encoder_conv(flattened_features)
        prompt_embeds = prompt_embeds.view(2 * bsz, 2 * self.args.prompt_len, 768)
        prompt_key, prompt_value = prompt_embeds.chunk(2, dim=1)

        key = torch.cat((combined_features, prompt_key), dim=1)
        value = torch.cat((combined_features, prompt_value), dim=1)

        combined_att_mask = torch.cat((text_atts_all, image_atts_all), dim=1)
        prompt_att_mask = torch.ones((2 * bsz, self.args.prompt_len)).to(self.args.device)
        key_att_mask = torch.cat((combined_att_mask, prompt_att_mask), dim=1)
        key_att_mask = (key_att_mask == 0)

        combined_features = combined_features.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        fused_features, _ = self.cross_attention(
            query=combined_features,
            key=key,
            value=value,
            key_padding_mask=key_att_mask
        )

        fused_features = fused_features + combined_features

        src_len = fused_features.size(0)
        if key_att_mask.size(1) != src_len:
            key_att_mask = key_att_mask[:, :src_len]

        additional_fused_features, _ = self.additional_attention(
            query=fused_features,
            key=fused_features,
            value=fused_features,
            key_padding_mask=key_att_mask
        )
        additional_fused_features = additional_fused_features + fused_features
        additional_fused_features = additional_fused_features.transpose(0, 1)
        neg_output = self.itm_head(additional_fused_features[:, 0, :])
        itm_labels = torch.cat([torch.zeros(bsz, dtype=torch.long), torch.ones(bsz, dtype=torch.long)], dim=0).to(self.args.device)

        return neg_output, itm_labels

    # 获取对比学习损失
    def get_contrastive_loss(self, text_feat, image_feat):
        logits = text_feat @ image_feat.t() / self.temp
        bsz = text_feat.shape[0]
        labels = torch.arange(bsz, device=self.args.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2
        return loss
