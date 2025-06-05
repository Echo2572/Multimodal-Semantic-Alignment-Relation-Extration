import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MMREProcessor(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.re_path = data_path['re_path']
        self.tokenizer = BertTokenizer.from_pretrained("H:/Pretrained Model/bert-base-uncased", do_lower_case=True,local_files_only=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']})

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

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

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
            assert len(sample_words) == len(sample_relations) == len(sample_imgids), "{}, {}, {}".format(
                len(sample_words), len(sample_relations), len(sample_imgids))
            return {'words': sample_words, 'relations': sample_relations, 'heads': sample_heads, 'tails': sample_tails, \
                    'imgids': sample_imgids, 'dataid': sample_dataid, 'aux_imgs': aux_imgs}

        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict


class MMREDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, sample_ratio=1.0,
                 mode="train") -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.mode = mode

        self.data_dict = self.processor.load_from_file(mode, sample_ratio)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
        self.data_dict['heads'][idx], self.data_dict['tails'][idx], self.data_dict['imgids'][idx]
        item_id = self.data_dict['dataid'][idx]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
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
        extend_word_list = " ".join(extend_word_list)

        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
        encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

        re_label = self.re_dict[relation]

        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)

            if self.aux_img_path is not None:
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                for i in range(min(3, len(aux_img_paths))):
                    try:
                        aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    except FileNotFoundError:
                        aux_img = Image.open(os.path.join(self.aux_img_path, 'inf.png')).convert('RGB')

                    aux_img = self.transform(aux_img)
                    aux_imgs.append(aux_img)

                for i in range(3 - len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3
            return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, img_path
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label)
