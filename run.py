import os
import argparse
import torch
import numpy as np
import random
import warnings
from torchvision import transforms
from torch.utils.data import DataLoader

from models.model import MSAREModel
from processor.dataset import MMREProcessor, MMREDataset
from modules.train import RETrainer

warnings.filterwarnings("ignore", category=UserWarning)

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


def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=10, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=10, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")

    parser.add_argument('--use_prompt', action='store_true')

    parser.add_argument('--use_contrastive', action='store_true')
    parser.add_argument('--temp', default=0.07, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--use_matching', action='store_true')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    args = parser.parse_args()

    global IMG_PATH, AUX_PATH

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed)
    print(args)

    if not args.use_prompt:
        IMG_PATH, AUX_PATH = None, None

    processor = MMREProcessor(DATA_PATH)

    train_dataset = MMREDataset(processor, transform, IMG_PATH, AUX_PATH, args.max_seq, sample_ratio=1.0,
                                mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

    dev_dataset = MMREDataset(processor, transform, IMG_PATH, AUX_PATH, args.max_seq, sample_ratio=0.5, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = MMREDataset(processor, transform, IMG_PATH, AUX_PATH, args.max_seq, sample_ratio=1.0, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True)

    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    tokenizer = processor.tokenizer
    model = MSAREModel(num_labels, tokenizer, args=args)

    trainer = RETrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                        processor=processor, args=args)

    if args.do_train:
        trainer.train()
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        args.load_path = os.path.join(args.save_path, 'my_12247_1234.pth')
        trainer.test()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
