import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report as sk_classification_report
from transformers.optimization import get_linear_schedule_with_warmup
import re
from .metrics import eval_result


class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()


class RETrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, args=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.re_dict = processor.get_relation_dict()
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs

        self.step = 0
        self.args = args

        if self.args.use_prompt:
            self.before_multimodal_train()
        else:
            self.before_train()

    def train(self):
        self.step = 0
        self.model.train()
        print("***** Running training *****")
        print("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        print("  Num epoch = %d", self.args.num_epochs)
        print("  Batch size = %d", self.args.batch_size)
        print("  Learning rate = {}".format(self.args.lr))
        print("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:
            print("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            print("Load model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels, _ = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)

            pbar.close()
            self.pbar = None
            print("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            print("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        print("***** Running evaluate *****")
        print("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        print("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0

                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels, _ = self._step(batch, mode="dev")
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()

                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values()),
                                                     target_names=list(self.re_dict.keys()), digits=4)
                print("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)

                print("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}." .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:
                    print("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1

                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                        print("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self):
        self.model.eval()
        print("\n***** Running testing *****")
        print("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        print("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:
            print("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            print("Load model successful!")
        true_labels, pred_labels, img_path = [], [], []

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                id2rel = {v: k for k, v in self.re_dict.items()}

                for i, batch in enumerate(self.test_data):
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels, i_paths = self._step(batch, mode="dev")
                    total_loss += loss.detach().cpu().item()

                    probs = torch.softmax(logits, dim=-1)
                    preds = logits.argmax(-1)

                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    img_path.extend(i for i in i_paths)

                    batch_size = logits.size(0)
                    for j in range(batch_size):
                        print(f"\nTest Sample {i * batch_size + j + 1}:")
                        print(f"Image Path: {i_paths[j]}")
                        print(f"True Relation: {id2rel[labels[j].item()]} (ID: {labels[j].item()})")
                        print(f"Predicted Relation: {id2rel[preds[j].item()]} (ID: {preds[j].item()})")
                        prob_list = [f"{prob:.4f}" for prob in probs[j].tolist()]
                        print(f"Probabilities: {prob_list}")

                    pbar.update()

                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values()),
                                                     target_names=list(self.re_dict.keys()), digits=4)
                print("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                print("Test f1 score: {}, acc: {}.".format(micro_f1, acc))

    def _step(self, batch, mode="train"):
        if mode != "predict":
            if self.args.use_prompt:
                input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, i_path = batch
            else:
                images, aux_imgs, i_path = None, None, None
                input_ids, token_type_ids, attention_mask, labels = batch
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 labels=labels, images=images, aux_imgs=aux_imgs)
            return outputs, labels, i_path

    def before_train(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    def before_multimodal_train(self):
        optimizer_grouped_parameters = []
        contrastive_task = re.compile(r'bert\.(encoder\.layer\.[0-5]\..|embeddings\..)')
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []

        for name, param in self.model.named_parameters():
            if contrastive_task.search(name) or 'temp' in name or 'temp_lamb' in name \
                    or 'vision_proj' in name or 'text_proj' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        main_task = re.compile(r'bert\.(encoder\.layer\.([6-9]|10|11)\..)')
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []

        for name, param in self.model.named_parameters():
            if main_task.search(name):
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'itm_head' in name \
                    or 'alpha' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        for name, param in self.model.named_parameters():
            if 'image_model' in name:
                param.require_grad = False

        self.optimizer = optimizer_grouped_parameters
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
