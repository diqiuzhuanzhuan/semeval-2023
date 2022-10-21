# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from typing import Any, AnyStr, Union, Optional, overload
from torch.utils.data import Dataset
from allennlp.common.registrable import Registrable
from transformers import AutoTokenizer
import os, torch
import pytorch_lightning as pl
from task2.configuration import config
from allennlp.common.params import Params
from task2.data_man.meta_data import ConllItem, read_conll_item_from_file, get_id_by_type, get_type_by_id, get_id_to_labes_map
from task2.data_man.meta_data import _assign_ner_tags, extract_spans


class ConllDataset(Dataset, Registrable):
    
    def __init__(
        self,
        encoder_model='bert-base-uncased'
        ) -> None:
        super().__init__()
        self.instances = []
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model, add_prefix_space=True)
    @overload
    def __getitem__(self, index: Any) -> Any:
        ...

    @overload
    def encode_input(self, item) -> Any:
        ...

    def __len__(self) -> int:
        return len(self.instances)

    def read_data(self, conll_file: Union[AnyStr, os.PathLike]):
        self.instances = read_conll_item_from_file(conll_file)

        
@ConllDataset.register('baseline_dataset')
class BaselineDataset(ConllDataset):

    def __getitem__(self, index: Any) -> Any:
        if index >= self.__len__():
            raise IndexError('index value must be not more than the maximum length.')
        return self.encode_input(self.instances[index])

    def encode_input(self, item: ConllItem):
        id, tokens, labels = item.id, item.tokens, item.labels
        token_masks, new_labels, input_ids, token_type_ids, attention_mask, label_ids = [], [], [], [], [], []

        input_ids.append(self.tokenizer.cls_token_id)
        label_ids.append(get_id_by_type('O'))
        attention_mask.append(1)
        token_type_ids.append(0)
        token_masks.append(False)

        
        for i, token in enumerate(tokens):
            outputs = self.tokenizer(token.lower())
            subtoken_len = len(outputs['input_ids']) - 2
            input_ids.extend(outputs['input_ids'][1:-1])
            attention_mask.extend(outputs['attention_mask'][1:-1])
            token_type_ids.extend([0] * subtoken_len)
            token_masks.extend([True]+ [False] * (subtoken_len-1))
            if labels is not None:
                tag = labels[i]
                sub_tags = [tag] + [tag.replace('B-', 'I-')] * (subtoken_len-1)
                label_ids.extend([get_id_by_type(sub_tag) for sub_tag in sub_tags])


        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(get_id_by_type('O'))
        attention_mask.append(1)
        token_type_ids.append(0)
        token_masks.append(False)
        gold_spans = extract_spans([get_type_by_id(label_id) for label_id in label_ids])

        return id, input_ids, token_type_ids, attention_mask, token_masks, label_ids, gold_spans

        
        
class ConllDataModule(Registrable, pl.LightningDataModule):
    pass

@ConllDataModule.register('baseline_data_module')
class BaselineDataModule(ConllDataModule):
    def __init__(
        self,
        reader: ConllDataset,
        lang: AnyStr='English',
        batch_size: int=16
        ) -> None:
        super().__init__()
        self.reader = reader
        self.batch_size = batch_size
        self.lang = lang

    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.reader.read_data(config.train_file[self.lang])
        if stage == 'validate':
            self.reader.read_data(config.validate_file[self.lang])

        if stage == 'test':
            pass
            
        if stage == 'predict':
            pass

        self.stage = stage


    def collate_batch(self, batch):
        batch_size = len(batch)
        batch_ = list(zip(*batch))
        id, input_ids, token_type_ids, attention_mask, token_masks, label_ids, gold_spans = batch_
        max_len = max([len(_) for _ in input_ids])
        input_ids_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        token_type_ids_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        attention_mask_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        if label_ids:
            max_label_len = max([len(_) for _ in label_ids])
            label_ids_tensor = torch.empty(size=[batch_size, max_label_len], dtype=torch.long).fill_(0)
        else:
            label_ids_tensor = None
            gold_spans = None
        for i in range(batch_size):
            available_length = len(input_ids[i])
            input_ids_tensor[i][0:available_length] = torch.tensor(input_ids[i], dtype=torch.long)
            token_type_ids_tensor[i][0:available_length] = torch.tensor(token_type_ids[i], dtype=torch.long)
            attention_mask_tensor[i][0:available_length] = torch.tensor(attention_mask[i], dtype=torch.long)
            if label_ids_tensor is not None:
                label_ids_length = len(label_ids[i])
                label_ids_tensor[i][:label_ids_length] = torch.tensor(label_ids[i], dtype=torch.long)

        return id, input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, token_masks, label_ids_tensor, gold_spans

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=4)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=4)


        
if __name__ == '__main__':
    params = Params({
        'type': 'baseline_dataset',
        'encoder_model': 'xlm-roberta-base' 
    })
    baseline_dataset = ConllDataset.from_params(params)
    baseline_dataset.read_data('./task2/data/semeval_2021_task_11_trial_data.txt')
    for ele in baseline_dataset:
        print(ele)
        break

    params = Params({
        'type': 'baseline_data_module',
        'reader': Params({
            'type': 'baseline_dataset',
            'encoder_model': 'xlm-roberta-base' 
            }),
        'lang': 'English',
        'batch_size': 2
    })
    dm = ConllDataModule.from_params(params=params)
    dm.setup('fit')
    for batch in dm.train_dataloader():
        print(batch)