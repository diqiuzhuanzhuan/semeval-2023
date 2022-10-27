# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from copy import deepcopy
import logging
from typing import Any, AnyStr, Union, Optional, overload
from torch.utils.data import Dataset
from allennlp.common.registrable import Registrable
from transformers import AutoTokenizer
import os, torch
import pytorch_lightning as pl
import ahocorasick
from intervaltree import IntervalTree, Interval
from task2.configuration import config
from allennlp.common.params import Params
from task2.data_man.meta_data import ConllItem, read_conll_item_from_file, get_id_by_type, get_type_by_id, get_id_to_labes_map
from task2.data_man.meta_data import _assign_ner_tags, extract_spans, get_wiki_knowledge


class ConllDataset(Dataset, Registrable):
    
    def __init__(
        self,
        encoder_model='bert-base-uncased'
        ) -> None:
        super().__init__()
        self.instances = []
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model, add_prefix_space=True)

    def __getitem__(self, index: Any) -> Any:
        if index >= self.__len__():
            raise IndexError('index value must be not more than the maximum length.')
        return self.encode_input(self.instances[index])

    @overload
    def encode_input(self, item) -> Any:
        ...

    def __len__(self) -> int:
        return len(self.instances)

    def read_data(self, conll_file: Union[AnyStr, os.PathLike]):
        self.instances = read_conll_item_from_file(conll_file)

        
@ConllDataset.register('baseline_dataset')
class BaselineDataset(ConllDataset):


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
        tag_len = len(input_ids)

        return id, input_ids, token_type_ids, attention_mask, token_masks, tag_len, label_ids, gold_spans

        
        
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
            pass

        if stage == 'test':
            pass
            
        if stage == 'predict':
            pass

        self.stage = stage


    def collate_batch(self, batch):
        batch_size = len(batch)
        batch_ = list(zip(*batch))
        id, input_ids, token_type_ids, attention_mask, token_masks, tag_lens, label_ids, gold_spans = batch_
        max_len = max([len(_) for _ in input_ids])
        input_ids_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        token_type_ids_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        attention_mask_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        if label_ids:
            max_label_len = max([len(_) for _ in label_ids])
            label_ids_tensor = torch.empty(size=[batch_size, max_label_len], dtype=torch.long).fill_(-100)
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

        return id, input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, token_masks, tag_lens, label_ids_tensor, gold_spans

    def train_dataloader(self):
        self.reader.read_data(config.train_file[self.lang])
        train_reader = deepcopy(self.reader)
        return torch.utils.data.DataLoader(train_reader, batch_size=self.batch_size, collate_fn=self.collate_batch, shuffle=True, num_workers=8)

    def val_dataloader(self):
        self.reader.read_data(config.validate_file[self.lang])
        val_reader = deepcopy(self.reader)
        return torch.utils.data.DataLoader(val_reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)

    def test_dataloader(self):
        self.reader.read_data(config.test_file[self.lang])
        test_reader = deepcopy(self.reader)
        return torch.utils.data.DataLoader(test_reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)
    
    def predict_dataloader(self):
        self.reader.read_data(config.test_file[self.lang])
        test_reader = deepcopy(self.reader)
        return torch.utils.data.DataLoader(test_reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)


@ConllDataset.register('dictionary_fused_dataset')        
class DictionaryFusedDataset(ConllDataset):
    entity_vocab = None

    def __init__(
        self, 
        encoder_model='bert-base-uncased'
        ) -> None:
        super().__init__(encoder_model)
        self.entity_vocab = get_wiki_knowledge(config.wikigaz_file)
        self._make_entity_automation()
        
    def _make_entity_automation(self):
        logging.info('start to build automation with all external entities')
        self.entity_automation = ahocorasick.Automaton()
        tmp = dict()
        for k in self.entity_vocab:
            self.entity_automation.add_word(k.lower(), (self.entity_vocab[k], k.lower()))
            tmp[k.lower()] = self.entity_vocab[k]
        for k in tmp:
            self.entity_vocab[k] = tmp[k]
        self.entity_automation.make_automaton()
        logging.info('automation is built successfully')

    def _search_entity(self, sentence: AnyStr):
        ans = []
        words = set(sentence.split(" "))
        tree = IntervalTree()
        entity_by_pos = dict()

        for end_index, (insert_order, original_value) in self.entity_automation.iter(sentence):
            start_index = end_index - len(original_value) + 1
            if start_index >= 1 and sentence[start_index-1] != " ":
                continue
            if end_index < len(sentence) - 1 and sentence[end_index+1] != " ":
                continue
            tree.remove_envelop(start_index, end_index)
            should_continue = False
            for item in tree.items():
                if start_index >= item.begin and end_index <= item.end:
                    should_continue = True
                    continue
            if should_continue:
                continue
            if original_value.count(" ") > 0:
                tree.add(Interval(start_index, end_index)) 
            elif original_value in words:
                if len(original_value) > 1:
                    tree.add(Interval(start_index, end_index)) 
        for interval in sorted(tree.items()):
            entity = sentence[interval.begin: interval.end+1]
            token_pos = (sentence[:interval.begin].count(' '), sentence[:interval.begin].count(' ') + entity.count(' '))
            for i in range(token_pos[0], token_pos[1]+1):
                entity_by_pos[i] = entity
            ans.append(entity)
          
        return ans, entity_by_pos


    def encode_input(self, item) -> Any:
        id, tokens, labels = item.id, item.tokens, item.labels
        sentence = " ".join(tokens)
        token_masks, new_labels, input_ids, token_type_ids, attention_mask, label_ids = [], [], [], [], [], []
        entities, entity_by_pos = self._search_entity(sentence=sentence)
        # half top
        input_ids.append(self.tokenizer.cls_token_id)
        if labels is not None:
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
        attention_mask.append(1)
        token_type_ids.append(0)
        token_masks.append(False)
        if labels is not None:
            gold_spans = extract_spans([get_type_by_id(label_id) for label_id in label_ids])
            label_ids.append(get_id_by_type('O'))

        # half bottom
        entity_information = "$".join([entity + '(' + self.entity_vocab[entity] + ')' for entity in entities])
        outputs = self.tokenizer(entity_information.lower())
        input_ids.extend(outputs['input_ids'][1:-1])
        attention_mask.extend(outputs['attention_mask'][1:-1])
        token_type_ids.extend([1] * len(outputs['input_ids'][1:-1]))
        if labels is not None:
            label_ids.extend([-100 for _ in outputs['input_ids'][1:-1]])
        
        # the last [SEP]]
        input_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        token_type_ids.append(1)
        if labels is not None:
            label_ids.append(-100)
        tag_len = len(input_ids)

        return id, input_ids, token_type_ids, attention_mask, token_masks, tag_len, label_ids, gold_spans


@ConllDataset.register('span_aware_dataset')
class SpanAwareDataset(DictionaryFusedDataset):
    def __init__(
        self, 
        encoder_model='bert-base-uncased'
        ) -> None:
        super().__init__(encoder_model)

    def encode_input(self, item) -> Any:
        id, tokens, labels = item.id, item.tokens, item.labels
        sentence = " ".join(tokens)
        token_masks, new_labels, input_ids, token_type_ids, attention_mask, label_ids = [], [], [], [], [], []
        entities = self._search_entity(sentence=sentence)
        
        # half top

        return super().encode_input(item)
        

if __name__ == '__main__':
    params = Params({
        'type': 'baseline_dataset',
        'encoder_model': 'xlm-roberta-base' 
    })
    baseline_dataset = ConllDataset.from_params(params)
    baseline_dataset.read_data(config.train_file['English'])
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
        break
    
    params = Params({
        'type': 'dictionary_fused_dataset',
        'encoder_model': 'xlm-roberta-base' 
        }
    )
    dic_fused_dataset = ConllDataset.from_params(params=params)
    dic_fused_dataset.read_data(config.train_file['English'])
    for item in dic_fused_dataset:
        print(item)
