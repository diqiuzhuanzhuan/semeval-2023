# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from copy import deepcopy
import itertools, collections
from typing import Any, AnyStr, Dict, List, Tuple, Union, Optional, overload
from torch.utils.data import Dataset
from allennlp.common.registrable import Registrable
from transformers import AutoTokenizer
import os, torch
import numpy as np
import pytorch_lightning as pl
import ahocorasick
from intervaltree import IntervalTree, Interval
from task2.configuration import config
from task2.configuration.config import logging
from allennlp.common.params import Params
from task2.data_man.meta_data import ConllItem, read_conll_item_from_file, get_id_by_type, get_type_by_id, get_id_to_labes_map
from task2.data_man.meta_data import _assign_ner_tags, extract_spans, get_wiki_knowledge, join_tokens, get_wiki_title_knowledge, get_wiki_title_google_type, get_wiki_entities

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class ConllDataset(Dataset, Registrable):
    
    def __init__(
        self,
        encoder_model='bert-base-uncased',
        lang: AnyStr='English'
        ) -> None:
        super().__init__()
        self.instances = []
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model, add_prefix_space=True)
        self.lang = lang

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
        train_file: AnyStr,
        val_file: AnyStr,
        lang: AnyStr='English',
        batch_size: int=16
        ) -> None:
        super().__init__()
        self.reader = reader
        self.batch_size = batch_size
        self.lang = lang
        self.train_file = train_file
        self.val_file = val_file

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
        if len(np.shape(attention_mask[0])) == 2:
            attention_mask_tensor = torch.empty(size=[batch_size, max_len, max_len], dtype=torch.long).fill_(0)
        else:
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
            if len(np.shape(attention_mask[0])) == 2:
                attention_mask_tensor[i][0:available_length, 0:available_length] = torch.tensor(attention_mask[i], dtype=torch.long)
            else:
                attention_mask_tensor[i][0:available_length] = torch.tensor(attention_mask[i], dtype=torch.long)
            if label_ids_tensor is not None:
                label_ids_length = len(label_ids[i])
                label_ids_tensor[i][:label_ids_length] = torch.tensor(label_ids[i], dtype=torch.long)

        return id, input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, token_masks, tag_lens, label_ids_tensor, gold_spans

    def train_dataloader(self):
        self.reader.read_data(self.train_file)
        train_reader = deepcopy(self.reader)
        return torch.utils.data.DataLoader(train_reader, batch_size=self.batch_size, collate_fn=self.collate_batch, shuffle=True, num_workers=8)

    def val_dataloader(self):
        self.reader.read_data(self.val_file)
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
    all_types = set()

    def __init__(
        self, 
        encoder_model='bert-base-uncased',
        lang: AnyStr='English'
        ) -> None:
        super().__init__(encoder_model, lang)
        #self.entity_vocab = get_wiki_knowledge(config.wikigaz_file)
        #self.entity_vocab = get_wiki_title_google_type(config.wiki_title_with_google_type_file[lang])
        self.entity_vocab = collections.defaultdict(list)
        for k in config.wiki_entity_data:
            vocab = get_wiki_entities(config.wiki_entity_data[k])
            for entity in vocab:
                self.entity_vocab[entity].extend(vocab[entity])
                self.all_types.update(set(vocab[entity]))

        self.tokenizer.add_tokens(list(self.all_types))
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

    def _search_entity(self, tokens: List[AnyStr]) -> Tuple[List[AnyStr], Dict]:
        ans = []
        tree = IntervalTree()
        entity_by_pos = dict()
        sentence, token_begin_index_by_sentence_pos, token_end_index_by_sentence_pos = join_tokens(tokens=tokens)

        for end_index, (insert_order, original_value) in self.entity_automation.iter(sentence):
            start_index = end_index - len(original_value) + 1
            end_index = end_index + 1 #make [start_index, end_index) ---> so original == sentence[start_index: end_index]
            start_index = end_index - len(original_value)

            if start_index not in token_begin_index_by_sentence_pos:
                continue
            if end_index not in token_end_index_by_sentence_pos:
                continue
            tree.remove_envelop(start_index, end_index)
            should_not_add = False
            for item in tree.items():
                if start_index >= item.begin and end_index <= item.end:
                    should_not_add = True
                    continue
            if should_not_add:
                continue
            for item in tree.overlap(start_index, end_index):
                if (item.end - item.begin) >= (end_index - start_index):
                    should_not_add = True
                else:
                    tree.remove(item)
            if should_not_add:
                continue
            tree.add(Interval(start_index, end_index)) 
            
        for interval in sorted(tree.items()):
            entity = sentence[interval.begin: interval.end]
            token_pos = (token_begin_index_by_sentence_pos[interval.begin], token_end_index_by_sentence_pos[interval.end])
            for i in range(token_pos[0], token_pos[1]):
                entity_by_pos[i] = entity
            if entity not in ans:
                ans.append(entity)
        return ans, entity_by_pos

    def get_entity_type(self, entity: AnyStr):
        #entity_type = self.entity_vocab[entity].get('type', [])
        entity_type = self.entity_vocab.get(entity, [])
        if entity_type is None:
            entity_type = []
        return '|'.join(entity_type)

    def encode_input(self, item) -> Any:
        id, tokens, labels = item.id, item.tokens, item.labels
        token_masks, input_ids, token_type_ids, attention_mask, label_ids = [], [], [], [], []
        entities, _ = self._search_entity(tokens)
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

        tag_len = len(input_ids) # only half top need to predict labels

        # half bottom
        entity_information = "$".join([entity + '(' + self.get_entity_type(entity) + ')' for entity in entities if self.get_entity_type(entity)])
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

        return id, input_ids, token_type_ids, attention_mask, token_masks, tag_len, label_ids, gold_spans


@ConllDataset.register('span_aware_dataset')
class SpanAwareDataset(DictionaryFusedDataset):
    def __init__(
        self, 
        encoder_model='bert-base-uncased',
        lang='English'
        ) -> None:
        super().__init__(encoder_model, lang)

    def calc_mask_value(self, entity: AnyStr, entities: List[AnyStr]):
        if entity in entities:
            return entities.index(entity) + 2 # 0 and 1 have been used so we plus 2

    def encode_input(self, item) -> Any:
        id, tokens, labels = item.id, item.tokens, item.labels
        token_masks, new_labels, input_ids, token_type_ids, attention_mask, label_ids = [], [], [], [], [], []
        entities, entity_by_pos = self._search_entity(tokens=tokens)
        ### debug to see how many entities were in the dictionary
        '''
        gold_spans = extract_spans(item.labels)
        gold_entities = []
        gold_labels = []
        for k in gold_spans:
            if gold_spans[k] == 'O':
                continue
            gold_entities.append(join_tokens(tokens[k[0]:k[1]+1])[0])
            gold_labels.append(labels[k[0]][2:])
        #logging.info('{}'.format(join_tokens(tokens)[0]))    
        #logging.info('gold entities: {}'.format(gold_entities))
        #logging.info('gold labels: {}'.format(gold_labels))
        for i, entity in enumerate(gold_entities):
            if entity in entities:
                pass
            else:
                if gold_labels[i] == 'ORG':
                    logging.info('{}'.format(join_tokens(tokens)[0]))    
                    logging.info('{}: {} is not in'.format(entity, gold_labels[i]))
        '''
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
            if i in entity_by_pos:
                attention_mask.extend([self.calc_mask_value(entity_by_pos[i], entities)] * subtoken_len)
            else:
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

        tag_len = len(input_ids) # only half top need to predict labels

        for entity in entities:
            entity_type = self.get_entity_type(entity)
            if not entity_type:
                continue
            outputs = self.tokenizer(entity_type.lower())
            subtoken_len = len(outputs['input_ids']) - 2
            input_ids.extend(outputs['input_ids'][1:-1])
            attention_mask.extend([self.calc_mask_value(entity, entities)] * subtoken_len)
            token_type_ids.extend([1] * subtoken_len)
            if labels is not None:
                label_ids.extend([-100] * subtoken_len)

        def _nxor(a, b):
            if a == b:
                return 1
            else:
                return 0
        span_aware_attention_mask = [_nxor(i, j) for i, j in itertools.product(attention_mask, attention_mask)]
        span_aware_attention_mask = np.reshape(span_aware_attention_mask, newshape=[len(attention_mask), len(attention_mask)])
        span_aware_attention_mask[:tag_len, :tag_len] = 1

        return id, input_ids, token_type_ids, span_aware_attention_mask, token_masks, tag_len, label_ids, gold_spans

@ConllDataset.register('gemnet_fused_dataset')
class GemnetFusedDataset(DictionaryFusedDataset):

    def __init__(
        self, 
        encoder_model='bert-base-uncased',
        lang='English'
        ) -> None:
        super().__init__(encoder_model, lang)
        self.gemnet_code = [0] * len(get_id_to_labes_map)

    def encode_input(self, item) -> Any:
        id, tokens, labels = item.id, item.tokens, item.labels
        token_masks, new_labels, input_ids, token_type_ids, attention_mask, label_ids = [], [], [], [], [], []
        entities, entity_by_pos = self._search_entity(tokens=tokens)
        id, tokens, labels = item.id, item.tokens, item.labels
        token_masks, input_ids, token_type_ids, attention_mask, label_ids = [], [], [], [], []
        entities, _ = self._search_entity(tokens)
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

        tag_len = len(input_ids) # only half top need to predict labels

        # half bottom
        entity_information = "$".join([entity + '(' + self.get_entity_type(entity) + ')' for entity in entities if self.get_entity_type(entity)])
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

        return id, input_ids, token_type_ids, attention_mask, token_masks, tag_len, label_ids, gold_spans
         

if __name__ == '__main__':
    """
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
            'type': 'span_aware_dataset',
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
        break
    """

    params = Params({
        'type': 'span_aware_dataset',
        'encoder_model': 'xlm-roberta-base' 
        }
    )
    params = Params({
        'type': 'baseline_data_module',
        'reader': Params({
            'type': 'span_aware_dataset',
            'encoder_model': 'xlm-roberta-base' ,
            'lang': 'Chinese'
            }),
        'lang': 'Chinese',
        'train_file': config.train_file['Chinese'],
        'val_file': config.validate_file['Chinese'],
        'batch_size': 2
    })
    dm = ConllDataModule.from_params(params=params)
    for batch in dm.train_dataloader():
        pass