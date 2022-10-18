# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from typing import Any, AnyStr, Union, Optional
from torch.utils.data import Dataset
from allennlp.common.registrable import Registrable
from transformers import AutoTokenizer
import os
from allennlp.common.params import Params
from task2.data_man.meta_data import ConllItem, read_conll_item_from_file, get_id_by_type, get_type_by_id



class ConllDataset(Dataset, Registrable):
    
    def __init__(
        self,
        encoder_model='bert-base-uncased'
        ) -> None:
        super().__init__()
        self.instances = []
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model, add_prefix_space=True)

    def __getitem__(self, index: Any) -> Any:
        raise NotImplemented('')

    def encode_input(self, item):
        raise NotImplemented('')

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
        new_tokens, new_labels, input_ids, attention_mask, label_ids = [], [], [], [], []
        if labels is not None:
            pass
        for token, tag in zip(tokens, labels):
            outputs = self.tokenizer(token.lower())
            subtoken_len = len(outputs['input_ids']) - 2
            input_ids.extend(outputs['input_ids'][1:-1])
            attention_mask.extend(outputs['attention_mask'][1:-1])
            sub_tags = [tag] + [tag.replace('B-', 'I-')] * (subtoken_len-1)
            label_ids.extend([get_id_by_type(sub_tag) for sub_tag in sub_tags])

        return id, input_ids, attention_mask, label_ids

        
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
