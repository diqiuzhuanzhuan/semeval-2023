# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from dataclasses import dataclass
from typing import AnyStr, Dict, List


LABEL_NAME = ['B-PER', 'I-PER', 'B-CW', 'I-CW', 'B-PROD', 'I-PROD', 'B-CORP', 'I-CORP', 'B-LOC', 'I-LOC', 'O']

def get_id_to_type():
    return_map = dict()
    for i, ele in enumerate(LABEL_NAME):
        return_map[i] = ele
    return return_map

def get_type_by_id(id):
    if id >= len(LABEL_NAME) or id < 0:
        raise ValueError('id should not more than {}'.format(len(LABEL_NAME)-1))
    return LABEL_NAME[id]


@dataclass
class ConllItem:
    id: AnyStr
    tokens: List[AnyStr]
    labels: List[AnyStr]

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return ConllItem(
            id = json_dict['id'],
            tokens = json_dict['tokens'],
            labels = json_dict['labels']
        )