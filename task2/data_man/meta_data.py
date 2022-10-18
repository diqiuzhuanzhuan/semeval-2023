# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import AnyStr, Dict, List, Union
import gzip, os


LABEL_NAME = ['B-PER', 'I-PER', 'B-CW', 'I-CW', 'B-PROD', 'I-PROD', 'B-CORP', 'I-CORP', 'B-GRP', 'I-GRP', 'B-LOC', 'I-LOC', 'O']

return_map = dict()
for i, ele in enumerate(LABEL_NAME):
    return_map[ele] = i

def get_type_by_id(id):
    if id >= len(LABEL_NAME) or id < 0:
        raise ValueError('id should not more than {}'.format(len(LABEL_NAME)-1))
    return LABEL_NAME[id]

def get_id_by_type(type):
    return return_map[type]


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


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-":# or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False


def read_conll_item_from_file(file: Union[AnyStr, bytes, os.PathLike]):
    file = Path(file).as_posix()
    fin = gzip.open(file, 'rt') if file.endswith('.gz') else open(file, 'rt')
    ans = []
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '') for line in lines]

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        fields = [list(field) for field in zip(*fields)]
        ans.append(
            ConllItem.from_dict({
                'id': metadata,
                'tokens': fields[0],
                'labels': fields[-1] if len(fields) == 4 else None
                })
        )
    return ans

if __name__ == '__main__':
    for conll_item in read_conll_item_from_file('./task2/data/semeval_2021_task_11_trial_data.txt'):
        print(conll_item)