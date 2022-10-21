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
    if not is_id_legal(id):
        raise ValueError('id should not more than {}'.format(len(LABEL_NAME)-1))
    return LABEL_NAME[id]

def get_id_by_type(type):
    return return_map[type]

def get_id_to_labes_map():
    return {id: label for id, label in enumerate(LABEL_NAME)}

def is_id_legal(id):
    if id >= len(LABEL_NAME) or id < 0:
        return False
    return True

def get_num_labels():
    return len(LABEL_NAME)


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


def _assign_ner_tags(ner_tag, rep_):
    '''
    Changing the token_masks so that only the first sub_word of a token has a True value, while the rest is False. This will be used for storing the predictions.
    :param ner_tag:
    :param rep_:
    :return:
    '''
    ner_tags_rep = []

    sub_token_len = len(rep_)
    mask_ = [False] * sub_token_len

    if len(mask_):
        mask_[0] = True

    if ner_tag[0] == 'B':
        in_tag = 'I' + ner_tag[1:]

        ner_tags_rep.append(ner_tag)
        ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
    else:
        ner_tags_rep.extend([ner_tag] * sub_token_len)
    return ner_tags_rep, mask_


def extract_spans(tags):
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    for _id, nt in enumerate(tags):
        indicator = nt[0]
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            # do nothing
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans

if __name__ == '__main__':
    for conll_item in read_conll_item_from_file('./task2/data/semeval_2021_task_11_trial_data.txt'):
        print(conll_item)