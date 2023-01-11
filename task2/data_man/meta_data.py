# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from dataclasses import dataclass
import itertools
import json
import unicodedata
import pandas as pd
from task2.configuration.config import logging
from pathlib import Path
from typing import AnyStr, Dict, List, Union
import gzip, os
import zipfile
from task2.configuration import config


#LABEL_NAME = ['B-PER', 'I-PER', 'B-CW', 'I-CW', 'B-PROD', 'I-PROD', 'B-CORP', 'I-CORP', 'B-GRP', 'I-GRP', 'B-LOC', 'I-LOC', 'O']
LABEL_NAME = [
    'B-AerospaceManufacturer', 'B-AnatomicalStructure', 'B-ArtWork', 'B-Artist', 'B-Athlete', 'B-CarManufacturer', 'B-Cleric', 'B-Clothing',
    'B-Disease', 'B-Drink', 'B-Facility', 'B-Food', 'B-HumanSettlement', 'B-MedicalProcedure', 'B-Medication/Vaccine', 'B-MusicalGRP',
    'B-MusicalWork', 'B-ORG', 'B-OtherLOC', 'B-OtherPER', 'B-OtherPROD', 'B-Politician', 'B-PrivateCorp', 'B-PublicCorp',
    'B-Scientist', 'B-Software', 'B-SportsGRP', 'B-SportsManager', 'B-Station', 'B-Symptom', 'B-Vehicle', 'B-VisualWork',
    'B-WrittenWork', 'I-AerospaceManufacturer', 'I-AnatomicalStructure', 'I-ArtWork', 'I-Artist', 'I-Athlete', 'I-CarManufacturer',
    'I-Cleric', 'I-Clothing', 'I-Disease', 'I-Drink', 'I-Facility', 'I-Food', 'I-HumanSettlement', 'I-MedicalProcedure',
    'I-Medication/Vaccine', 'I-MusicalGRP', 'I-MusicalWork', 'I-ORG', 'I-OtherLOC', 'I-OtherPER', 'I-OtherPROD',
    'I-Politician', 'I-PrivateCorp', 'I-PublicCorp', 'I-Scientist', 'I-Software', 'I-SportsGRP', 'I-SportsManager', 'I-Station',
    'I-Symptom', 'I-Vehicle', 'I-VisualWork', 'I-WrittenWork', 'O'
    ]
    # 'OtherCw', 'OtherCorp', 'TechCorp' don't emerge

LABEL_BY_TOP_CATEGORY = {
    'Person': {'OtherPER', 'SportsManager', 'Cleric', 'Politician', 'Athlete', 'Artist', 'Scientist'},
    'Product': {'OtherPROD', 'Drink', 'Food', 'Vehicle', 'Clothing'},
    'Medical': {'Disease', 'Symptom', 'AnatomicalStructure', 'Medication/Vaccine', 'MedicalProcedure'},
    'Location': {'Facility', 'OtherLOC', 'HumanSettlement', 'Station'},
    'Creative Works': {'VisualWork', 'MusicalWork', 'WrittenWork', 'ArtWork', 'Software', 'OtherCW'},
    'Group': {'MusicalGRP', 'PublicCorp', 'PrivateCorp', 'OtherCorp', 'AerospaceManufacturer', 'SportsGRP', 'CarManufacturer', 'TechCorp', 'ORG'}
}

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


def read_conll_item_from_file(file: Union[AnyStr, bytes, os.PathLike], just_label=False):
    file = Path(file).as_posix()
    fin = gzip.open(file, 'rt', encoding='utf-8') if file.endswith('.gz') else open(file, 'rt', encoding='utf-8')
    ans = []
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '') for line in lines]

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        fields = [list(field) for field in zip(*fields)]
        if not just_label:
            ans.append(
                ConllItem.from_dict({
                    'id': metadata,
                    'tokens': fields[0],
                    'labels': fields[-1] if len(fields) == 4 else None
                    })
            )
        else:
            ans.append(
                ConllItem.from_dict({
                    'id': metadata,
                    'tokens': None,
                    'labels': fields[-1]
                    })
            )
    logging.info("read {} item from {}.".format(len(ans), str(file)))
    return ans

def write_conll_item_to_file(file: Union[AnyStr, bytes, os.PathLike], items: List[ConllItem], lang: AnyStr):
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir()
    fout = gzip.open(str(file), 'wt', encoding='utf-8') if str(file).endswith('.gz') else open(str(file), 'wt', encoding='utf-8')
    for item in items:
        lines = ["{}".format(item.id)]
        lines.extend([' _ _ '.join([t, l]) for t,l in zip(item.tokens, item.labels)])
        fout.write("\n".join(lines))
        fout.write("\n\n")
    fout.close()


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

human_readble_words_by_type = {
    "LOC": "location",
    "CORP": "corperation",
    "GRP": "group",
    "PER": "person",
    "CW": "creative work",
    "PROD": "product"
}

def get_human_readble_words_by_type(type: AnyStr):
    return human_readble_words_by_type[type]

def get_wiki_knowledge(file: Union[AnyStr, bytes, os.PathLike]):
    file = Path(file)
    entity_vocab = dict()
    with zipfile.ZipFile(file=str(file)) as myzip:
        with myzip.open(os.path.basename(str(file).strip(".zip"))) as f:
            for line in f:
                fields = line.decode("utf-8").strip("\n").strip("\r").split("\t")
                if len(fields) < 4:
                    fields = line.decode("utf-8").strip("\n").strip("\r").split(",")
                if len(fields) < 4:
                    continue
                entity, entity_type = fields[3].lower(), get_human_readble_words_by_type(fields[1])
                if entity in entity_vocab:
                    if not isinstance(entity_vocab[entity.lower()], str):
                        entity_vocab[entity.lower()] = entity_type
                    if entity_type not in entity_vocab[entity]:
                        entity_vocab[entity.lower()] = entity_vocab[entity.lower()] + "|" + entity_type
                else:
                    entity_vocab[entity.lower()] = entity_type
    logging.info('read wikigaz entity: {} '.format(len(entity_vocab)))
    return entity_vocab

def get_wiki_title_knowledge(file: Union[AnyStr, bytes, os.PathLike]) -> Dict:
    file = Path(file)
    entity_vocab = dict()
    data = pd.read_csv(str(file), delimiter='\t', compression='gzip')
    for entity in data['page_title']:
        if len(str(entity)) < 2:
            continue
        entity_vocab[str(entity)] = str(entity)
    logging.info('read wiki title eneity {}'.format(len(entity_vocab)))
    return entity_vocab

def get_wiki_title_google_type(file: Union[AnyStr, bytes, os.PathLike]) -> Dict:
    file = Path(file)
    if file.exists():
        with gzip.open(str(file), 'r') as f:
            data = json.loads(f.read().decode('utf-8'))
    else:
        data = dict()
    return data

def write_wiki_title_google_type(file: Union[AnyStr, bytes, os.PathLike], wiki_knowledge: Dict):
    file = Path(file)
    with gzip.open(str(file), 'w') as f:
        f.write(json.dumps(wiki_knowledge).encode('utf-8'))

def get_wiki_entities(file: Union[AnyStr, bytes, os.PathLike]) -> Dict[AnyStr, List]:
    data = read_json_gzip(file)
    new = dict()
    for k in data:
        new[k.lower()] = data[k]
    logging.info('get wiki entities {}'.format(len(new)))
    return new

def write_json_gzip(file: Union[AnyStr, bytes, os.PathLike], json_dict: Dict):
    file = Path(file)
    with gzip.open(str(file), 'w') as f:
        f.write(json.dumps(json_dict, ensure_ascii=True).encode('utf-8'))

def read_json_gzip(file: Union[AnyStr, bytes, os.PathLike]):
    file = Path(file)
    with gzip.open(str(file), 'r') as f:
        data = json.load(f)
    return data

def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if '\u4e00' <= cp <= '\u9fa5':
        return True
    return False

    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def join_tokens(tokens: List[AnyStr]) -> AnyStr:
    sentence = ""
    state = 0
    token_begin_index_by_sentence_pos = dict()
    token_end_index_by_sentence_pos = dict()
    for i, token in enumerate(tokens):
        if state == 0:
            sentence += token
            state = 1
        elif state == 1:
            if is_chinese_char(token) or (len(token) == 1 and unicodedata.category(token).startswith("P")):
                sentence += token
            else:
                sentence += " " + token
                state = 2
        elif state == 2:
            if is_chinese_char(token) or (len(token) == 1 and unicodedata.category(token).startswith("P")):
                sentence += token
                state = 1
            else:
                sentence += " " + token
        # [begin, end)
        token_begin_index_by_sentence_pos[len(sentence)-len(token)] = i
        token_end_index_by_sentence_pos[len(sentence)] = i
    
    return sentence, token_begin_index_by_sentence_pos, token_end_index_by_sentence_pos


########## for auxiliary classifier #################
@dataclass
class DescriptionTypeItem:
    text: AnyStr
    label: AnyStr

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return DescriptionTypeItem(
            text = json_dict['text'],
            label = json_dict['label']
        )
        

def generate_description_type_item():
    pass
    

if __name__ == '__main__':
    for conll_item in read_conll_item_from_file('./task2/data/semeval_2021_task_11_trial_data.txt'):
        print(conll_item)
        break

    #entity_vocab = get_wiki_knowledge(config.wikigaz_file)
    entity_vocab = get_wiki_entities(config.wiki_data['English'])
    print(len(entity_vocab))