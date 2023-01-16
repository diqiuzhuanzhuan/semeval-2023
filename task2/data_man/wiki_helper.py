# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import time
from typing import AnyStr, Dict
from task2.configuration.config import logging
from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
from pathlib import Path
import time
from qwikidata.utils import dump_entities_to_json
from task2.configuration import config
from task2.data_man.meta_data import extract_spans, join_tokens, write_json_gzip, read_conll_item_from_file, LABEL_BY_TOP_CATEGORY, DEFAULT_LABEL_BY_TOP_CAGEGORY, read_json_gzip
import collections
import pandas as pd
import wikidata_plain_sparql as wikidata
#for t in ['artist', 'politician', 'cleric', 'scientist', 'athlete', 'sportsmanager']:
#    for k in pd.read_csv('{}.tsv'.format(t), header=None).values.flatten().tolist():
#        PERSON_TYPE[k] = t
import sys
from qwikidata.sparql import get_subclasses_of_item
from SPARQLWrapper import SPARQLWrapper, JSON, TSV

endpoint_url = "https://query.wikidata.org/sparql"


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
        
P_OCCUPATION = "P106"
PERSON_TYPE = {
    "Q82955": 'politician',
    "Q2259532": 'cleric',
    "Q901": 'scientist',
    "Q81096": 'scientist',
    "Q483501": 'artist',
    "Q2066131": 'athlete',
    'Q838476': 'sportsmanager', #体育总监
    'Q41583': 'sportsmanager', #教练
}


GROUP_TYPE = {
    'Q4438121': 'sport organization',  
    'Q936518': 'aerospace manufacturer',
    'Q215380': 'musical group',
    'Q32178211': 'musical group',
    'Q891723': 'public corporation',
    'Q7257717': 'public corporation',
    'Q5621421': 'private corparation',
    'Q786820': 'automobile manufacturer',
    'Q18388277': 'technology company',
    'Q43229':'orginazation',
    'Q4830453':'orginazation',
}

def query_person(qid: AnyStr, type_by_id: Dict):
    """_summary_

    Args:
        qid (AnyStr): a string identifier from wiki
        type_by_id (Dict): a dict indicating which occupation type an id belongs to
    """
    query = 'SELECT DISTINCT ?item WHERE {' + \
            '?item p:P106 ?statement0. ' + \
            '?statement0 (ps:P106/(wdt:P279*)) wd:'+ qid +'.' + \
            '}' 
    data = get_results(endpoint_url=endpoint_url, query=query)
    ids = [result['item']['value'].replace('http://www.wikidata.org/entity/', '') for result in data["results"]["bindings"]]
    for id in ids:
        type_by_id[id] = PERSON_TYPE[qid]
    return type_by_id

def group_query(qid: AnyStr, type_by_id: Dict):
    """_summary_

    Args:
        qid (AnyStr): a string identifier from wiki
        type_by_id (Dict): a dict indicating which group type an id belongs to
    """
    query = 'SELECT ?item WHERE ' + \
        '{ ?item wdt:P31/wdt:P279* wd:' + qid + '.}'
    data = get_results(endpoint_url=endpoint_url, query=query)
    ids = [result['item']['value'].replace('http://www.wikidata.org/entity/', '') for result in data["results"]["bindings"]]
    for id in ids:
        if id in type_by_id:
            if GROUP_TYPE[qid] != 'orginazation':
                type_by_id[id] = GROUP_TYPE[qid]
        else:
            type_by_id[id] = GROUP_TYPE[qid]
             
    return type_by_id

def occupation_query(qid: AnyStr):
    """_summary_

    Args:
        qid (_type_): _description_
        type_by_id (Dict): a dict indicating which occupation type an id belongs to

    Returns:
        _type_: _description_
    """
    qids = get_subclasses_of_item(qid)
    return qids

def get_all_id_by_type(type='GROUP'):
    type_by_id = collections.defaultdict(set)
    if type == 'GROUP':
        for k in GROUP_TYPE:
            logging.info(k)
            group_query(k, type_by_id)
    elif type == 'PERSON':
        for k in PERSON_TYPE:
            logging.info(k)
            occupation_query(k, type_by_id)
    else:
        pass
    logging.info(len(type_by_id))
    return type_by_id

def has_occupation_of_what(item: WikidataItem, truthy: bool = True) -> set:
    """Return True if the Wikidata Item has occupation."""
    if truthy:
        claim_group = item.get_truthy_claim_group(P_OCCUPATION)
    else:
        claim_group = item.get_claim_group(P_OCCUPATION)

    occupation_qids = [
        claim.mainsnak.datavalue.value["id"]
        for claim in claim_group
        if claim.mainsnak.snaktype == "value"
    ]
    if len(occupation_qids) == 0:
        return []
    # we just get the occupation with the highest priority
    return occupation_qids[0]



# create an iterable of WikidataItem representing politicians


def enumerate_person(wjd):
    t1 = time.time()
    person_vocab = collections.defaultdict(set)
    for ii, entity_dict in enumerate(wjd):
        
        if entity_dict["type"] == "item":
            entity = WikidataItem(entity_dict)
            types = has_occupation_of_what(entity)
            if types:
                labels = entity._entity_dict.get('labels')
                aliases = entity._entity_dict.get('aliases')
                for lan in config.ios_639_1_code:
                    description = entity.get_description(lan) or ""
                    value = labels.get(lan, dict()).get('value', '')
                    person_vocab[value].add(description) if value else None
                    for dic in aliases.get(lan, []):
                        person_vocab[dic['value']].add(description) if dic['value'] else None
            
        if ii % 10000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(
                "found {} person among {} entities [entities/s: {:.2f}]".format(
                    len(person_vocab), ii, ii / dt
                )
            )
    return person_vocab

def enumerate_item(wjd, cared_class_ids):
    t1 = time.time()
    class_id_by_entity = collections.defaultdict(set)
    for ii, entity_dict in enumerate(wjd):
        
        if entity_dict["type"] == "item":
            entity = WikidataItem(entity_dict)
            qid = entity._entity_dict.get('id')
            try:
                is_instance_of_class_id = entity._entity_dict['claims']['P31'][0]['mainsnak']['datavalue']['value']['id']
            except Exception as e:
                logging.error(e)
                continue
            if is_instance_of_class_id not in cared_class_ids:
                continue

            labels = entity._entity_dict.get('labels')
            aliases = entity._entity_dict.get('aliases')
            for lan in config.ios_639_1_code:
                value = labels.get(lan, dict()).get('value', '')
                class_id_by_entity[value].update(set([is_instance_of_class_id])) if value else None
                for dic in aliases.get(lan, []):
                    class_id_by_entity[value].update(set([is_instance_of_class_id])) if value else None
                    class_id_by_entity[dic['value']].update(set([is_instance_of_class_id])) if dic['value'] else None
            
        if ii % 10000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(
                "found {}  among {} entities [entities/s: {:.2f}]".format(
                    len(class_id_by_entity), ii, ii / dt
                )
            )
    return class_id_by_entity


def find_by_top_category(conll_file: AnyStr, category: AnyStr) -> Dict:
    conll_items = read_conll_item_from_file(conll_file)
    type_by_entity = dict()
    for item in conll_items:
        gold_spans = extract_spans(item.labels)
        gold_entities = []
        gold_labels = []
        for k in gold_spans:
            if gold_spans[k] == 'O':
                continue
            gold_entities.append(join_tokens(item.tokens[k[0]:k[1]+1])[0])
            gold_labels.append(item.labels[k[0]][2:])
        #logging.info('{}'.format(join_tokens(tokens)[0]))    
        #logging.info('gold entities: {}'.format(gold_entities))
        #logging.info('gold labels: {}'.format(gold_labels))
        for i, entity in enumerate(gold_entities):
            label = gold_labels[i]
            if label in LABEL_BY_TOP_CATEGORY[category]:
                type_by_entity[entity] = label
    return type_by_entity

def main(wjd: WikidataJsonDump):
    type_by_person = dict()
    for lang in config.code_by_lang:
        type_by_person.update(find_by_top_category(config.train_file[lang], 'Person'))
        type_by_person.update(find_by_top_category(config.validate_file[lang], 'Person'))
    if Path('person.bak.gz').exists():
        person_vocab = read_json_gzip('person.bak.gz')
    else:
        person_vocab = enumerate_person(wjd)
        new_person_vocab = dict()
        for k in person_vocab:
            new_person_vocab[k] = list(person_vocab[k])
        write_json_gzip('person.bak.gz', new_person_vocab)
    type_by_occupation_id = collections.defaultdict(set)
    for person in type_by_person: 
        if person not in person_vocab:
            continue
        occupation_ids = person_vocab[person]
        for id in occupation_ids:
            if id in type_by_occupation_id:
                continue
            qids = occupation_query(id)
            time.sleep(0.5)
            for qid in qids:
                if qid == 'Q177220':
                    print(qid)
                if qid in type_by_occupation_id and type_by_person[person] not in type_by_occupation_id[qid]:
                    logging.error("{}-{}-{}-{}--{}".format(id, qid, person, type_by_occupation_id[qid], type_by_person[person]))
                type_by_occupation_id[qid].add(type_by_person[person])
    for person in person_vocab:
        import itertools
        person_vocab[person] = list(set(list(itertools.chain(*[list(type_by_occupation_id[id]) for id in person_vocab[person] if id in type_by_occupation_id]))))
        if not person_vocab[person]:
            person_vocab[person] = ['OtherPER']
    
    write_json_gzip('person_entities.json.gz', person_vocab)
        
    
def main_category(wjd: WikidataJsonDump, category: AnyStr):
    type_by_entity = dict()
    for lang in config.code_by_lang:
        type_by_entity.update(find_by_top_category(config.train_file[lang], category))
        type_by_entity.update(find_by_top_category(config.validate_file[lang], category))
    
    if category == 'Group':
        # the class id of the highest level
        qids = get_subclasses_of_item('Q43229')
        class_id_by_entity = enumerate_item(wjd, set(qids))
    new_vocab = dict()
    for k in class_id_by_entity:
        new_vocab[k] = list(class_id_by_entity[k])
    write_json_gzip('{}.bak.gz'.format(category), new_vocab)
    del new_vocab
    type_by_class_id = dict()
    for entity in class_id_by_entity:
        if entity not in type_by_entity:
            continue
        class_ids = class_id_by_entity[entity]
        for class_id in class_ids:
            if class_id in type_by_class_id:
                continue
            qids = get_subclasses_of_item(class_id)
            time.sleep(0.5)
            for qid in qids:
                type_by_class_id[qid] = type_by_entity[entity]
                
    for entity in class_id_by_entity:
        class_id_by_entity[entity] = list(set([type_by_class_id[id] for id in class_id_by_entity[entity] if id in type_by_class_id]))
        if not class_id_by_entity[entity]:
            class_id_by_entity[entity] = DEFAULT_LABEL_BY_TOP_CAGEGORY[category]
    
    write_json_gzip(config.wiki_entity_data[str(category).lower()], class_id_by_entity)

    



# write the iterable of WikidataItem to disk as JSON
#out_fname = "filtered_entities.json"
#dump_entities_to_json(person, out_fname)
#wjd_filtered = WikidataJsonDump(out_fname)

# load filtered entities and create instances of WikidataItem
#for ii, entity_dict in enumerate(wjd_filtered):
#    item = WikidataItem(entity_dict)
#    print(item)

if __name__ == "__main__":
    # create an instance of WikidataJsonDump
    wjd_dump_path = '/Users/malong/Downloads/wikidata-20220103-all.json.gz'
    #wjd_dump_path = '/Users/malong/Downloads/humans.ndjson'
    wjd = WikidataJsonDump(wjd_dump_path)
    main(wjd=wjd)