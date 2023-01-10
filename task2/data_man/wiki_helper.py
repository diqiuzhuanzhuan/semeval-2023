# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import time
from typing import AnyStr, Dict
from task2.configuration.config import logging
from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
import time
from qwikidata.utils import dump_entities_to_json
from task2.configuration import config
from task2.data_man.meta_data import extract_spans, join_tokens, write_json_gzip, read_conll_item_from_file
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

person_vocab = collections.defaultdict(set)
type_by_id = collections.defaultdict(set)

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
            types = has_occupation_of_what(entity, type_by_id)
            if types:
                labels = entity._entity_dict.get('labels')
                aliases = entity._entity_dict.get('aliases')
                for lan in config.ios_639_1_code:
                    value = labels.get(lan, dict()).get('value', '')
                    person_vocab[value].add(types) if value else None
                    for dic in aliases.get(lan, []):
                        person_vocab[dic['value']].add(types) if dic['value'] else None
            
        if ii % 10000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(
                "found {} person among {} entities [entities/s: {:.2f}]".format(
                    len(person_vocab), ii, ii / dt
                )
            )
            break
    return person_vocab

def enumerate_group(wjd, type_by_id):
    t1 = time.time()
    group_vocab = collections.defaultdict(set)
    for ii, entity_dict in enumerate(wjd):
        
        if entity_dict["type"] == "item":
            entity = WikidataItem(entity_dict)
            qid = entity._entity_dict.get('id')
            if qid in type_by_id:
                types = [type_by_id[qid]]
                labels = entity._entity_dict.get('labels')
                aliases = entity._entity_dict.get('aliases')
                for lan in config.ios_639_1_code:
                    value = labels.get(lan, dict()).get('value', '')
                    group_vocab[value].update(set(types)) if value else None
                    for dic in aliases.get(lan, []):
                        group_vocab[dic['value']].update(set(types)) if dic['value'] else None
            
        if ii % 10000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(
                "found {} group among {} entities [entities/s: {:.2f}]".format(
                    len(group_vocab), ii, ii / dt
                )
            )
    return group_vocab


def find_person(conll_file: AnyStr) -> Dict:
    conll_items = read_conll_item_from_file(conll_file)
    type_by_person = dict()
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
            type_by_person[entity] = label
    return type_by_person

def main(wjd: WikidataJsonDump):
    type_by_person = dict()
    for lang in config.code_by_lang:
        type_by_person.update(find_person(config.train_file[lang]))
        type_by_person.update(find_person(config.validate_file[lang]))
    person_vocab = enumerate_person(wjd)
    #write_json_gzip('person.bak.gz', person_vocab)
    has_query_ids = set()
    type_by_occupation_id = dict()
    for person in type_by_person: 
        if person not in person_vocab:
            continue
        occupation_ids = person_vocab[person]
        for id in occupation_ids:
            if id in has_query_ids:
                continue
            qids = occupation_query(id)
            has_query_ids.add(id)
            for qid in qids:
                type_by_occupation_id[qid] = type_by_person[person]
    for person in person_vocab:
        person_vocab[person] = [type_by_occupation_id[id] for id in person_vocab[person]]
        if not person_vocab[person]:
            person_vocab[person] = ['Other-PER']
    
    write_json_gzip('person_entities.json.gzip', person_vocab)
        
    
        



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
    wjd_dump_path = 'E:\wikidata-20220103-all.json.gz'
    #wjd_dump_path = '/Users/malong/Downloads/humans.ndjson'
    wjd = WikidataJsonDump(wjd_dump_path)
    main(wjd=wjd)