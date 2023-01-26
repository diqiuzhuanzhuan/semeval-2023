# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import time
from typing import AnyStr, Dict
import sys, hashlib
import urllib.request
import urllib.parse
import grequests
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

def build_params(query: AnyStr, fr: AnyStr, to: AnyStr):
    import random
    app_id = '20230117001533917'
    key = 'Uk_s8D1EEUOTTjtZzJGx'
    salt = str(random.randint(0, 1000))
    params = {
        'q': query,
        'from': fr,
        'to': to,
        'appid': app_id,
        'salt': salt,
        'sign': hashlib.md5((app_id+ query + salt + key).encode('utf-8')).hexdigest()
    }
    return params

def build_request(querys, frs: AnyStr, tos: AnyStr):
    service_url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    gets = []
    for query, fr, to in zip(querys, frs, tos):
        params = build_params(query=query, fr=fr, to=to)
        url = service_url + '?' + urllib.parse.urlencode(params)
        gets.append(grequests.get(url))
    return gets

def build_response(res_list):
    ans = []
    for i in range(len(res_list)):
        try:
            data = res_list[i].json()['trans_result'][0]['dst']
        except Exception as e:
            logging.info(res_list[0].json())
            data = None
        ans.append(data)
    return ans

def translate(query, fr, to):
    task_list = build_request([query], [fr], [to])
    res_list = grequests.map(task_list)
    ans = build_response(res_list)
    return ans[0]


SAVE_CACHE = read_json_gzip('saved_cache.json.gz')
# create an iterable of WikidataItem representing politicians
def enumerate_person(wjd):
    t1 = time.time()
    person_vocab = collections.defaultdict(dict)
    #for lan in config.ios_639_1_code:
    #    person_vocab[lan] = collections.defaultdict(set)
    for ii, entity_dict in enumerate(wjd):
        
        if entity_dict["type"] == "item":
            entity = WikidataItem(entity_dict)
            types = has_occupation_of_what(entity)
            if types:
                labels = entity._entity_dict.get('labels')
                aliases = entity._entity_dict.get('aliases')
                default_description = entity.get_description('en')
                for lan in config.ios_639_1_code:
                    description = entity.get_description(lan)
                    # translate
                    if default_description and not description and lan == 'zh':
                        to_lan = {
                            'fr': 'fra',
                            'fa': 'per',
                            'eu': 'spa',
                            'bn': 'ben',
                            'sv': 'swe',
                            'uk': 'ukr'
                        }
                        #SAVE_CACHE[default_description] = 0
                        if default_description in SAVE_CACHE and SAVE_CACHE[default_description]:
                            description = SAVE_CACHE[default_description]
                        #else:
                            #description = translate(default_description, 'en', to_lan.get(lan, lan))
                            #if description:
                            #    SAVE_CACHE[default_description] = description
                        break
                    if not description:
                        description = default_description
                    
                    value = labels.get(lan, dict()).get('value', '')
                    if value and value not in person_vocab:
                        person_vocab[value] = collections.defaultdict(set)
                    person_vocab[value][lan].add(description) if value else None
                    for dic in aliases.get(lan, []):
                        if not dic['value']:
                            continue
                        if dic['value'] not in person_vocab:
                            person_vocab[dic['value']] = collections.defaultdict(set)
                        person_vocab[dic['value']][lan].add(description)
            
        if ii % 10000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(
                "found {} person among {} entities [entities/s: {:.2f}]".format(
                    len(person_vocab), ii, ii / dt
                )
            )
    #write_json_gzip('save_cache.json.gz', SAVE_CACHE)
    return person_vocab

def enumerate_item(wjd, cared_class_ids):
    t1 = time.time()
    description_by_entity = collections.defaultdict(dict)
    for ii, entity_dict in enumerate(wjd):
        if ii % 10000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(
                "found {}  among {} entities [entities/s: {:.2f}]".format(
                    len(description_by_entity), ii, ii / dt
                )
            )
        
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
                description = entity.get_description(lan) or entity.get_description('en') or ''
                value = labels.get(lan, dict()).get('value', '')
                if value and value not in description_by_entity:
                    description_by_entity[value] = collections.defaultdict(set)
                description_by_entity[value][lan].add(description) if value else None
                for dic in aliases.get(lan, []):
                    if not dic['value']:
                        continue
                    if dic['value'] not in description_by_entity:
                        description_by_entity[dic['value']] = collections.defaultdict(set)
                    description_by_entity[dic['value']][lan].add(description)
            
    return description_by_entity


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
    if False and Path('person.bak.gz').exists():
        person_vocab = read_json_gzip('person.bak.gz')
    else:
        person_vocab = enumerate_person(wjd)
        new_person_vocab = dict()
        
        for k in person_vocab:
            new_person_vocab[k] = collections.defaultdict(set)
            for lan in config.ios_639_1_code:
                t = person_vocab[k][lan]
                if len(t) > 1 and '' in t:
                    t.remove('')
                new_person_vocab[k][lan] = list(t)
        write_json_gzip('person_description.json.gz', new_person_vocab)
    
def main_category(wjd: WikidataJsonDump, category: AnyStr):
    
    #if category == 'Group':
        # the class id of the highest level
    qids = get_subclasses_of_item('Q43229')
   #     class_id_by_entity = enumerate_item(wjd, set(qids))
    #elif category == 'Medical':
    qids += get_subclasses_of_item('Q12136') 
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q11190')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q796194')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q4936952')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q12140')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q134808')
    time.sleep(0.5)
   #     class_id_by_entity = enumerate_item(wjd, set(qids))
   # elif category == 'Creative Works':
    qids += get_subclasses_of_item('Q110910970')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q2188189')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q47461344')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q17537576')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q114511703')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q7397')
    time.sleep(0.5)
        #class_id_by_entity = enumerate_item(wjd, set(qids))
    #elif category == 'Product':
    qids += get_subclasses_of_item('Q11460')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q42889')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q2095')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q40050')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q2424752')
    time.sleep(0.5)
    #elif category == 'Location':
    qids += get_subclasses_of_item('Q13226383')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q486972')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q486972')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q12819564')
    time.sleep(0.5)
    qids += get_subclasses_of_item('Q115095765')

    class_id_by_entity = enumerate_item(wjd, set(qids))
    new_vocab = collections.defaultdict(dict)
    for k in class_id_by_entity:
        if k not in new_vocab:
            new_vocab[k] = collections.defaultdict(list)
        for lan in class_id_by_entity[k]:
            new_vocab[k][lan] = list(class_id_by_entity[k][lan])
    write_json_gzip('{}_description.json.gz'.format(category), new_vocab)
    del new_vocab
    return
# write the iterable of WikidataItem to disk as JSON
#out_fname = "filtered_entities.json"
#dump_entities_to_json(person, out_fname)
#wjd_filtered = WikidataJsonDump(out_fname)

# load filtered entities and create instances of WikidataItem
#for ii, entity_dict in enumerate(wjd_filtered):
#    item = WikidataItem(entity_dict)
#    print(item)

if __name__ == "__main__":
    #task_list = build_request(['hello'], 'en', 'zh')
    #res_list = grequests.map(task_list)
    #print(res_list)
    #data = read_json_gzip('person_entities.json.gz')
    #for entity in data:
    #    if data[entity]

    # create an instance of WikidataJsonDump
    wjd_dump_path = '/Users/malong/Downloads/latest-all.json.gz'
    #wjd_dump_path = '/Users/malong/Downloads/humans.ndjson'
    wjd = WikidataJsonDump(wjd_dump_path)
    main(wjd=wjd)
    main_category(wjd, category='all')