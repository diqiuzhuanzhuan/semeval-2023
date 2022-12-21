# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import time
from typing import AnyStr
from task2.configuration.config import logging
from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
import time
from qwikidata.utils import dump_entities_to_json
from task2.configuration import config
from task2.data_man.meta_data import write_json_gzip
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

def query_person(qid):
    query = 'SELECT DISTINCT ?item WHERE {' + \
            '?item p:P106 ?statement0. ' + \
            '?statement0 (ps:P106/(wdt:P279*)) wd:'+ qid +'.' + \
            '}' 
    data = get_results(endpoint_url=endpoint_url, query=query)
    ids = [result['item']['value'].replace('http://www.wikidata.org/entity/', '') for result in data["results"]["bindings"]]
    for id in ids:
        type_by_id[id] = PERSON_TYPE[qid]

def group_query(qid):
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
             

def occupation_query(qid):
    qids = get_subclasses_of_item(qid)
    for id in qids:
        type_by_id[id] = PERSON_TYPE[qid]

def get_all_id_by_type(type='GROUP'):
    if type == 'GROUP':
        for k in GROUP_TYPE:
            logging.info(k)
            group_query(k)
    elif type == 'PERSON':
        for k in PERSON_TYPE:
            logging.info(k)
            occupation_query(k)
    else:
        pass
    logging.info(len(type_by_id))

def has_occupation_politician(item: WikidataItem, truthy: bool = True) -> bool:
    """Return True if the Wikidata Item has occupation politician."""
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
    types = set()
    for k in occupation_qids:
        if k in type_by_id:
            types.add(type_by_id[k])
            break
        if len(types):
            break

    if not types:
        types.add('person')
    return types

get_all_id_by_type('GROUP')

# create an instance of WikidataJsonDump
wjd_dump_path = '/Users/malong/Downloads/wikidata-20220103-all.json.gz'
#wjd_dump_path = '/Users/malong/Downloads/humans.ndjson'
wjd = WikidataJsonDump(wjd_dump_path)

# create an iterable of WikidataItem representing politicians
person = []
t1 = time.time()
person_vocab = collections.defaultdict(set)
def enumerate_person():
    for ii, entity_dict in enumerate(wjd):
        
        if entity_dict["type"] == "item":
            entity = WikidataItem(entity_dict)
            types = has_occupation_politician(entity)
            if types:
                labels = entity._entity_dict.get('labels')
                aliases = entity._entity_dict.get('aliases')
                for lan in config.ios_639_1_code:
                    value = labels.get(lan, dict()).get('value', '')
                    person_vocab[value].update(set(types)) if value else None
                    for dic in aliases.get(lan, []):
                        person_vocab[dic['value']].update(set(types)) if dic['value'] else None
            
        if ii % 10000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(
                "found {} person among {} entities [entities/s: {:.2f}]".format(
                    len(person_vocab), ii, ii / dt
                )
            )

def enumerate_group():
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
                    person_vocab[value].update(set(types)) if value else None
                    for dic in aliases.get(lan, []):
                        person_vocab[dic['value']].update(set(types)) if dic['value'] else None
            
        if ii % 10000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(
                "found {} person among {} entities [entities/s: {:.2f}]".format(
                    len(person_vocab), ii, ii / dt
                )
            )
enumerate_group()

print(person_vocab)
out_fname = "group_entities.json.gz"
for k in person_vocab:
    person_vocab[k] = sorted(list(person_vocab[k]))

write_json_gzip(out_fname, person_vocab)


# write the iterable of WikidataItem to disk as JSON
#out_fname = "filtered_entities.json"
#dump_entities_to_json(person, out_fname)
#wjd_filtered = WikidataJsonDump(out_fname)

# load filtered entities and create instances of WikidataItem
#for ii, entity_dict in enumerate(wjd_filtered):
#    item = WikidataItem(entity_dict)
#    print(item)