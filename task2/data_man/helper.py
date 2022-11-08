# -*- coding: utf-8 -*- 
# # author: Feynman 
# # email: diqiuzhuanzhuan@gmail.com

from typing import AnyStr, Dict, Iterable
import urllib.request
import urllib.parse
from task2.data_man.meta_data import get_wiki_title_knowledge, get_wiki_title_google_type, write_wiki_title_google_type
from task2.configuration import config
from task2.configuration.config import logging
from tqdm import tqdm
import grequests
import random

#api_key = 'AIzaSyAYG8RO5R50nae4S9TsYAAckCRaR0yHQZI'
#api_key = 'AIzaSyCwNFSBkbsJuIrO4mkB4GhMMgqc_RlV5dU'
#api_key = 'AIzaSyDcAiF6FoWuMjiJF5z3OwY9FUte59VsWD0'
#api_key = 'AIzaSyBNg0Sfg57jOC5htu_erVWITNOhCTdXB7g'
#api_key = 'AIzaSyDEL2J2MT0HZEIT_eWIwuItVtgtkGCz4GE'
#api_key = 'AIzaSyAHqMMZyuAbPgMBXZ7gYz1cK3dZOCrSbRA'
#api_key = 'AIzaSyCEq3qm9ZV6CTcfMB2VrVSWEC-0GJq4rrE'

api_keys = [
    'AIzaSyAYG8RO5R50nae4S9TsYAAckCRaR0yHQZI',
    'AIzaSyCwNFSBkbsJuIrO4mkB4GhMMgqc_RlV5dU',
    'AIzaSyDcAiF6FoWuMjiJF5z3OwY9FUte59VsWD0',
    'AIzaSyBNg0Sfg57jOC5htu_erVWITNOhCTdXB7g',
    'AIzaSyDEL2J2MT0HZEIT_eWIwuItVtgtkGCz4GE',
    'AIzaSyAHqMMZyuAbPgMBXZ7gYz1cK3dZOCrSbRA',
    'AIzaSyCEq3qm9ZV6CTcfMB2VrVSWEC-0GJq4rrE',
    'AIzaSyDtJkpiEYoezWIM1xg1B8EKcIxRXGn3hIM', 
    'AIzaSyDTtMdP6TGMw1lndvQ0w_aSVQucG7UTm9Q',
    'AIzaSyAZwaHtdvInhh9jMMyGTnHfkuMSZ3hadB8',
    'AIzaSyCqYO0qN5Cl47GeKqePGmyNe6cNrgXw7wA',
    'AIzaSyAuaHL_48NIZM8ZyLAWFUduruXgYQOwXa0',
    'AIzaSyCDFnBtTaHXQo7sF4IP4NUANf24TdGDV30',
    'AIzaSyDkgSzCbkmNFn7L6HR8pNFjq2ruFhq51j0',
    'AIzaSyAjvHVxZheXtvLoeARUuKDA3wlW3LXcUgU',
    'AIzaSyDvYUjRjM0cTCrSsj2rh2BTtiz0x7ontA0',
    'AIzaSyA9NatGsm5I5Jass3qBdTpi1L4gBOu-Vmo',
    'AIzaSyCZuoVhUC_GvbOkrbV4ykrQm3QF-f1mHM0',
]
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'

iso639code_by_lang = {
    'Chinese': 'zh',
    'English': 'en'
}

def build_params(query: AnyStr, lang: AnyStr):
    api_key = api_keys[random.randint(0, len(api_keys)-1)]
    params = {
        'query': query,
        'limit': 1,
        'indent': True,
        'languages': iso639code_by_lang[lang],
        'key': api_key,
    }
    return params

def build_request(querys, lang):
    gets = []
    for query in querys:
        params = build_params(query=query, lang=lang)
        url = service_url + '?' + urllib.parse.urlencode(params)
        gets.append(grequests.get(url))
    return gets

def build_response(query: AnyStr, response):
    ans = dict()
    if response is None:
        logging.error('visit failed for {}'.format(query))
        return ans
    if response.status_code == 429:
        logging.error('Over resource limits. Try a more restrictive request')
        return ans
    if response.status_code != 200:
        logging.error('visit failed for {}'.format(query))
        return ans
    data = response.json()
    if "error" in data:
        logging.error('error: {}'.format(data))
    elif 'itemListElement' not in data or len(data['itemListElement']) == 0:
        logging.warning('{} is not in google.'.format(query))
        ans[query] = dict()
    else:
        significant_part = data['itemListElement'][0]['result']
        ans[query] = {
            'type': significant_part['@type'] if '@type' in significant_part else None,
            'description': significant_part['description'] if 'description' in significant_part else None
            }
    return ans
    
def get_google_knowledge(lang: AnyStr, querys: Iterable[AnyStr], ans: Dict):
    batch_querys = []
    batch_count = 50
    deal_count = 0
    for query in tqdm(querys, total=len(querys)):
        if query in ans:
            continue
        deal_count += 1
        if deal_count % 5000 == 0:
            logging.info('save current results in case of unexpected interruption, now we have {} entities.'.format(len(ans)))
            write_wiki_title_google_type(file=config.wiki_title_with_google_type_file[lang], wiki_knowledge=ans)
        if len(batch_querys) < batch_count:
            batch_querys.append(query)
            continue
        task_list = build_request(batch_querys, lang)
        res_list = grequests.map(task_list)
        ans_list = [build_response(query, response) for query, response in zip(batch_querys, res_list)]
        for item in ans_list:
            ans.update(item)
        batch_querys.clear()

    if batch_querys:
        task_list = build_request(batch_querys, lang)
        res_list = grequests.map(task_list)
        ans_list = [build_response(query, response) for query, response in zip(batch_querys, res_list)]
        for item in ans_list:
            ans.update(item)
    return ans

def main(lang: AnyStr='Chinese'):

    entity_vocab = get_wiki_title_knowledge(config.wiki_title_file[lang])
    querys = entity_vocab.keys()
    ans = get_wiki_title_google_type(config.wiki_title_with_google_type_file[lang])
    ans = get_google_knowledge(lang=lang, querys=querys, ans=ans)
    write_wiki_title_google_type(file=config.wiki_title_with_google_type_file[lang], wiki_knowledge=ans)

if __name__ == "__main__":
    main('Chinese')
