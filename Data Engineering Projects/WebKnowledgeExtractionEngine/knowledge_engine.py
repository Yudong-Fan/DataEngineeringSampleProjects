# coding: utf-8
import requests
import wikipedia
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
import gzip
from bs4 import BeautifulSoup
import re
import spacy
import trafilatura
import lxml
import warnings
import json
import torch
import os
from bag_model import encoder, model_tem
import hashlib

warnings.catch_warnings()
warnings.simplefilter("ignore")

KEYNAME = "WARC-TREC-ID"
HTML_KEY = "<!DOCTYPE html"
nlp = spacy.load('en_core_web_trf')
# use this if cuda is not available!
# nlp = spacy.load('en_core_web_md')

remove_type = ["DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"]  # exclude NER type list
remove_tags = ['script', 'noscript', 'style' 'form', 'code']  # exclude HTML tags

# offline dictionaries to cache the requested wiki data
data_wikipedia_cache = {}
wikipedia_candidate_cache = {}
wikipedia_disambiguation_cache = {}
redirect_cache = {}
page_view_cache = {}


def get_redirects(title):
    # return all titles for the given title's redirects
    redirects = [title]
    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "format": "json",
        "titles": "{}".format(title),
        "prop": "redirects"
    }

    try:
        if title in redirect_cache.keys():
            resp = redirect_cache[title]
        else:
            resp = requests.get(url=URL, params=PARAMS)
            redirect_cache[title] = resp
    except:
        return redirects

    if resp.status_code != 200:
        return redirects

    data = resp.json()
    pages = data["query"]["pages"]
    for k, v in pages.items():
        if 'redirects' in v:
            for re in v["redirects"]:
                redirects.append(re['title'])
    return redirects


def get_page_views(title):
    # return the page view count of the given wikipedia page (from 2022-10-01 to 2022-10-31)
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{}/monthly/2022100100/2022103100'.format(
        title)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
    try:
        if title in page_view_cache.keys():
            resp = page_view_cache[title]
        else:
            resp = requests.get(url, headers=headers)
            page_view_cache[title] = resp
    except:
        return 0

    if resp.status_code != 200:
        return 0

    data = resp.json()
    page_views = data['items'][0]['views']

    return page_views


def sort_by_weighted_scores(tmp_res_list):
    # calculate and sort by the weighted score
    if len(tmp_res_list) == 0:
        return []
    max_views = max(tmp_res_list, key=lambda x: x[4])[4]
    min_views = min(tmp_res_list, key=lambda x: x[4])[4]
    diff = max_views - min_views
    # if the largest views count is 0, the entity is most likely unlinkable
    if max_views == 0:
        return []
    if diff == 0:
        diff = max_views
    # weighted_score = 0.5 * similarity + 0.5 * normalized_page_view
    for index_, tuple_ in enumerate(tmp_res_list):
        normalized_page_view = (tuple_[4] - min_views) / diff
        weighted_score = tuple_[3] * 0.5 + normalized_page_view * 0.5
        tmp_res_list[index_] = (tuple_[0], tuple_[1], weighted_score)
    return sorted(tmp_res_list, key=lambda x: x[2], reverse=True)


def compute_similarity(candidate_redirects, entity_redirects):
    max_sim_score = 0
    # similarity measurement inspired by Guo et al.
    target_names_len = sum([len(name) for name in entity_redirects])

    for name in entity_redirects:
        lcs_list = []
        for cand in candidate_redirects:
            match = SequenceMatcher(None, cand, name).find_longest_match(0, len(cand), 0, len(name))
            lcs = cand[match.a: match.a + match.size]
            lcs_list.append(lcs)
        lcs_len = sum([len(lcs) for lcs in lcs_list])
        sim_score = lcs_len / target_names_len
        if sim_score > max_sim_score:
            max_sim_score = sim_score
    return max_sim_score


def normalize_similarity(candidate_list, min_sim, max_sim):
    # normalize the similarity such that the similarity score is between 0 and 1
    normalized_candidate_list = []
    diff = max_sim - min_sim
    for candidate in candidate_list:
        if diff == 0 and candidate[3] <= 1:
            normalized_candidate_list.append((candidate[0], candidate[1], candidate[2], candidate[3], candidate[4]))
        elif diff == 0 and candidate[3] > 1:
            normalized_candidate_list.append((candidate[0], candidate[1], candidate[2], 1.0, candidate[4]))
        else:
            normalized_sim = (candidate[3] - min_sim) / diff
            normalized_candidate_list.append((candidate[0], candidate[1], candidate[2], normalized_sim, candidate[4]))
    return normalized_candidate_list


def prepare_candidates(disambiguations, title, entity):
    entity_redirects = get_redirects(title)
    prepared_candidates = []
    sim_list = []
    for cand_title in disambiguations:
        try:
            if cand_title in wikipedia_candidate_cache.keys():
                cand_page = wikipedia_candidate_cache[cand_title]
            else:
                cand_page = wikipedia.page(cand_title, auto_suggest=False)
                wikipedia_candidate_cache[cand_title] = cand_page
        except Exception as e:
            continue

        url = cand_page.url
        candidate_redirects = get_redirects(cand_title)
        similarity = compute_similarity(candidate_redirects, entity_redirects)
        sim_list.append(similarity)
        # in case the similarity between candidate and entity is less than the threshold, drop this candidate and proceed
        if similarity < 0.3:
            continue
        # if the similarity is larger than the threshold, request for the page_views for calculating the weighted score
        page_view = get_page_views(cand_title)
        if (entity, url, cand_title, similarity, page_view) not in prepared_candidates:
            prepared_candidates.append((entity, url, cand_title, similarity, page_view))
    if len(prepared_candidates) > 0:
        # normalized the similarity score
        prepared_candidates = normalize_similarity(prepared_candidates, min(sim_list), max(sim_list))
    return prepared_candidates


def get_wikipedia_link(entity):
    # return the entity' linking in wikipedia
    if entity in data_wikipedia_cache.keys():
        return data_wikipedia_cache[entity]

    DEFAULT_PAGE_AMBIGUOUS = False
    disambiguations = []
    title = None
    try:
        if entity in wikipedia_candidate_cache.keys():
            page_py = wikipedia_candidate_cache[entity]
        else:
            page_py = wikipedia.page(entity, auto_suggest=False)
            wikipedia_candidate_cache[entity] = page_py
    # ocassionally there are requests directly or redirect link to some disambiguation pages
    except wikipedia.exceptions.DisambiguationError as e:
        DEFAULT_PAGE_AMBIGUOUS = True
        # remove the disambiguation suffix in the title and extract candidates
        title = e.title.replace(' (disambiguation)', '')
        if title in wikipedia_disambiguation_cache.keys():
            disambiguations = wikipedia_disambiguation_cache[title]
        else:
            disambiguations = e.options
            disambiguations.append(title)
            wikipedia_disambiguation_cache[title] = disambiguations
    except Exception as e:
        return None

    if DEFAULT_PAGE_AMBIGUOUS:
        # in such case the candidates are generated already
        pass
    else:
        # manually request the disambiguation page of the wikipedia entity and extract candidates
        title = page_py.title
        try:
            wikipedia.page("{}_(disambiguation)".format(title))
        except wikipedia.exceptions.DisambiguationError as e:
            if title in wikipedia_disambiguation_cache.keys():
                disambiguations = wikipedia_disambiguation_cache[title]
            else:
                disambiguations = e.options
                disambiguations.append(title)
                wikipedia_disambiguation_cache[title] = disambiguations
        except Exception as e:
            disambiguations.append(title)
    res_list = prepare_candidates(disambiguations, title, entity)
    res_list = sort_by_weighted_scores(res_list)
    if len(res_list) > 0:
        data_wikipedia_cache[entity] = res_list[0]
        # return the candidate ranked at the first place
        return res_list[0]
    # in case no candidate survives, return None => the entity is unlinkable
    return None


def get_relation(text, pos1, pos2):
    # relation extraction
    rel2id = json.load(open(os.path.join('bag_model/data/rel2id.json')))
    rel2wiki = json.load(open(os.path.join('bag_model/data/rel2wiki.json')))
    word2id = json.load(open(os.path.join('bag_model/data/glove.6B.50d_word2id.json')))

    sentence_encoder = encoder.CNNEncoder(
        token2id=word2id,
        max_length=128,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        dropout=0.5
    )

    model = model_tem.BagOne(sentence_encoder, len(rel2id), rel2id, rel2wiki)
    ckpt = os.path.join('bag_model/ckpt/trained_model.pth.tar')
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    inp = [{"context": text, "head": {"pos": pos1}, "tail": {"pos": pos2}}]

    res = model.predict(inp)
    if res:
        return res
    return None


def get_relation_url(relation):
    # return relation wikipedia linking
    params = dict(
        action='wbsearchentities',
        format='json',
        language='en',
        uselang='en',
        type='property',
        search=relation
    )
    try:
        response = requests.get('https://www.wikidata.org/w/api.php?', params).json()
        res = response.get('search')[0]['url'][2:]
        return res
    except Exception as e:
        return None


def pipeline_worker(key, line):
    tmp_data_list = line
    entity_one_line = []
    entity_context = []
    data_url_cache = {}
    for tmp_data in tmp_data_list:
        entity = tmp_data[0]
        if entity in entity_one_line:
            continue
        else:
            entity_one_line.append(entity)
            entity_context.append(tmp_data)

    data_pass_to_RE_pipeline = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        entity_links = pool.map(get_wikipedia_link, entity_one_line)
        for i, entity_link in enumerate(entity_links):
            try:
                if entity_link != None:
                    # entity linking result output
                    print("ENTITY: {}\t{}\t{}".format(key, entity_link[0], entity_link[1]))
                    data_url_cache[entity_link[0]] = entity_link[1]
                    data_pass_to_RE_pipeline.append(entity_context[i])
            except:
                pass
    data_pass_to_RE_pipeline = sorted(data_pass_to_RE_pipeline, key=lambda x: (x[3], x[0]))
    length = len(data_pass_to_RE_pipeline)
    if length > 1:
        for i in range(1, length, 1):
            # only check entities' relation when they have the same hash code
            tmp_data_cur = data_pass_to_RE_pipeline[i]
            tmp_data_pre = data_pass_to_RE_pipeline[i - 1]

            hash_cur = tmp_data_cur[3]
            hash_pre = tmp_data_pre[3]
            if hash_pre == hash_cur:
                entity_cur = tmp_data_cur[0]
                entity_pre = tmp_data_pre[0]

                pos_cur = tmp_data_cur[2]
                pos_pre = tmp_data_pre[2]

                text = tmp_data_cur[4]
                url_pre = data_url_cache[entity_pre] if entity_pre in data_url_cache.keys() else None
                url_cur = data_url_cache[entity_cur] if entity_cur in data_url_cache.keys() else None
                if url_pre and url_cur:
                    relation = get_relation(text, pos_pre, pos_cur)
                    if relation[1] >= 0.15:
                        relation_url = get_relation_url(relation[0])
                        # entity relation result output
                        if relation and relation_url:
                            tmp = "RELATION: {}\t{}\t{}\t{}\t{}".format(key, url_pre, url_cur, relation[0], relation_url)
                            print(tmp)


def prune_html(payload):
    # use trafilatura to prune html
    res = trafilatura.extract(payload)
    return res


def backup_extraction(payload):
    bs = BeautifulSoup(payload, 'lxml')
    bs.prettify()
    for s in bs(remove_tags):
        s.extract()
    # split texts from different tags with space and strip the whitespace from the beginning and end
    text = bs.get_text(" ", strip=True)
    return text


def get_html(payload):
    html = ""
    start_index = -1
    lines = payload.splitlines()
    for index, value in enumerate(lines):
        if value.startswith(HTML_KEY):
            start_index = index
            break
    for line in lines[start_index:-1]:
        html += line
    return html


def NER(text):
    doc = nlp(text)
    # exclude some entity labels to improve the overall accuracy of the NER
    entity = [(X.text, X.label_, (X.start_char, X.end_char)) for X in doc.ents if X.label_ not in remove_type]
    entity = list(set(entity))
    return entity


def pipeline(payload):
    if payload == '':
        return

    key = None
    for line in payload.splitlines():
        if line.startswith(KEYNAME):
            key = line.split(': ')[1]
            break

    if key:
        payload = get_html(payload)
        extracted_text = prune_html(payload)
        # if trafila returns nothing, use the naive extraction
        if not extracted_text:
            extracted_text = backup_extraction(payload)
        if extracted_text:
            entity = []
            # split the payload on newlines and perform NER line by line
            for line in extracted_text.split('\n'):
                # remove newlines and tabs for each line before put into NER
                current_context = re.sub(r'[\n\t\r]+', ' ', line)
                # remove unneeded punctuations
                current_context = re.sub(r'[^\w\d\s\']+', '', current_context)
                current_context_hash = str(hashlib.md5(current_context.encode('utf8')).hexdigest())
                current_line_entities = NER(current_context)
                if len(current_line_entities) > 0:
                    for e in current_line_entities:
                        entity.append((e[0], e[1], e[2], current_context_hash, current_context))
                        
            if entity:
                pipeline_worker(key, entity)


def split_records(stream):
    payload = ''
    for line in stream:
        if line.strip() == "WARC/1.0":
            yield payload
            payload = ''
        else:
            payload += line
    yield payload


if __name__ == '__main__':
    import sys

    try:
        _, INPUT = sys.argv
    except Exception as e:
        print('Usage: python3 starter-code.py INPUT')
        sys.exit(0)

    with gzip.open(INPUT, 'rt', errors='ignore') as fo:
        for record in split_records(fo):
            # pipeline running
            pipeline(record)
