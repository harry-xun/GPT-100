# python split.py --lang python --src_file ../data/python/jsons/ --src_dir ../data/python/processed_with_verdict/ --out_dir ../data/python/processed_with_verdict/ --test_cases ../data/atcoder_test_cases --with_verdict yes

# Modified FixEval code

import os
import json
import random
import argparse
from tqdm import tqdm
from pprint import pprint
from glob import glob
from collections import defaultdict
import sys
from difflib import SequenceMatcher
import pandas as pd
from random import sample
# from deduplication import DuplicateDetector
sys.path.append("..")
random.seed(1234)
from joblib import Parallel, delayed

# from codegen.preprocessing.lang_processors.java_processor import JavaProcessor
# from codegen.preprocessing.lang_processors.python_processor import PythonProcessor

root_folder = "FixEval/third_party"
# jprocessor = JavaProcessor(root_folder=root_folder)
# pyprocessor = PythonProcessor(root_folder=root_folder)

# write out args here
lang = "python"
src_file = "FixEval/data/python/jsons/"
src_dir = "FixEval/data/python/processed_with_verdict/"
out_dir = "FixEval/data/python/processed_with_verdict/" 
test_cases = "FixEval/data/atcoder_test_cases"
with_verdict = "yes"

def load_collected_test_suit():
    problemlist=pd.read_csv("FixEval/src/problem_list.csv")
    problems = defaultdict(list)
    for index, row in tqdm(problemlist.iterrows()):
        if(row['dataset']=='AtCoder'):
            if("AtCoder Regular Contest" in row['name']):
                number = row['name'].split(" ")[3]
                problems["ARC"+number].append(row['id'])
            if("AtCoder Beginner Contest" in row['name']):
                number = row['name'].split(" ")[3]
                problems["ABC"+number].append(row['id'])
            if("AtCoder Grand Contest" in row['name']):
                number = row['name'].split(" ")[3]
                problems["AGC"+number].append(row['id'])
    folders = glob(f"{test_cases}/*")

    final_keys = []
    for idx in range(len(folders)):
        folders[idx] = folders[idx].replace(f"{test_cases}/", "")
    #print(folders)
    for key in problems.keys():
        if key in folders:
            if len(problems[key]) == len(glob(f"{test_cases}/"+key+"/*")):
                final_keys.append(key)
                
        elif key.lower() in folders :
            if len(problems[key]) == len(glob(f"{test_cases}/"+ key.lower() +"/*")):
                final_keys.append(key)

    problemid_to_tc = {}
    for key in problems:
        if(key in final_keys):
            for idx, prob_id in enumerate(problems[key]):
                folder_list = glob(f"{test_cases}/"+key+"/*")
                if(len(folder_list)==0):
                    folder_list = glob(f"{test_cases}/"+key.lower()+"/*")
                problemid_to_tc[prob_id] = folder_list[idx]
    
    print("len(problemid_to_tc) = ", len(problemid_to_tc))
    return problemid_to_tc

def calculate_similarity(code1_tokens, code2_tokens):
    code1 = ' '.join(code1_tokens)
    code2 = ' '.join(code2_tokens)
    return SequenceMatcher(None, code1, code2).ratio()

def deduplicate_jaccard(database, processor):
    accepted_sub = set()
    problem_to_dataidx = defaultdict(list)
    sim = []
    for idx,dt in enumerate(database):    
        if dt[1]['submission_id'] not in accepted_sub:
            accepted_sub.add(dt[1]['submission_id'])
            problem_to_dataidx[dt[1]['problem_id']].append(idx)
    duplicate_submission_id = []
    def solve(problem):
        print(len(sim))
        for idx in problem_to_dataidx[problem]:
            for idx2 in problem_to_dataidx[problem]:
                if idx!=idx2:
                    sim.append(calculate_similarity(database[idx][1]['code_tokens'], database[idx2][1]['code_tokens']))
    
    #Parallel(n_jobs=8, prefer="threads")(delayed(solve)(problem) for problem in tqdm(problem_to_dataidx.keys()))
    #print(len(sim))
    #print(sum(sim) / len(sim))
    #return []
    exclude_submissions = set()
    for problem in tqdm(problem_to_dataidx.keys()):
        try:
            detector = DuplicateDetector()
            data_idx_list = problem_to_dataidx[problem]
            if(len(data_idx_list)<=3):
                continue
            for idx in data_idx_list:
                detector.add_file(id = idx,tokens = processor.tokenize_code(database[idx][1]['code_tokens']))   
            exclude_document_ids = detector.compute_ids_to_exclude()
            for id in exclude_document_ids:
                exclude_submissions.add(database[idx][1]['submission_id'])
        except Exception as e:
            #print(e)
            pass
    deduplication_database = []
    for data in database:
        if data[1]['submission_id'] not in exclude_submissions:
            deduplication_database.append(data.copy())
    return deduplication_database

def problem_split():

    train_examples = []
    valid_examples = []
    test_examples = []
    unique_data = set()
    idx = 0
    files = src_file
    #print(files)
    data = []
    for file in glob(files+'*.json'):
        print(file)
        with open(file, 'r') as f:
            temp = json.load(f)
            data.extend(temp)
    data = data[:-1]
    problems_of_lang = set()
    for ex in tqdm(data):
        problems_of_lang.add(ex[0]['problem_id'])
    # processor = jprocessor if lang == 'java' else pyprocessor
    # print("previous data size ", len(data))
    # data = deduplicate_jaccard(data,processor)
    #sys.exit(0)
    print("data size after deduplication", len(data))
    

    problemid_to_tc = load_collected_test_suit()

    invalid_problems = ['p03619','p03429', 'p03334','p03110', 'p03836', 'p03394', 'p02678', 'p03046', 'p04035', 'p02669', 'p02977', 'p02997', 'p03938', 'p02692', 'p03267', 'p02975', 'p02825', 'p03952', 'p02731', 'p02936', 'p02902', 'p03263', 'p02972', 'p02690', 'p04007', 'p03257', 'p03095', 'p03746', 'p02903', 'p03097', 'p02963', 'p03245', 'p02976', 'p02694', 'p02697', 'p03044', 'p02861', 'p02850']
    print("len of problems solved in ", lang, len(problems_of_lang))
    
    train_problems = set()
    valid_problems = set()
    test_problems = []
    
    for problem in problemid_to_tc.keys():
        if problem in problems_of_lang and problem not in invalid_problems:
            test_problems.append(problem)
    test_problems = list(set(test_problems))

    valid_problems = set(test_problems[int(0.1*len(problems_of_lang)):min(len(test_problems),int(0.2*len(problems_of_lang)))])
    test_problems = test_problems[:int(0.1*len(problems_of_lang))]
    test_problems = set(test_problems)
    
    # More FixEval code, but deleted for irrelevance
    return test_problems