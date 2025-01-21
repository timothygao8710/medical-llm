from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import json
import random
import torch.nn.functional as F
import os
from question_loader import get_correct_answer, get_row_query, get_row_query_cot
import pickle
from utils import get_entropy_from_probabilities, dump_data
from LLM import get_next_token_fast, EOS_TOKEN, generate_response
import numpy as np

##### SETTINGS #####
cache_dir = '/tmp'
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "medalpaca/medalpaca-7b"
possible_outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
batch_size = 8
# redownload = False
redownload = True
data_outpath = './data/test'
######################

if redownload:
    model_cache_path = os.path.join(cache_dir, model_name)
    if os.path.exists(model_cache_path):
        os.rmdir(model_cache_path)

tot_questions = 2000
# tot_questions = get_data_len()
print(tot_questions)

res = []
correct_count = 0

temp = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7}

with tqdm(total=tot_questions) as pbar:
    for row in range(tot_questions):
        cor_ans = get_correct_answer(row)
        query = get_row_query_cot(row)
        response, entropies = generate_response(query)
        
        print(query, response, entropies)
        
        temp = response.split('Therefore, the final answer is: ')
        if(len(temp) > 1):
            model_ans = temp[1][0]
        else:
            model_ans = None
        
        print(model_ans)
        print('=' * 100)
        
        correct_count += 1 if model_ans == cor_ans else 0
        pbar.set_postfix({'Correct %': f'{(correct_count / (row + 1)) * 100:.2f}%'})
        pbar.update(1)

        print(row, model_ans == cor_ans, np.mean(entropies), model_ans, cor_ans)
        res.append({
                "row": row,
                "avg_entropy": np.mean(entropies),
                "entropies": entropies,
                "is_correct": model_ans == cor_ans,
                "model_response": response,
                "model_ans":model_ans
            })

    if (row + 1) % 100 == 0:
        dump_data(res, data_outpath)        

dump_data(res, data_outpath)