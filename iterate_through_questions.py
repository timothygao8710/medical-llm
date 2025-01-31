import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import json
import random
import torch.nn.functional as F
import os
from question_loader import get_correct_answer, get_row_query, get_row_query_cot, get_data_len
import pickle
from utils import get_entropy_from_probabilities, dump_data
from LLM import load_model, generate
import numpy as np

load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")  # Must be explicitly called before using other functions

# tot_questions = 20
tot_questions = get_data_len()
print(tot_questions)

res = []
correct_count = 0

temp = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7}

def extract_answer(llm_response):
    matches = re.findall(r"<box>(.*?)</box>", llm_response)
    return matches[-1] if matches else None

with tqdm(total=tot_questions) as pbar:
    for row in range(tot_questions):
        cor_ans = get_correct_answer(row)
        query = get_row_query_cot(row)
        # response, entropies = generate_response(query)
        response, entropies = generate(query, max_length=3000), []
        
        model_ans = extract_answer(response)
        
        print("0"*100)
        print(response)
        print("0"*100)
        
        correct_count += 1 if model_ans == cor_ans else 0
        pbar.set_postfix({'Correct %': f'{(correct_count / (row + 1)) * 100:.2f}%'})
        pbar.update(1)
        
        print(row, model_ans == cor_ans, "model ans", model_ans, "cor ans", cor_ans)
        print('=' * 100)
        
        res.append({
                "row": row,
                "avg_entropy": np.mean(entropies),
                "entropies": entropies,
                "is_correct": model_ans == cor_ans,
                "model_response": response,
                "model_ans":model_ans
            })

#     if (row + 1) % 100 == 0:
#         dump_data(res, data_outpath)        

# dump_data(res, data_outpath)
