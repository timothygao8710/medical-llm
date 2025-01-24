import pandas as pd
import ast
import random
import string

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]

FEW_SHOT = """
######################
-Examples-
######################
Question -

A 50-year-old man comes to the physician because of a 6-month history of difficulties having sexual intercourse due to erectile dysfunction. He has type 2 diabetes mellitus that is well controlled with metformin. He does not smoke. He drinks 5–6 beers daily. His vital signs are within normal limits. Physical examination shows bilateral pedal edema, decreased testicular volume, and increased breast tissue. The spleen is palpable 2 cm below the left costal margin. Abdominal ultrasound shows an atrophic, hyperechoic, nodular liver. An upper endoscopy is performed and shows dilated submucosal veins 2 mm in diameter with red spots on their surface in the distal esophagus. Therapy with a sildenafil is initiated for his erectile dysfunction. Which of the following is the most appropriate next step in management of this patient's esophageal findings?

Choices -
A Injection sclerotherapy
B Nadolol therapy
C Losaratan therapy
D Octreotide therapy
E Isosorbide mononitrate therapy
F Endoscopic band ligation
G Transjugular intrahepatic portosystemic shunt
H Metoprolol therapy

As an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is
######################
B
######################
Question -

A 7-year-old boy is brought to the physician by his mother because his teachers have noticed him staring blankly on multiple occasions over the past month. These episodes last for several seconds and occasionally his eyelids flutter. He was born at term and has no history of serious illness. He has met all his developmental milestones. He appears healthy. Neurologic examination shows no focal findings. Hyperventilation for 30 seconds precipitates an episode of unresponsiveness and eyelid fluttering that lasts for 7 seconds. He regains consciousness immediately afterward. An electroencephalogram shows 3-Hz spikes and waves. Which of the following is the most appropriate pharmacotherapy for this patient?

Choices -

A Vigabatrin
B Lamotrigine
C Pregabalin
D Clonazepam
E Carbamazepine
F Ethosuximide
G Phenytoin
H Levetiracetam

As an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is
######################
F
######################
-Real Data-
######################
"""

data = pd.read_json(path_or_buf='/accounts/projects/binyu/timothygao/Benign-Perturbation-Attack/data_clean/questions/US/US_qbank.jsonl', lines=True)
print(data.head())

random_state = 42

def get_data_len():
    return len(data)

def get_row_query(i):
    query = ''
    # query += FEW_SHOT
    query += 'Question -\n\n' + get_row_question(i) + '\n\nChoices -\n\n' + getOptionsString(get_row_options_dict(i), False)
    query += '\nAs an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is '
    return query 

def get_row_query_cot(i):
    query = '''
You are an extremely experienced and knowledgeable medical professional answering a question in your domain.

Think step-by-step to answer the following question:\n\n
'''
    
    query += 'Question -\n\n' + get_row_question(i) + '\n\nChoices -\n\n' + getOptionsString(get_row_options_dict(i), False)
    
    query += '\n'
    
    query += '''
'''
    
    return query     
    
def get_row_question(i):
    return data['question'][i]

def get_row_options_dict(i):
    return data['options'][i]

def get_correct_answer(row):
    return data['answer'][row]

def getOptionsString(options, randomized=False):
    if isinstance(options, str):
        options = strToDict(options)
    options = [(i, options[i]) for i in options]
    res = ''
    idx = 0
    for i in options:  
        res += choices[idx] + ' ' + i[1].split('\n')[0]
        res += '\n'
        idx += 1
    return res

def get_ith_option(row, i):
    options = get_row_options_dict(row)
    
    if isinstance(options, str):
        options = strToDict(options)
    
    options = [options[j] for j in options]

    return options[i].split('\n')[0]

def strToDict(Str):
    try:
        return ast.literal_eval(Str)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to dictionary: {e}")
        return {}

if __name__ == '__main__':
    # test = get_compare_query_func(0)
    # print(test(2, 1))
    # print(test(1, 2))
    # print(get_is_correct_query(97))
    # print(get_row_query(97))
    # print(get_correct_answer(97))
    
    # high_entropy_correct = [9904, 13, 2147,
    #                         1842, 1737, 3212, 1521, 754, 483, 3390]
    
    # # high_entropy_correct = [2497, 2577, 1080]
    
    # for i in high_entropy_correct:
    #     print(get_row_query(i))
    #     print(get_correct_answer(i))
    #     print('\n')

    # print(get_row_query(5236))    
    # print(get_correct_answer(5236))
    
    # print(get_row_query(9444))    
    # print(get_correct_answer(9444))
    
    # print(get_row_query(13641))    
    # print(get_correct_answer(13641))
    
    

    low_entropy_incorrect = [11273, 3086, 12453, 9482, 14119, 14031, 9444, 2143, 3832, 160]
    
    low_entropy_incorrect = [11273, 3086, 12453]
    
    for i in low_entropy_incorrect:
        # print(getOptionsArr(i))
        print(get_row_query(i))
        print(get_correct_answer(i))
        print('\n')
    
    print(get_row_query(0))
    
    