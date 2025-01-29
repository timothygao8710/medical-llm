from utils import *
from huggingface_hub import login
from datasets import load_from_disk
from LLM import generate

##### SETTINGS #####
cache_dir = '/tmp'
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
batch_size = 8
redownload = False
######################

def generate_answer(dataset, question_index, max_length=500):
    """
    Creates a LLM prompt given the K_QA question_index
    """
    question = dataset[question_index]["Question"]
    prompt = f"""
    You are a medical professional who is willing to answer any medical-based question provided. 
    Please respond to each question with a succinct but detailed answer, being sure to first 
    answer the most important aspects of the question before going to less important details. 
    Answer this question as if you were a student writing a short answer response on a test.
    
    Here is your question:
    {question}
    """
    print(prompt)
    model_answer = generate(prompt, max_length) #
    return model_answer

def check_entailment(premise, hypothesis):
    """
    Checks if the premise (the model generated answer) entails the hypothesis (one of the ground truths
    for this question). This is accomplished by asking a LLM model if this entailment is true, separate
    from the model used to answer the original question.
    """
    prompt = f"""
    You are a medical professional who is willing to help score student responses to a medical exam. Given
    a student's answer and a ground truth, determine whether or not any aspect of the student's answer entails
    the ground truth. That is, determine if the student accurately addresses this ground truth in their answer.
    
    Here is the student's answer:
    {premise}
    
    And here is the ground truth:
    {hypothesis}
    
    Simply answer 'True' if the student's answer does entail the ground truth, or 'False' otherwise.
    Do not provide any additional reasoning.
    """
    print(prompt)
    model_response = generate(prompt, max_length=5)
    print(model_response)
    is_entailed = model_response.split(" ")[0]
    return bool(is_entailed)

def check_contradiction(premise, hypothesis):
    """
    Checks if the premise (the model generated answer) contradicts the hypothesis (one of the ground truths
    for this question). This is accomplished by asking a LLM model if this there is a contradiction, separate
    from the model used to answer the original question.
    """
    prompt = f"""
    You are a medical professional who is willing to help score student responses to a medical exam. Given
    a student's answer and a ground truth, determine whether or not any aspect of the student's answer contradicts
    the ground truth. If the student simply does not address this ground truth, that is not a contradiction.
    
    Here is the student's answer:
    {premise}
    
    And here is the ground truth:
    {hypothesis}
    
    Simply answer 'True' if the student's answer does contradict the ground truth, or 'False' otherwise.
    Do not provide any additional reasoning.
    """
    print(prompt)
    model_response = generate(prompt, max_length=5)
    print(model_response)
    is_contradicted = model_response.split(" ")[0]
    return bool(is_contradicted)

def comp_score(dataset, question_index, model_answer):
    must_haves = dataset[question_index]["Must_have"]
    total_entails = 0
    for must_have in must_haves:
        if check_entailment(model_answer, must_have):
            total_entails += 1
    return total_entails / len(must_haves)

# Hallucination is not a good name, consider changing this to something else
# "Contradiction Rate"
def hall_score(dataset, question_index, model_answer):
    must_haves = dataset[question_index]["Must_have"]
    total_hallucinations = 0
    for must_have in must_haves:
        if check_contradiction(model_answer, must_have):
            total_hallucinations += 1
    return total_hallucinations # / len(must_haves) Think about this more

# Implement Factuality ideas from this: https://arxiv.org/pdf/2406.09714
if __name__ == "__main__":
    
    k_qa = load_from_disk("data_clean/k_qa.hf")
    
    print(k_qa[0])
    
    # model_answer = generate_answer(k_qa, 0)
    
    # print(model_answer)
    
    # comprehensiveness = comp_score(k_qa, 0, model_answer)
    
    # hallucinations = hall_score(k_qa, 0, model_answer)
    
    # print("Comp Score:". comprehensiveness)
    # print("Hall Score:", hallucinations)