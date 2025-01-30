from utils import *
from login import huggingface_login
from datasets import load_from_disk
import torch.nn.functional as F
from semantic_uncertainty.uncertainty.models.base_model import BaseModel
from semantic_uncertainty.entailment import BaseEntailment
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 

class KQAModel(BaseModel):
    def __init__(self, model_name, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    
    def predict(self, input_data, temperature):
        inputs = self.tokenizer(input_data, return_tensors="pt").to(DEVICE)
        
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                pad_token_id=pad_token_id,
            )
            
        self.token_limit = 4096
        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        return full_answer
    
    def get_p_true(self, input_data):
        pass

def generate_answer(dataset, model, question_index):
    """
    Creates a LLM prompt given the K_QA question_index
    """
    question = dataset[question_index]["Question"]
    prompt = "You are a medical professional who is willing to answer any medical-based question provided. " 
    prompt += "Please respond to each question with a succinct but detailed answer, being sure to first " 
    prompt += "answer the most important aspects of the question before going to less important details. " 
    prompt += "Here is your question:\n"
    prompt += question + "\n"
    prompt += "Response:\n"

    offset = len(prompt)
    model_answer = model.predict(prompt, 0.1)
    return model_answer[offset:]

def check_implication(model, premise, hypothesis):
    prompt = "You are a medical professional who scoring student responses to "
    prompt += "a medical exam. Given a student's answer and a ground truth, determine "
    prompt += "if the student accurately addresses this ground truth in their answer.\n"
    prompt += "Here is the ground truth:\n"
    prompt += premise + "\n"
    prompt += "Here is the student's answer:\n"
    prompt += hypothesis + "\n"
    prompt += "Respond only with 'entailment', 'contradiction', or 'neutral'. "
    prompt += "Do not include any additional text or an explanation.\n "
    prompt += "Response:"
    
    offset = len(prompt)
    model_answer = model.predict(prompt, 0.1)
    model_answer = model_answer[offset:]
    binary_response = model_answer.lower()[:30]
    if 'entailment' in binary_response:
        return 2
    elif 'neutral' in binary_response:
        return 1
    elif 'contradiction' in binary_response:
        return 0
    else:
        return 1

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
    
    huggingface_login()
    
    qa_model = KQAModel(MODEL_NAME, max_new_tokens=400)
    
    k_qa = load_from_disk("~/medical-llm/data_clean/k_qa.hf")
    
    response = generate_answer(k_qa, qa_model, 0)
    
    must_haves = k_qa[0]["Must_have"]
    premise = must_haves[0]
    
    entailed = check_implication(qa_model, premise, response)
    print(entailed)