from utils import *
from login import huggingface_login
from datasets import load_from_disk
from semantic_uncertainty.uncertainty.models.base_model import BaseModel
from semantic_uncertainty.uncertainty.models.huggingface_models import HuggingfaceModel, StoppingCriteriaSub
import semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy as se
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

class KQAModel(BaseModel):
    def __init__(self, model_name, max_new_tokens=1000):
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
            print(outputs)
            
        self.token_limit = 4096
        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        return full_answer
    
    def get_p_true(self, input_data):
        pass

def generate_answer(dataset, model: HuggingfaceModel, question_index):
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
    model_answer = model.predict(prompt, 0.1)
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
    
    huggingface_login()
    
    entailment_model = se.EntailmentDeberta()
    qa_model = KQAModel(MODEL_NAME, max_new_tokens=100)
    
    k_qa = load_from_disk("~/medical-llm/data_clean/k_qa.hf")
    
    response = generate_answer(k_qa, qa_model, 0)
    
    print(response)
    
    # comprehensiveness = comp_score(k_qa, 0, model_answer)
    
    # hallucinations = hall_score(k_qa, 0, model_answer)
    
    # print("Comp Score:". comprehensiveness)
    # print("Hall Score:", hallucinations)