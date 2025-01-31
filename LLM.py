import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np
from utils import get_entropy_from_probabilities

# Global variables
MODEL_LOADED = False
tokenizer = None
model = None
DEVICE = None
possible_outputs = None

def ensure_model_loaded(func):
    """Decorator to ensure the model is loaded before function execution."""
    def wrapper(*args, **kwargs):
        if not MODEL_LOADED:
            raise RuntimeError("You must call load_model() before using this function.")
        return func(*args, **kwargs)
    return wrapper

def load_model(name):
    """Loads the model and sets global variables."""
    global MODEL_LOADED, tokenizer, model, DEVICE, possible_outputs

    ##### SETTINGS #####
    cache_dir = '/tmp'
    model_name = name
    possible_outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
    batch_size = 8
    redownload = True
    data_outpath = './data/all_entropies'
    ######################

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if redownload:
        model_cache_path = os.path.join(cache_dir, model_name)
        if os.path.exists(model_cache_path):
            os.rmdir(model_cache_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
    tokenizer.pad_token = tokenizer.eos_token

    MODEL_LOADED = True  # Mark as loaded

def torch_to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().numpy()

@ensure_model_loaded
def get_next_token_fast(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    allowed_tokens = tokenizer.convert_tokens_to_ids(possible_outputs)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        filtered_logits = next_token_logits[allowed_tokens]
        probs = F.softmax(filtered_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        return np.take(possible_outputs, torch_to_numpy(sorted_indices)), torch_to_numpy(sorted_probs)

@ensure_model_loaded
def get_any_next_token_greedy(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model(**input_ids)
    logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(logits).item()
    
    if next_token_id == tokenizer.eos_token_id:
        return None, None
        
    return tokenizer.decode([next_token_id]), get_entropy_from_probabilities(torch_to_numpy(F.softmax(logits, dim=-1)))

@ensure_model_loaded
def get_next_token(prompt_batch, top_k):
    inputs = tokenizer(prompt_batch, padding=True, return_tensors="pt").to(DEVICE)
    allowed_tokens = tokenizer.convert_tokens_to_ids(possible_outputs)
    logits_bias = torch.full((len(prompt_batch), model.config.vocab_size), -float('inf')).to(DEVICE)
    logits_bias[:, allowed_tokens] = 0

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :] + logits_bias
        probs = F.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)

    top_k_responses = [tokenizer.convert_ids_to_tokens(top_k_indices[i]) for i in range(len(prompt_batch))]
    return top_k_responses, torch_to_numpy(top_k_probs)

@ensure_model_loaded
def generate(prompt, max_length=800):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.000001,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@ensure_model_loaded
def generate_response(prompt: str, max_tokens: int = 1000, stop_sequences: list = None) -> str:
    stop_sequences = stop_sequences or [tokenizer.eos_token]
    generated_tokens, entropies = [], []
    response_text = ""
    
    for _ in range(max_tokens):
        next_token, entropy = get_any_next_token_greedy(prompt + response_text)  
        if next_token is None:
            break
            
        generated_tokens.append(next_token)
        response_text = "".join(generated_tokens)
        
        should_stop = any(response_text.endswith(seq) for seq in stop_sequences)
        if should_stop:
            response_text = response_text.rstrip(stop_sequences[0])
            break
        
        entropies.append(entropy)

    return response_text, entropies

if __name__ == "__main__":
    # load_model("facebook/opt-125m")  # Must be explicitly called before using other functions
    load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")  # Must be explicitly called before using other functions

    prompt = "What is the capital of France?"
    print(generate(prompt, max_length=10))
    print(generate_response(prompt, max_tokens=10))
