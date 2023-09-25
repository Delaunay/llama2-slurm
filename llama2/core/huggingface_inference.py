import os

os.environ["HF_DATASETS_CACHE"] = "/media/newton/T7/cache"
os.environ["TRANSFORMERS_CACHE"] = "/media/newton/T7/cache"

from transformers import AutoTokenizer
import transformers
import torch


models = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-70b-chat-hf",
]


def load(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


for model in models:
    try:
        load(model)
    except Exception as err:
        print(err)
        
print("Done")

