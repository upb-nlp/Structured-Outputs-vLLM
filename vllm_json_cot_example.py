import torch
from tqdm import tqdm
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

model_name = "meta-llama/Llama-3.2-3B-Instruct"

llm = LLM(
    model=model_name,
    dtype=torch.bfloat16,
    max_model_len=2048,
    enable_prefix_caching=True,
)

from pydantic import BaseModel
from enum import Enum

class Difficulty(str, Enum):
    easy = "easy"
    hard = "hard"

class MathSolution(BaseModel):
    reasoning: str
    answer: int
    difficulty: Difficulty

json_schema = MathSolution.model_json_schema()
guided_decoding_json = GuidedDecodingParams(
    json=json_schema,
)
sampling_params = SamplingParams(
    guided_decoding=guided_decoding_json,
    max_tokens=2048,
    temperature=0.8,
)
prompt = """For the following math problem, first think step by step and then give the final answer. Finally, rate the problem as either easy or hard. 
Output in a JSON format with keys 'reasoning', 'answer' and 'difficulty'."""

contexts = [
    "If there are 3 cars and each car has 4 wheels, how many wheels are there in total?",
    "A farmer has 15 sheep and all but 8 run away. How many sheep are left on the farm?",
    "What is 7 multiplied by 6, minus 10?"
]

messages_list = [
    [
        {
            'role': 'system', 'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user', 'content': f"{prompt}\n\nReview: {context}",
        }
    ] for context in contexts
]

responses = []
batch_size = 64
for i in tqdm(range(0, len(messages_list), batch_size)):
    end_interval = min(i+batch_size, len(messages_list))

    texts = messages_list[i:end_interval]

    completion = llm.chat(texts, sampling_params, use_tqdm=False)

    res = [json.loads(comp.outputs[0].text) for comp in completion]

    responses += res

print(responses[0])