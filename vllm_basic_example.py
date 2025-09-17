import torch
from tqdm import tqdm 

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

model_name = "meta-llama/Llama-3.2-3B-Instruct"

llm = LLM(
    model=model_name,
    dtype=torch.bfloat16,
    max_model_len=2048,
    enable_prefix_caching=True,
)
sampling_params = SamplingParams(
    max_tokens=2048,
    temperature=1,
)

prompts = ["Tell me a story."] * 200

messages_list = [
    [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ] for prompt in prompts
]

responses = []
batch_size = 64
for i in tqdm(range(0, len(messages_list), batch_size)):
    end_interval = min(i+batch_size, len(messages_list))
    texts = messages_list[i:end_interval]

    completion = llm.chat(texts, sampling_params, use_tqdm=False)

    res = [comp.outputs[0].text for comp in completion]
    responses += res

print(responses[0])