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
guided_decoding_category = GuidedDecodingParams(choice=["Positive", "Neutral", "Negative"])
sampling_params = SamplingParams(
    guided_decoding=guided_decoding_category,
    max_tokens=2048,
    temperature=0,
)
prompt = f"For the following review, categorize it as Positive, Neutral or Negative."

contexts = [
    "I absolutely love this product! It has changed my life for the better. Highly recommend to everyone.",
    "The product is okay, does what it says but nothing extraordinary. It's neither good nor bad.",
    "I'm very disappointed with this purchase. It broke after just one use and customer service was unhelpful."
]

messages_list = [
    [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': f"{prompt}\n\nReview: {context}"}
    ] for context in contexts
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
