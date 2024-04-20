import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device= device,
)


messages = [
    {"role": "system", "content": "You are someone who knows everything."},
    {"role": "user", "content": "Tell me about quantum field theory in full details."},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256000,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
