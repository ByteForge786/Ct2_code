import ctranslate2
import transformers
from huggingface_hub import snapshot_download
model_id = "SagarKrishna/Llama-3-8B-Text2SQL_Instruct-ct2-int8_float16"
model_path = snapshot_download(model_id)
model = ctranslate2.Generator(model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_ids))

results = model.generate_batch([input_tokens], include_prompt_in_result=False, max_length=256, sampling_temperature=0.6, sampling_topp=0.9, end_token=terminators)
output = tokenizer.decode(results[0].sequences_ids[0])

print(output)
