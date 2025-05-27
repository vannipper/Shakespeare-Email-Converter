from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",               # force CPU
    torch_dtype=torch.float32       # can also try float16 if supported
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Format prompt manually for LLaMA-3-style chat
prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a pirate chatbot who always responds in pirate speak!<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Who are you?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

output = pipe(prompt, max_new_tokens=256, do_sample=True)
print(output[0]["generated_text"])
