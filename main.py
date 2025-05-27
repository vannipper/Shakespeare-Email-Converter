from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
token = "your_token_here"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=token,
    device_map="auto",            # let accelerate handle device placement
    torch_dtype=torch.float16     # or float32 if preferred
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
while True:
    user_input = input()
    if user_input.lower() == "exit":
        break

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a pirate chatbot who always responds in pirate speak!<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    output = pipe(prompt, max_new_tokens=32, do_sample=True)
    print(output[0]["generated_text"])
