from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Path to your local fine-tuned model directory
model_path = "./shakespeare_modelv1"

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",            # Let accelerate handle device placement (works if you have multiple GPUs)
    torch_dtype=torch.float16     # Use float16 if your GPU supports it; else remove or set to float32
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

while True:
    user_input = input("Input email to convert to Shakespearean (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "The first word will be a tone. The rest of the message will be an email message. Convert the email message into Shakespearean language using the indicated tone.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    output = pipe(prompt, max_new_tokens=256, do_sample=True)
    print(output[0]["generated_text"])