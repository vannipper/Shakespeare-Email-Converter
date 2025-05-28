from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from transformers import BitsAndBytesConfig
import torch

model_path = "./shakespeare_modelv1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load base model with 4-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Load LoRA adapters from your fine-tuned model directory
model = PeftModel.from_pretrained(base_model, model_path, device_map="auto", torch_dtype=torch.float16)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

while True:
    user_input = input("Enter tone + email (or 'exit' to quit):\n")
    if user_input.lower() == "exit":
        break

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "The first word will be a tone. The rest of the message will be an email message. Convert the email message into Shakespearean language using the indicated tone.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    output = pipe(prompt, max_new_tokens=128, do_sample=True)
    print(output[0]["generated_text"])
