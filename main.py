from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Meta-Llama-3-8B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Explain black holes to a 5-year-old."
print("Generating...")
output = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

print("\n=== Output ===")
print(output[0]["generated_text"])
