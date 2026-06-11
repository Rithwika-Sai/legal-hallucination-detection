import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup Model and Path
model_id = "mistralai/Mistral-7B-Instruct-v0.2" # or "tiiuae/falcon-7b-instruct"
output_file = "outputs/mistral_output.txt"

# 2. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 3. Read your source file
with open("data/test.json", "r", encoding="utf-8") as f:
    source_text = f.read()[:2000] # Taking a chunk to fit in memory

# 4. Create the Prompt
prompt = f"Summarize the following legal document strictly: {source_text}"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 5. Generate
outputs = model.generate(**inputs, max_new_tokens=200)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 6. SAVE THE FILE (This is what your hallucination script needs!)
with open(output_file, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"Done! Created {output_file}")