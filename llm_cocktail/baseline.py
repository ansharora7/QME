from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
dataset = load_dataset("ag_news")

model_path = "./phi2-agnews-gen/checkpoint-45000"

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

correct = 0
total = 0
failures = []

for example in tqdm(dataset["test"]):
    text = example["text"]
    true_label = example["label"]

    prompt = f"Input: {text}\nLabel:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Safe label extraction
    pred_str = None
    if "Label:" in generated:
        try:
            pred_str_raw = generated.split("Label:")[-1].strip()
            pred_str = pred_str_raw.split()[0] if pred_str_raw else None
        except IndexError:
            pred_str = None

    # Map generated string to label index
    pred_label = None
    if pred_str:
        for idx, name in label_map.items():
            if name.lower().startswith(pred_str.lower()):
                pred_label = idx
                break

    if pred_label is None:
        failures.append({"text": text, "generated": generated})
    if pred_label == true_label:
        correct += 1

    total += 1
    print(correct, total)
accuracy = correct / total
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
print(f"Failed to classify {len(failures)} out of {total} samples.")

# Optional: Save or print failures
# import json
# with open("failures.json", "w") as f:
#     json.dump(failures, f, indent=2)
