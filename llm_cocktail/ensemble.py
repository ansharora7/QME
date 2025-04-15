from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
dataset = load_dataset("ag_news")

model_path = "./phi2-agnews-gen/checkpoint-45000"
model_path_2 = "./phi2-agnews-gen/checkpoint-44500"
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(model_path_2, trust_remote_code=True, device_map="cpu")
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.eval()
# base_model.to("cuda" if torch.cuda.is_available() else "cpu")

finetuned_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")
finetuned_model.config.pad_token_id = tokenizer.pad_token_id
finetuned_model.eval()


# finetuned_model.to("cuda" if torch.cuda.is_available() else "cpu")

def average_into_base_model(base_model, fine_model, alpha=0.5):
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if name in fine_model.state_dict():
                param.data = alpha * fine_model.state_dict()[name] + (1 - alpha) * param.data
    return base_model


print("🔁 Averaging weights...")
merged_model = average_into_base_model(base_model, finetuned_model, alpha=0.5)
merged_model.to("cuda" if torch.cuda.is_available() else "cpu")
merged_model.eval()

label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
dataset = load_dataset("ag_news")

correct = 0
total = 0
failures = []

for example in tqdm(dataset["test"]):
    text = example["text"]
    true_label = example["label"]

    prompt = f"Input: {text}\nLabel:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(merged_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_str = None
    if "Label:" in generated:
        try:
            pred_str_raw = generated.split("Label:")[-1].strip()
            pred_str = pred_str_raw.split()[0] if pred_str_raw else None
        except IndexError:
            pred_str = None

    # Map to label index
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

accuracy = correct / total
print(f"\n🧠 Accuracy of averaged model: {accuracy * 100:.2f}%")
print(f"❌ Failed to classify {len(failures)} out of {total} samples.")
