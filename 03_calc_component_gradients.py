import os
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from config import hf_cache_dir
import glob
import re
import argparse

# Parse user parameters from command line
parser = argparse.ArgumentParser(description="Calculate component gradients for OLMo model.")
parser.add_argument('--prefix_len', type=int, default=200, help='Prefix length (default: 200)')
parser.add_argument('--generation_len', type=int, default=100, help='Generation length (default: 100)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
parser.add_argument('--run_low_prop', action='store_true', help='Run for proportions <= 0.15 (default: False)')
parser.add_argument('--heldout', action='store_true', help='Use heldout pickle files (default: False)')
args = parser.parse_args()

prefix_len = args.prefix_len
generation_len = args.generation_len
batch_size = args.batch_size
run_low_prop = args.run_low_prop
heldout = args.heldout

# Find all relevant pickle files
pattern = f"results_prefix{prefix_len}_gen{generation_len}_start*_end*.pkl"
pickle_files = glob.glob(pattern)
if heldout:
    pattern = f"results_heldout_prefix{prefix_len}_gen{generation_len}_start*_end*.pkl"
else:
    pattern = f"results_prefix{prefix_len}_gen{generation_len}_start*_end*.pkl"
pickle_files = glob.glob(pattern)

# Load model and tokenizer

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-1124-7B", cache_dir=hf_cache_dir)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")

# Move model to GPU
if torch.cuda.is_available():
    model = model.to('cuda')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
else:
    print("Warning: CUDA is not available. Running on CPU.")
model.eval()

# Load dataset
# (Assume start_index and n_examples are not needed, we will get indices from pickle files)
ds = load_dataset("allenai/dolmino-mix-1124", "flan", cache_dir=hf_cache_dir)
paragraphs = ds['train']['text']


# Collect indices based on flag
selected_indices = set()
for pf in pickle_files:
    with open(pf, "rb") as f:
        results = pickle.load(f)
    if run_low_prop:
        for idx, prop in zip(results["index"], results["proportions"]):
            if prop <= 0.15:
                selected_indices.add(idx)
    else:
        for idx, prop in zip(results["index"], results["proportions"]):
            if prop >= 0.8:
                selected_indices.add(idx)
selected_indices = sorted(list(selected_indices))
if run_low_prop:
    print(f"Found {len(selected_indices)} indices with proportion <= 0.15.")
else:
    print(f"Found {len(selected_indices)} indices with proportion >= 0.8.")

# Results dictionary for DataFrame
results_dict = {
    "data_index": [],
    "layer": [],
    "head": [],
    "proj_type": [],
    "grad_max": [],
    "grad_norm": []
}

end_idx = len(selected_indices) if not run_low_prop else 2000
for batch_start in range(0, end_idx, batch_size):
    batch_indices = selected_indices[batch_start:batch_start+batch_size]
    batch_paragraphs = [paragraphs[idx] for idx in batch_indices]
    tokenized = tokenizer(batch_paragraphs, return_tensors='pt', padding=True, truncation=False, return_token_type_ids=False)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = tokenized['input_ids'].to(device)

    # Filter out short paragraphs in batch
    if input_ids.shape[1] < (prefix_len + generation_len):
        continue
    prefix_ids = input_ids[:, :prefix_len].to(device)

    # Generate responses for batch
    with torch.no_grad():
        # prefix_ids already on device
        if isinstance(model, torch.nn.DataParallel):
            outputs = model.module.generate(input_ids=prefix_ids, max_new_tokens=generation_len, do_sample=False)
        else:
            outputs = model.generate(input_ids=prefix_ids, max_new_tokens=generation_len, do_sample=False)

    # For each item in batch, calculate NLL and gradients
    for i, idx in enumerate(batch_indices):
        # Check paragraph length again for each item
        if input_ids[i].shape[0] < prefix_len + generation_len:
            continue
        gen_ids = outputs[i][prefix_len:prefix_len+generation_len].unsqueeze(0).to(device)
        prefix_ids_i = prefix_ids[i].unsqueeze(0).to(device)
        input_for_nll = torch.cat([prefix_ids_i, gen_ids], dim=1)
        labels = input_for_nll.clone()
        labels[:, :prefix_len] = -100  # Only calculate loss on generated part
        input_for_nll = input_for_nll.to(device)
        labels = labels.to(device)
        model.zero_grad()
        output = model(input_ids=input_for_nll, labels=labels)
        loss = output.loss
        print(f"Index {idx}: NLL loss = {loss.item():.4f}")

        # Compute gradients
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Only process Q, K, V, O projection weights
                if any(proj in name for proj in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight']):
                    # Get layer number from name (e.g., layers.30)
                    layer_match = re.search(r'layers\.(\d+)', name)
                    if layer_match:
                        layer_num = int(layer_match.group(1))
                    else:
                        continue
                    # Get projection type
                    proj_type = [proj for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] if proj in name][0]
                    # Split by head
                    weight = param.grad.detach().cpu()
                    num_heads = getattr(model.module if isinstance(model, torch.nn.DataParallel) else model, 'config').num_attention_heads
                    head_dim = weight.shape[0] // num_heads
                    for head in range(num_heads):
                        head_grad = weight[head*head_dim:(head+1)*head_dim, :]
                        grad_max = head_grad.max().item()
                        grad_norm = head_grad.norm().item()
                        results_dict["data_index"].append(idx)
                        results_dict["layer"].append(layer_num)
                        results_dict["head"].append(head)
                        results_dict["proj_type"].append(proj_type)
                        results_dict["grad_max"].append(grad_max)
                        results_dict["grad_norm"].append(grad_norm)
                # ...existing code for other params if needed...
        model.zero_grad()

# Save results to pickle file
if heldout:
    if run_low_prop:
        output_pickle = f"grad_results_heldout_lowprop_prefix{prefix_len}_gen{generation_len}.pkl"
    else:
        output_pickle = f"grad_results_heldout_prefix{prefix_len}_gen{generation_len}.pkl"
else:
    if run_low_prop:
        output_pickle = f"grad_results_lowprop_prefix{prefix_len}_gen{generation_len}.pkl"
    else:
        output_pickle = f"grad_results_prefix{prefix_len}_gen{generation_len}.pkl"
with open(output_pickle, "wb") as f:
    pickle.dump(results_dict, f)
print(f"Saved gradient results to {output_pickle}")
