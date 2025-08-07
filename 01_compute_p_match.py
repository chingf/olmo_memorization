import argparse
import pickle
import torch
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from difflib import SequenceMatcher
from config import hf_cache_dir

class ParagraphDataset(Dataset):
    def __init__(self, paragraphs, tokenizer, prefix_len, generation_len):
        self.paragraphs = paragraphs
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.generation_len = generation_len

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        paragraph = self.paragraphs[idx]
        tokenized_paragraph = self.tokenizer(paragraph, return_tensors='pt', return_token_type_ids=False)
        n_paragraph_tokens = tokenized_paragraph['input_ids'].shape[1]
        if n_paragraph_tokens < self.prefix_len + self.generation_len:
            return None  # Skip paragraphs that are too short
        paragraph_prefix = {
            'input_ids': tokenized_paragraph['input_ids'][:, :self.prefix_len],
            'attention_mask': tokenized_paragraph['attention_mask'][:, :self.prefix_len]
        }
        return idx, paragraph_prefix, tokenized_paragraph

def collate_fn(batch):
    # Filter out None values (skipped paragraphs)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    indices, prefixes, tokenized_paragraphs = zip(*batch)
    input_ids = torch.cat([p['input_ids'] for p in prefixes], dim=0)
    attention_mask = torch.cat([p['attention_mask'] for p in prefixes], dim=0)
    return indices, {'input_ids': input_ids, 'attention_mask': attention_mask}, tokenized_paragraphs

def main(prefix_len, generation_len, n_examples, batch_size, start_index, heldout=False):
    # Load the model and tokenizer
    olmo = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-2-1124-7B", cache_dir=hf_cache_dir, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")

    # Enable multi-GPU processing
    if torch.cuda.device_count() > 1:
        olmo = torch.nn.DataParallel(olmo)

    # Load the dataset
    ds = load_dataset("allenai/dolmino-mix-1124", "flan", cache_dir=hf_cache_dir)
    paragraphs = ds['train']['text'][start_index:start_index + n_examples]

    # Create a DataLoader for batch processing
    dataset = ParagraphDataset(paragraphs, tokenizer, prefix_len, generation_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Initialize results
    results = {
        "index": [],
        "proportions": [],
        "generated_tokens": []
    }

    save_interval = 2000
    processed = 0
    prev_partial_filename = None
    filename_prefix = "results_heldout" if heldout else "results"

    # Process the dataset in batches
    for batch in dataloader:
        if batch is None:
            continue
        indices, prefixes, tokenized_paragraphs = batch

        # Generate tokens
        with torch.no_grad():
            device = next(olmo.parameters()).device
            prefixes = {k: v.to(device) for k, v in prefixes.items()}
            responses = olmo.module.generate(**prefixes, max_new_tokens=generation_len, do_sample=False)

        for idx, response, tokenized_paragraph in zip(indices, responses, tokenized_paragraphs):
            n_tokens_actually_generated = response.shape[0] - prefix_len

            # Decode the generated tokens and the actual tokens
            generated_tokens = tokenizer.convert_ids_to_tokens(response[prefix_len:])
            actual_tokens = tokenizer.convert_ids_to_tokens(
                tokenized_paragraph['input_ids'][0, prefix_len:prefix_len + n_tokens_actually_generated])

            # Find the longest matching subsequence on a token-by-token level
            matcher = SequenceMatcher(None, generated_tokens, actual_tokens)
            match = matcher.find_longest_match(0, len(generated_tokens), 0, len(actual_tokens))
            correct_predictions = match.size
            total_predictions = len(actual_tokens)

            # Calculate the proportion of correct predictions for this example
            proportion = correct_predictions / total_predictions if total_predictions > 0 else 0

            # Save results
            results["index"].append(idx + start_index)
            results["proportions"].append(proportion)
            results["generated_tokens"].append(n_tokens_actually_generated)

            print(f"Index: {idx + start_index}, Proportion: {proportion:.2f}, Generated Tokens: {n_tokens_actually_generated}")

            processed += 1
            if processed % save_interval == 0:
                end_index = indices[-1] + start_index
                partial_filename = f"{filename_prefix}_prefix{prefix_len}_gen{generation_len}_start{start_index}_end{end_index}.pkl"
                if prev_partial_filename and os.path.exists(prev_partial_filename):
                    os.remove(prev_partial_filename)
                with open(partial_filename, "wb") as f:
                    pickle.dump(results, f)
                print(f"Partial results saved to {partial_filename}")
                prev_partial_filename = partial_filename

    # Save results to a pickle file
    final_filename = f"{filename_prefix}_prefix{prefix_len}_gen{generation_len}_start{start_index}_end{start_index+n_examples}.pkl"
    if prev_partial_filename and os.path.exists(prev_partial_filename):
        os.remove(prev_partial_filename)
    with open(final_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {final_filename}")

if __name__ == "__main__":
    """
    Example: python compute_p_match.py --prefix_len 200 --generation_len 100 --n_examples 100000 --batch_size 256
    """

    parser = argparse.ArgumentParser(description="Run OLMo model with batch processing and evaluate token match proportions.")
    parser.add_argument("--prefix_len", type=int, required=True, help="Length of the prefix in tokens.")
    parser.add_argument("--generation_len", type=int, required=True, help="Number of tokens to generate.")
    parser.add_argument("--n_examples", type=int, required=True, help="Number of examples to process.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for dataset slicing.")
    parser.add_argument("--heldout", action="store_true", help="If set, use 'results_heldout' as the filename prefix.")
    args = parser.parse_args()

    main(args.prefix_len, args.generation_len, args.n_examples, args.batch_size, args.start_index, args.heldout)