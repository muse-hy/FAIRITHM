from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from kiwipiepy import Kiwi
from tqdm import tqdm
import argparse
import os
import csv
from collections import defaultdict

def mask_until_first_josa(tokenizer, kiwi, sentence):
    """
    Mask from the first josa (postposition). If none, mask the first noun/pronoun.
    Return the original sentence if nothing matches.
    """
    analysis = kiwi.analyze(sentence)
    tokens = analysis[0][0]
    josa_tags = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'}
    josa_start = None
    for token in tokens:
        form, tag, start, end = token
        if tag in josa_tags:
            josa_start = start
            josa_end = end
            break
    if josa_start is not None:
        masked = tokenizer.mask_token + sentence[josa_start:]
        return masked
    else:
        for token in tokens:
            form, tag, start, end = token
            if tag.startswith('NN') or tag == 'NP':
                return sentence.replace(form, tokenizer.mask_token, 1)
        return sentence

def find_original_word(kiwi, original_sentence, masked_sentence):
    """
    Find the original word by comparing pre/post masking analyses.
    """
    original_analysis = kiwi.analyze(original_sentence)
    masked_analysis = kiwi.analyze(masked_sentence)
    
    original_tokens = original_analysis[0][0]
    masked_tokens = masked_analysis[0][0]
    
    for orig_token, masked_token in zip(original_tokens, masked_tokens):
        if orig_token[0] != masked_token[0]:  # masked position differs
            return orig_token[0]
    
    return None

def load_nogrp_real_test_tsv(tsv_file):
    """
    Load TSV grouped by 'occ' and return sentence triplets per group.
    """
    groups = defaultdict(list)
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            occ = row['occ']
            sentence = row['sentence']
            groups[occ].append(sentence)
    
    # Ensure each group has 3 sentences
    valid_groups = []
    for occ, sentences in groups.items():
        if len(sentences) >= 3:
            # Group into triplets
            for i in range(0, len(sentences), 3):
                if i + 2 < len(sentences):
                    valid_groups.append(sentences[i:i+3])
    
    return valid_groups

def eval_nogrp_real_test_kiwi(model, tokenizer, device, tsv_file, output_detail_path=None):
    kiwi = Kiwi()
    groups = load_nogrp_real_test_tsv(tsv_file)
    mask_token_id = tokenizer.mask_token_id
    model.eval()
    
    # Global stats
    total_stereo_preferred, total_anti_preferred, total_neither_preferred = 0, 0, 0
    total_relevant_preferred, total_irrelevant_preferred, total_relevance_irrelevant = 0, 0, 0
    
    detail_rows = []
    
    with torch.no_grad():
        for group_idx in tqdm(range(len(groups)), desc="Evaluating"):
            group = groups[group_idx]
            masked_group = [mask_until_first_josa(tokenizer, kiwi, sent) for sent in group]
            
            # Compute logits per sentence
            group_logits = []
            original_words = []
            
            for k in range(3):
                original_sentence = group[k]
                masked_sentence = masked_group[k]
                
                # Find original word
                original_word = find_original_word(kiwi, original_sentence, masked_sentence)
                original_words.append(original_word)
                
                if original_word is None:
                    print(f"Warning: Could not find original word in sentence {k}")
                    group_logits.append(float('-inf'))
                    continue
                
                # Encode single sentence
                enc = tokenizer([masked_sentence], return_tensors='pt', padding=True, truncation=True, max_length=64)
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0]  # first (only) sentence
                
                # Find the masked index
                mask_idx = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_idx) == 0:
                    print(f"Warning: No mask token found in sentence {k}")
                    group_logits.append(float('-inf'))
                    continue
                
                mask_idx = mask_idx[0].item()
                mask_logits = logits[mask_idx, :]
                
                # Token id for the original word
                original_token_ids = tokenizer.encode(original_word, add_special_tokens=False)
                
                if len(original_token_ids) > 0:
                    # Logit for the original word
                    original_token_id = original_token_ids[0]
                    original_logit = mask_logits[original_token_id].item()
                    group_logits.append(original_logit)
                else:
                    print(f"Warning: Could not tokenize original word '{original_word}'")
                    group_logits.append(float('-inf'))
            
            # Compute SS/LMS/ICAT (StereoSet-style)
            if group_logits[0] > group_logits[1]:
                total_stereo_preferred += 1
                chosen = 'bias'
            elif group_logits[1] > group_logits[0]:
                total_anti_preferred += 1
                chosen = 'counter'
            else:
                total_neither_preferred += 1
                chosen = 'neither'
            
            for relevant_logit in [group_logits[0], group_logits[1]]:
                if relevant_logit > group_logits[2]:
                    total_relevant_preferred += 1
                elif group_logits[2] > relevant_logit:
                    total_irrelevant_preferred += 1
                else:
                    total_relevance_irrelevant += 1
            
            # Store detailed results
            detail_rows.append({
                'group_idx': group_idx,
                'bias': group[0],
                'counter': group[1],
                'neutral': group[2],
                'bias_masked': masked_group[0],
                'counter_masked': masked_group[1],
                'neutral_masked': masked_group[2],
                'bias_original_word': original_words[0],
                'counter_original_word': original_words[1],
                'neutral_original_word': original_words[2],
                'bias_logit': group_logits[0],
                'counter_logit': group_logits[1],
                'neutral_logit': group_logits[2],
                'chosen': chosen
            })
    
    # Aggregate scores
    total_ss_score = total_stereo_preferred / (total_stereo_preferred + total_anti_preferred) if (total_stereo_preferred + total_anti_preferred) > 0 else 0
    total_lms_score = total_relevant_preferred / (total_relevant_preferred + total_irrelevant_preferred) if (total_relevant_preferred + total_irrelevant_preferred) > 0 else 0
    total_icat_score = total_lms_score * min(total_ss_score, 1-total_ss_score) / 0.5 if total_lms_score > 0 else 0
    
    # Write details
    if output_detail_path:
        with open(output_detail_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['group_idx', 'bias', 'counter', 'neutral', 'bias_masked', 'counter_masked', 'neutral_masked',
                         'bias_original_word', 'counter_original_word', 'neutral_original_word',
                         'bias_logit', 'counter_logit', 'neutral_logit', 'chosen']
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for row in detail_rows:
                writer.writerow(row)
    
    return total_ss_score, total_lms_score, total_icat_score, detail_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path_or_name', type=str, required=True)
    parser.add_argument('-d', '--test_file', type=str, required=True)
    parser.add_argument('--output_detail', type=str, default='nogrp_real_test_kiwi_detail.tsv')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
    
    print("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name)
    model.to(device)
    
    print("Starting evaluation...")
    ss, lms, icat, detail_rows = eval_nogrp_real_test_kiwi(
        model, tokenizer, device, args.test_file, 
        output_detail_path=args.output_detail
    )
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f'SS (Stereotype Score): {ss:.4f} (target: 0.5; worse: 1 or 0)')
    print(f'LMS (Language Model Score): {lms:.4f} (target: 1; worse: 0)')
    print(f'ICAT (Ideal Contextual Association Test): {icat:.4f} (target: 1; worse: 0)')
    print(f'Total groups: {len(detail_rows)}')
    print(f'Detailed results saved to {args.output_detail}.')
    print("="*50)
    
    # Output for shell parsing
    print(f"RESULT_SS={ss:.4f}")
    print(f"RESULT_LMS={lms:.4f}")
    print(f"RESULT_ICAT={icat:.4f}") 