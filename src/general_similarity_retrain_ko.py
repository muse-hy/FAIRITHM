import torch
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import argparse
import logging
import os
from tqdm import tqdm
import sys
from pathlib import Path
from utils import set_random_seed
from model_utils import get_params_map, get_all_model_grads, accumulate_grad
from consts import PAD_TOKEN, MASK_TOKEN
from partition_params import create_param_partition
from torch.utils.data import DataLoader
import csv
import re
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

# -------------------- Subject-slot masking helpers --------------------

def find_subject_span(text: str) -> Tuple[int, int] or None:
    """Find char-span for grammatical subject: greedy until the first particle (은|는|이|가)."""
    m = re.search(r"^\s*([가-힣A-Za-z0-9_][^\s]*?(?:\s+[가-힣A-Za-z0-9_][^\s]*)*)(은|는|이|가)", text)
    if not m:
        return None
    start = m.start(1)
    end = m.end(1)
    return (start, end)


def span_to_token_indices(text: str, span: Tuple[int, int], tok) -> List[int]:
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    idxs: List[int] = []
    s0, e0 = span
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s is None or e is None:
            continue
        if max(s, s0) < min(e, e0):
            idxs.append(i)
    return idxs


def apply_subject_slot_mask(text: str, tok):
    """Mask the subject slot (before first particle). Return dict with input_ids, attention_mask, labels."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    span = find_subject_span(text)
    if span is not None:
        idxs = span_to_token_indices(text, span, tok)
        if idxs:
            mask_id = tok.mask_token_id or tok.convert_tokens_to_ids(tok.mask_token)
            for i in idxs:
                if 0 <= i < input_ids.numel():
                    labels[i] = input_ids[i]
                    input_ids[i] = mask_id
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------- Stats loader --------------------

def load_occ_stats(stats_file: str) -> Dict[str, Dict[str, float]]:
    """Load occ -> {pct_disadv, pct_adv} from TSV. Values are percentages (0..100)."""
    stats: Dict[str, Dict[str, float]] = {}
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                occ = (row.get("occ") or "").strip()
                if not occ:
                    continue
                try:
                    pdv = float(row.get("pct_disadv", 0) or 0)
                except Exception:
                    pdv = 0.0
                try:
                    pav = float(row.get("pct_adv", 0) or 0)
                except Exception:
                    pav = 0.0
                stats[occ] = {"pct_disadv": pdv, "pct_adv": pav}
    except Exception as e:
        logger.warning(f"Failed to load stats file {stats_file}: {e}")
    return stats


# -------------------- Dataset: pair first two in each triad per occ --------------------

class GroupPairDataset:
    """Load TSV (occ, sentence). For each occ, iterate sentences in triads and pair the first two (biased/counter), skipping the third (unrelated).
    Returns items as (occ, bias_text, counter_text).
    """

    def __init__(self, dataset_file: str):
        self.pairs: List[Tuple[str, str, str]] = []
        occ_to_sentences = {}
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    occ = (row.get("occ") or "").strip()
                    sent = (row.get("sentence") or "").strip()
                    if not occ or not sent:
                        continue
                    occ_to_sentences.setdefault(occ, []).append(sent)
        except Exception as e:
            logger.warning(f"Failed to load dataset {dataset_file}: {e}")
            occ_to_sentences = {}
        for occ, sents in occ_to_sentences.items():
            i = 0
            n = len(sents)
            while i + 1 < n:
                self.pairs.append((occ, sents[i], sents[i+1]))
                i += 3 if (i + 2 < n) else 2
        if not self.pairs:
            logger.warning("No valid (bias, counter) pairs were formed from the dataset. Training may be a no-op.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self.pairs[idx]


# NOTE: this is the most expensive operation here
def compute_similarities(param_partition, grads_1, grads_2): 
    """Compute cosine similarity per param partition between two gradient maps."""
    '''

    :param param_partition: list of (param_name, index) pairs denoting the partition of the weights
    :param grads_1: dict from param_name to gradient tensor
    :param grads_2: dict from param_name to gradient tensor
    '''

    sims = [] # same indexing as param_partition
    for param_name, indices in param_partition: 
        grad_1 = grads_1[param_name]
        grad_2 = grads_2[param_name]

        if grad_1 is None or grad_2 is None: 
            sims.append(torch.Tensor([5]).squeeze()) # exclude from selection
            continue

        # Index into the grads if this param is partitioned
        if indices is not None: 
            grad_1 = grad_1[indices]
            grad_2 = grad_2[indices]

        cosine_sim = F.cosine_similarity(grad_1, grad_2, dim=-1).detach().cpu()
        sims.append(cosine_sim)

    return sims


def find_parameters_to_keep(param_partition, similarities, k=10000, return_target_indices=False):
    """Select k entries with the lowest similarity."""
    # k is the number of the lowest similarities

    sim_stack = torch.stack(similarities)

    # k = k or sim_stack.shape[-1]
    top_k_result = sim_stack.topk(k, largest=False, sorted=True)

    target_indices = [ind.item() for ind in top_k_result[1]]
    params_to_keep = [param_partition[ind] for ind in target_indices]

    if return_target_indices: 
        return params_to_keep, target_indices
    else: 
        return params_to_keep


def _minimize_grads_2(grads_1, grads_2): 
    """Use grads_2 as the new gradient (minimize objective 2)."""
    return grads_2


def _maximize_grads_1(grads_1, grads_2): 
    """Negate grads_1 to maximize objective 1."""
    return -grads_1


def _maximize_grads(grads_1, grads_2): 
    """Maximize the sum (push both in opposite direction)."""
    return -(grads_1+grads_2)

# Pairwise symmetric gradient combination (counter − bias)
def _pairwise_symmetric(grads_1, grads_2):
    """Counter minus bias gradients."""
    return grads_2 - grads_1


def update_model_param_grads(optimizer, model_params_map, params_to_keep, grads_1, grads_2, device, new_grad_calc=_minimize_grads_2): 
    """
    Set param.grad for selected params using new_grad_calc(grads_1, grads_2).
    By convention, grads_1 are "pos" and grads_2 are "neg".
    """

    optimizer.zero_grad(set_to_none=False) # so that any grad not for param in params_to_keep is zero
    if params_to_keep is not None: 
        for param_name, indices in params_to_keep: 
            param = model_params_map[param_name]
            if indices is None: 
                new_grad = new_grad_calc(grads_1[param_name], grads_2[param_name])
                param.grad.data.copy_(new_grad.data)
            else: 
                new_grad = new_grad_calc(grads_1[param_name][indices], grads_2[param_name][indices])
                param.grad[indices] = new_grad.to(device)
    else: 
        for param_name, param in model_params_map.items(): 
            if grads_1[param_name] is not None and grads_2[param_name] is not None: 
                new_grad = new_grad_calc(grads_1[param_name], grads_2[param_name])
                param.grad.data.copy_(new_grad.data)


def take_optim_step(optimizer, model_params_map, param_partition, grads_1, grads_2, 
                    device, k=10000, new_grad_calc=_minimize_grads_2): 
    """Pick params by similarity (k lowest), set grads, then optimizer.step()."""
    if k is not None: # do param partitioning
        similarities = compute_similarities(param_partition, grads_1, grads_2)
        params_to_keep = find_parameters_to_keep(param_partition, similarities, k=k, return_target_indices=False)
        update_model_param_grads(optimizer, model_params_map, params_to_keep, grads_1, grads_2, device, new_grad_calc=new_grad_calc)
    else: 
        update_model_param_grads(optimizer, model_params_map, None, grads_1, grads_2, device, new_grad_calc=new_grad_calc)

    optimizer.step()


def do_an_mlm_backprop(model, input_ids, attention_mask, indices, target_tokens, vocab_size, device, model_name='', 
                        do_backprop=True, multiplier=None): 
    """Do a single MLM backward pass focusing on masked positions; return scalar and logits."""

    input_ids = input_ids.to(device)
    indices = indices.to(device)
    target_tokens = target_tokens.to(device)
    attention_mask = attention_mask.to(device)

    model.zero_grad(set_to_none=False)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    if 'roberta' not in model_name: 
        logits = outputs.prediction_logits
    else: 
        logits = outputs.logits

    indices = indices.unsqueeze(-1).repeat(1, vocab_size).unsqueeze(1)
    target_tokens = target_tokens.unsqueeze(-1)
    logits = logits.gather(1, indices)
    unsqueeze_later = logits.shape[0]==1
    logits = logits.squeeze()
    if unsqueeze_later: 
        logits = logits.unsqueeze(0)
    logits = logits.gather(1, target_tokens)

    if multiplier is not None: 
        logits = logits*multiplier

    final_output = logits.sum()

    if do_backprop: 
        final_output.backward()

    return final_output, logits


def do_an_lm_backprop(model, input_ids, attention_mask, labels, device): 
    """Do a single LM backward pass (maximize log-likelihood)."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    model.zero_grad(set_to_none=False)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    loss = -outputs.loss
    loss.backward()

    return loss


def _get_checkpoint_dir(epoch, dedupe='', custom_model_dir=''):
    """Build checkpoint path under the base models directory."""
    base = Path(os.getenv('CHECKPOINT_BASE_DIR', 'models/sim_checkpoints')).expanduser()
    if custom_model_dir:
        return str(base / custom_model_dir / f'model_{epoch}')
    else:
        return str(base / dedupe / f'model_{epoch}')


def save_model(model, tokenizer, epoch, dedupe='', custom_model_dir=''): 
    """Save model and tokenizer using Hugging Face format."""
    output_dir = _get_checkpoint_dir(epoch, dedupe=dedupe, custom_model_dir=custom_model_dir)
    logger.info(f'Saving model at {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def measure_bias_level(model, bias_input_ids, bias_attention_mask, counter_input_ids, counter_attention_mask, device):
    """Measure bias as mean absolute diff of softmax probs between bias and counter masked inputs."""
    with torch.no_grad():
        bias_input_ids = bias_input_ids.to(device)
        counter_input_ids = counter_input_ids.to(device)
        bias_attention_mask = bias_attention_mask.to(device)
        counter_attention_mask = counter_attention_mask.to(device)
        bias_outputs = model(input_ids=bias_input_ids, attention_mask=bias_attention_mask)
        counter_outputs = model(input_ids=counter_input_ids, attention_mask=counter_attention_mask)
        bias_probs = F.softmax(bias_outputs.logits, dim=-1)
        counter_probs = F.softmax(counter_outputs.logits, dim=-1)
        # bias/counter subjects may tokenize to different subword counts;
        # compare over the common prefix length (this heuristic only scales k)
        common_len = min(bias_probs.size(1), counter_probs.size(1))
        bias_score = torch.mean(torch.abs(bias_probs[:, :common_len] - counter_probs[:, :common_len]))
        return bias_score.item()


def dynamic_korean_parameter_selection(model, dataloader, tokenizer, device, base_k=10000, sample_size=100):
    """Pick k dynamically based on average bias observed on a small sample (subject-slot masking)."""
    bias_scores = []
    sample_count = 0
    for batch in dataloader:
        for occ, bias_text, counter_text in batch:
            if sample_count >= sample_size:
                break
            b = apply_subject_slot_mask(bias_text, tokenizer)
            c = apply_subject_slot_mask(counter_text, tokenizer)
            b_ids = b["input_ids"].unsqueeze(0)
            b_attn = b["attention_mask"].unsqueeze(0)
            c_ids = c["input_ids"].unsqueeze(0)
            c_attn = c["attention_mask"].unsqueeze(0)
            score = measure_bias_level(model, b_ids, b_attn, c_ids, c_attn, device)
            bias_scores.append(score)
            sample_count += 1
        if sample_count >= sample_size:
            break
    if bias_scores:
        avg_bias = np.mean(bias_scores)
        adjusted_k = int(base_k * (1 + min(float(avg_bias), 1.0)))
        logger.info(f'Average bias score: {avg_bias:.4f}, Adjusted k: {adjusted_k}')
        return adjusted_k
    return base_k


# -------------------- Weighted loss helper --------------------

def compute_weighted_mlm_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute per-sample MLM loss (mean over valid tokens) and return weighted average by sample weights.
    logits: [B, T, V], labels: [B, T] with -100 ignored, weights: [B]
    """
    B, T, V = logits.shape
    device = logits.device
    losses = []
    eps = 1e-8
    for i in range(B):
        yi = labels[i]
        valid = yi != -100
        if valid.sum() == 0:
            losses.append(torch.tensor(0.0, device=device))
            continue
        logit_i = logits[i][valid]  # [N, V]
        target_i = yi[valid]        # [N]
        li = F.cross_entropy(logit_i, target_i, reduction='mean')
        losses.append(li)
    losses = torch.stack(losses)  # [B]
    w = weights.to(device).clamp(min=0.0)
    weighted_mean = (w * losses).sum() / (w.sum() + eps)
    return weighted_mean


def retrain_ko_with_gradient_similarity(model, tokenizer, optimizer, device, dataloader, batch_size, 
                                       is_mlm=True, k=10000, num_epochs=5, sim_batch_size=-1, 
                                       new_grad_calc=_minimize_grads_2, proportion_dev=0.5, 
                                       dynamic_gradient_selection=False, agg_dim=-1, start_at_epoch=0, 
                                       args=None, model_name='', use_dynamic_k=True,
                                       stats_map: Dict[str, Dict[str, float]] = None):
    """Korean retraining using gradient similarity selection (param-wise)."""
    logger.info('Korean retraining with gradient similarity')

    if sim_batch_size == -1:
        sim_batch_size = batch_size

    if dynamic_gradient_selection:
        new_grad_calc = _maximize_grads

    if sim_batch_size is not None and (sim_batch_size <= 0 or sim_batch_size % batch_size != 0):
        raise ValueError(f'Batch size for computing similarity is invalid: {sim_batch_size}')

    params_map = get_params_map(model)
    param_partition = create_param_partition(params_map, dim_to_agg=agg_dim)

    for epoch in range(start_at_epoch, num_epochs):
        logger.info(f'On epoch {epoch+1}/{num_epochs}')
        model.train()

        curr_bias_grads = None
        curr_counter_grads = None
        curr_sim_batch_count = 0

        if use_dynamic_k and epoch == start_at_epoch:
            current_k = dynamic_korean_parameter_selection(model, dataloader, tokenizer, device, base_k=k)
        else:
            current_k = k

        for batch in tqdm(dataloader):  # list of (occ, bias_text, counter_text)
            # Build masked batches by subject slot
            occ_list: List[str] = []
            bias_batch_inputs: List[torch.Tensor] = []
            bias_batch_attn: List[torch.Tensor] = []
            bias_batch_labels: List[torch.Tensor] = []
            counter_batch_inputs: List[torch.Tensor] = []
            counter_batch_attn: List[torch.Tensor] = []
            counter_batch_labels: List[torch.Tensor] = []
            bias_w_list: List[float] = []
            counter_w_list: List[float] = []
            for occ, bias_text, counter_text in batch:
                b = apply_subject_slot_mask(bias_text, tokenizer)
                c = apply_subject_slot_mask(counter_text, tokenizer)
                bias_batch_inputs.append(b["input_ids"])
                bias_batch_attn.append(b["attention_mask"])
                bias_batch_labels.append(b["labels"])
                counter_batch_inputs.append(c["input_ids"])
                counter_batch_attn.append(c["attention_mask"])
                counter_batch_labels.append(c["labels"])
                occ_list.append(occ)
                # Weights from stats (percentages -> [0,1])
                pdv = pav = 0.0
                if stats_map and occ in stats_map:
                    pdv = float(stats_map[occ].get("pct_disadv", 0.0)) / 100.0
                    pav = float(stats_map[occ].get("pct_adv", 0.0)) / 100.0
                bias_w_list.append(pdv)
                counter_w_list.append(pav)

            # Pad and stack
            def _pad_stack(tensors: List[torch.Tensor], pad_value: int = 0):
                max_len = max(t.size(0) for t in tensors)
                out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
                for i, t in enumerate(tensors):
                    out[i, :t.size(0)] = t
                return out

            bias_input_ids = _pad_stack(bias_batch_inputs, pad_value=tokenizer.pad_token_id or 0).to(device)
            bias_attention_mask = _pad_stack(bias_batch_attn, pad_value=0).to(device)
            bias_labels = _pad_stack(bias_batch_labels, pad_value=-100).to(device)
            counter_input_ids = _pad_stack(counter_batch_inputs, pad_value=tokenizer.pad_token_id or 0).to(device)
            counter_attention_mask = _pad_stack(counter_batch_attn, pad_value=0).to(device)
            counter_labels = _pad_stack(counter_batch_labels, pad_value=-100).to(device)

            # Convert weights
            bias_weights = torch.tensor(bias_w_list, dtype=torch.float32, device=device)
            counter_weights = torch.tensor(counter_w_list, dtype=torch.float32, device=device)
            # Avoid degenerate zero-sum
            if bias_weights.sum().item() == 0.0:
                bias_weights = torch.ones_like(bias_weights)
            if counter_weights.sum().item() == 0.0:
                counter_weights = torch.ones_like(counter_weights)

            # Forward (without labels) for per-sample weighted losses
            optimizer.zero_grad(set_to_none=False)
            bias_logits = model(input_ids=bias_input_ids, attention_mask=bias_attention_mask).logits
            bias_loss = compute_weighted_mlm_loss(bias_logits, bias_labels, bias_weights)
            bias_loss.backward()
            bias_grads = get_all_model_grads(model)

            optimizer.zero_grad(set_to_none=False)
            counter_logits = model(input_ids=counter_input_ids, attention_mask=counter_attention_mask).logits
            counter_loss = compute_weighted_mlm_loss(counter_logits, counter_labels, counter_weights)
            counter_loss.backward()
            counter_grads = get_all_model_grads(model)

            curr_bias_grads = accumulate_grad(curr_bias_grads, bias_grads)
            curr_counter_grads = accumulate_grad(curr_counter_grads, counter_grads)

            curr_sim_batch_count += len(batch)

            if curr_sim_batch_count >= sim_batch_size:
                take_optim_step(optimizer, params_map, param_partition,
                                curr_bias_grads, curr_counter_grads,
                                device, k=current_k, new_grad_calc=new_grad_calc)
                curr_sim_batch_count = 0
                curr_bias_grads = None
                curr_counter_grads = None

        if curr_bias_grads is not None and curr_counter_grads is not None:
            take_optim_step(optimizer, params_map, param_partition,
                            curr_bias_grads, curr_counter_grads,
                            device, k=current_k, new_grad_calc=new_grad_calc)

        saved_model_dir = save_model(model, tokenizer, epoch+1,
                                   dedupe=getattr(args, 'dedupe', ''),
                                   custom_model_dir=getattr(args, 'custom_model_dir', ''))


def main(args):
    logger.info(f'Seed is {args.seed}')
    set_random_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fast tokenizer (offsets)
    try:
        if args.is_mlm:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True, use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN, trust_remote_code=True, use_fast=True)
            raise ValueError('No non-mlms currently')
        model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        logger.info("Trying alternative loading method...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True, use_fast=True, local_files_only=True)
            model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True, local_files_only=True)
        except Exception as e2:
            logger.error(f"Fallback method also failed: {e2}")
            logger.info("Trying with KoBERT tokenizer...")
            from kobert_tokenizer import KoBertTokenizer
            tokenizer = KoBertTokenizer.from_pretrained(args.model_path_or_name)
            model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))

    print("==== [DEBUG] Tokenizer/Model Mask Token Check ====")
    print("tokenizer.mask_token:", tokenizer.mask_token)
    print("tokenizer.mask_token_id:", tokenizer.mask_token_id)
    print("tokenizer.convert_tokens_to_ids('[MASK]'):", tokenizer.convert_tokens_to_ids("[MASK]"))
    print("tokenizer.convert_ids_to_tokens(tokenizer.mask_token_id):", tokenizer.convert_ids_to_tokens(tokenizer.mask_token_id))
    print("model.config.mask_token_id:", getattr(model.config, "mask_token_id", None))
    print("===================================================")

    model.train()
    model.to(device)

    # Pair loader from TSV triads
    dataset = GroupPairDataset(args.dataset_file)
    def _collate(batch):
        return batch  # list of (occ, bias_text, counter_text)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate, num_workers=0)

    # Load stats map for weighting
    stats_map = load_occ_stats(args.stats_file) if args.stats_file else {}

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    agg_dim = -1 if args.agg_input else -2
    if getattr(args, 'pairwise_symmetric', False):
        new_grad_calc = _pairwise_symmetric
    else:
        new_grad_calc = _minimize_grads_2 if args.use_advantaged_for_grad else _maximize_grads_1

    logger.info('Retraining now with gradient similarity')

    retrain_ko_with_gradient_similarity(model, tokenizer, optimizer, device, dataloader, args.batch_size,
                                       args.is_mlm, args.k, args.num_epochs, args.sim_batch_size,
                                       new_grad_calc, args.proportion_dev, args.dynamic_gradient_selection,
                                       agg_dim, args.start_at, args, args.model_path_or_name,
                                       use_dynamic_k=args.use_dynamic_k, stats_map=stats_map)


if __name__=='__main__': 
    parser = argparse.ArgumentParser(description = 'Korean bias unlearning with gradient similarity (Improved2)')
    parser.add_argument('-m', '--model_path_or_name', type=str, required=True, help='path to the model or name of the model')
    parser.add_argument('-l', '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-k', type=int, default=10000, help='the k in top k')
    parser.add_argument('--use-full-grad', dest='k', action='store_const', const=None, help='to use the full gradient (rather than k parts)')
    parser.add_argument('-n', '--num_epochs', type=int, default=5, help='number of epochs to train for (total)')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-d', '--dataset_file', type=str, default='data/train_set.tsv', help='tsv file with triplets')
    parser.add_argument('-s', '--stats_file', type=str, default='data/kobbq_stats.tsv', help='tsv with occ,pct_disadv,pct_adv')
    parser.add_argument('--start-at', type=int, default=0, help='start at checkpoint epoch number (e.g., 1, and if training 5 epochs then 4 more epochs will be done)')
    parser.add_argument('--dedupe', type=str, default='', help='dedupe string (basically just the name of the experiment), models will be saved to `sim_checkpoints/{dedupe}/model_{epoch}`')
    parser.add_argument('--output-agg', dest='aggregation', action='store_const', const='output', default='input', help='to use output aggregation (default: input aggregation)')
    parser.add_argument('--dynamic_gradient_selection', dest='dynamic_gradient_selection', action='store_true', default=False, help='to choose disadvantaged and advantaged dynamically (default: static based on WG)')
    parser.add_argument('--use-disadvantaged', dest='use_advantaged_for_grad', action='store_false', default=True, help='to take gradient step to maximize disadvantaged (default: minimize advantaged)')
    parser.add_argument('--use-same-params', dest='sim_batch_size', action='store_const', const=None, default=-1, help='to use the same params each epoch (default: picks params each batch)')
    parser.add_argument('--nogrp', action='store_true', default=False, help='(unused) legacy flag')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--proportion_dev', type=float, default=0.5, help='proportion of dev set (default: 0.5)')
    parser.add_argument('--custom-model-dir', type=str, default='', help='Custom directory name to save models. If specified, models will be saved to src/models/{custom_model_dir}/model_{epoch}. If not specified, uses the default dedupe path.')
    parser.add_argument('--use-dynamic-k', action='store_true', default=True, help='Use dynamic k selection based on bias level (default: True)')
    parser.add_argument('--pairwise_symmetric', action='store_true', default=False, help='Use pairwise symmetric gradient combination (counter minus bias)')

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Heuristic MLM detection for Korean models
    args.is_mlm = ('bert' in args.model_path_or_name or 
                   'electra' in args.model_path_or_name.lower() or 
                   'kobert' in args.model_path_or_name.lower() or
                   'distilkobert' in args.model_path_or_name.lower())
    
    args.sim_batch_size = args.sim_batch_size
    args.use_advantaged_for_grad = args.use_advantaged_for_grad
    args.agg_input = args.aggregation=='input'
    if args.dynamic_gradient_selection: 
        direction_selection = 'dynamic'
    elif args.use_advantaged_for_grad: 
        direction_selection = 'adv'
    else: 
        direction_selection = 'disadv'

    if args.k is None: 
        partition_usage = 'full_grad'
    elif args.sim_batch_size is None: 
        partition_usage = 'all'
    else: 
        partition_usage = 'notall'

    dedupe_model_name = args.model_path_or_name.split('/')[-1]
    mode = "inp" if args.agg_input else "outp"
    dedupe = f"{dedupe_model_name}/{partition_usage}/{mode}/{direction_selection}/{args.lr}/{args.batch_size}/{args.k}"

    main(args) 