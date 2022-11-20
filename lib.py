
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

def levenshtein_distance(initial_sequence, target_sequence, memory=None):
    if memory is None:
        memory = {}

    if len(initial_sequence) == 0:
        return len(target_sequence)
    if len(target_sequence) == 0:
        return len(initial_sequence)

    key = (tuple(initial_sequence), tuple(target_sequence))

    if key in memory:
        return memory[key]
        
    distance = 0
    if initial_sequence[0] == target_sequence[0]:
        distance = levenshtein_distance(initial_sequence[1:], target_sequence[1:], memory)
    else:
        distance =  1 + min(
            levenshtein_distance(initial_sequence[1:], target_sequence, memory),
            levenshtein_distance(initial_sequence, target_sequence[1:], memory),
            levenshtein_distance(initial_sequence[1:], target_sequence[1:], memory)
        )
    memory[key] = distance
    return distance


def iter_actions(initial_sequence, target_sequence):

    def replace(idx, token):
        def _action(tokens):
            tokens[idx] = token
            return tokens
        return _action, ("R", idx, token)

    def insert(idx, token):
        def _action(tokens):
            tokens.insert(idx, token)
            return tokens
        return _action, ("I", idx, token)

    def delete(idx):
        def _action(tokens):
            del tokens[idx]
            return tokens
        return _action, ("D", idx, None)

    for i in range(len(initial_sequence)):
        yield delete(i)
        for token in set(target_sequence):
            yield insert(i, token)
            if token != initial_sequence[i]:
                yield replace(i, token)
    
    for token in set(target_sequence):
            yield insert(len(initial_sequence), token)


def find_optimal_actions(initial_sequence, target_sequence):
    memory = {}
    initial_distance = levenshtein_distance(initial_sequence, target_sequence, memory)
    action_tups = []

    for action, action_tup in iter_actions(initial_sequence, target_sequence):
        new_sequence = action(initial_sequence.copy())
        new_distance = levenshtein_distance(new_sequence, target_sequence, memory)
        if new_distance < initial_distance:
            action_tups.append(action_tup)

    return action_tups


def optimal_replacement_policy(initial_sequence, target_sequence, vocab_size, empty_id):
    """
    Generates a few-hot matrix of shape (len(initial_sequence), vocab_size).
    This few-hot matrix, R, has elements R_{i,j} = 1 if setting the token at position i
    in the initial sequence to the token with token ID j causes an edit that reduces the
    Levenshtein distance between the initial sequence and the target sequence ignoring empty tokens. 
    R_{i, j} = 1 also if j is the token ID at position i and there are no possible replacement tokens
    for position i that would reduce the Levenshtein distance. Otherwise R_{i,j} = 0.

    Args:
        initial_sequence (list[int]): A list of token IDs in normal form
        target_sequence (list[int]): A list of token IDs in normal form
        vocab_size (int): The size of the vocabulary
    
    Returns:
        R (np.ndarray): A few-hot matrix of shape (len(initial_sequence), vocab_size)
    """

    actions = find_optimal_actions(initial_sequence[1:-1:2], target_sequence[1:-1:2])

    to_pos = lambda i: 2 * i + 1

    labels = np.zeros((len(initial_sequence), vocab_size))
    for action in actions:
        a, i, t = action
        if a == "R":
            labels[to_pos(i), t] = 1
        elif a == "I":
            labels[to_pos(i)-1, t] = 1
        elif a == "D":
            labels[to_pos(i), empty_id] = 1
    
    # add ones at the existing tokens where there is no action
    non_action_mask = np.where(labels.sum(axis=1) == 0)[0]
    initial_sequence = np.array(initial_sequence)
    labels[non_action_mask, initial_sequence[non_action_mask]] = 1

    return labels


def normalize_sequence(input_ids, empty_id):
    """
    Normalizes a sequence of token IDs by adding empty tokens between consecutive non-empty tokens
    and removing consecutive empty tokens. The sequence is also padded with empty tokens at the start
    and end.

    Examples where 0 is the empty token ID:

    [1, 2, 3, 4] -> [0, 1, 0, 2, 0, 3, 0, 4, 0]

    [1, 2, 0, 0, 0, 3, 4] -> [0, 1, 0, 2, 0, 3, 0, 4, 0]

    Returns:
        normalized_ids (list[int]): A list of token IDs
    """

    normalized_ids = [empty_id]

    for token_id in input_ids:
        if token_id != empty_id:
            if normalized_ids[-1] != empty_id:
                normalized_ids.append(empty_id)
            normalized_ids.append(token_id)
        else:
            if normalized_ids[-1] == empty_id:
                continue
            else:
                normalized_ids.append(token_id)

    if normalized_ids[-1] != empty_id:
        normalized_ids.append(empty_id)
    
    return normalized_ids


def corrupt_sequence(tokens, vocab_size, empty_id, n_steps, p_insert=0.333, p_delete=0.333, p_substitute=0.333):
    # normalize probabilities
    total = p_insert + p_delete + p_substitute
    p_insert /= total
    p_delete /= total
    p_substitute /= total

    normal_tokens = normalize_sequence(tokens, empty_id)
    # even indexes are empty tokens, odd indexes are normal tokens
    for _ in range(n_steps):
        # if substitute, select random non-empty token and with random token id
        # if insert, select random empty token and with random token id
        # if delete, select random non-empty token and with empty token id
        p = random.random()

        if p < p_insert:
            idx = random.randrange(0, len(normal_tokens), 2)
            normal_tokens[idx] = random.randrange(0, vocab_size)
        elif p < p_insert + p_delete and len(normal_tokens) > 1:
            idx = random.randrange(1, len(normal_tokens), 2)
            normal_tokens[idx] = empty_id
        elif len(normal_tokens) > 1:
            idx = random.randrange(1, len(normal_tokens), 2)
            normal_tokens[idx] = random.randrange(0, vocab_size)
        
        normal_tokens = normalize_sequence(normal_tokens, empty_id)

    return normal_tokens


def get_replacement_tokenizer(tokenizer_name, empty_token="[EMT]"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'additional_special_tokens': [empty_token]})
    return tokenizer


def get_replacement_model(model_name, vocab_size):
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.resize_token_embeddings(vocab_size)
    return model


class ReplacementDataset(Dataset):

    def __init__(self, text_samples, tokenizer, empty_id, n_corruptions_fn=None, p_insert=0.333, p_delete=0.333, p_substitute=0.333):
        self.text_samples = text_samples
        self.tokenizer = tokenizer
        self.empty_id = empty_id

        self.n_corruptions_fn = (lambda x: round(x * 0.15)) if n_corruptions_fn is None else n_corruptions_fn
        self.p_insert = p_insert
        self.p_delete = p_delete
        self.p_substitute = p_substitute
            
    

    def __len__(self):
        return len(self.text_samples)
            

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.text_samples[idx])
        tokens = inputs["input_ids"][1:-1]
        n_corruptions = self.n_corruptions_fn(len(tokens))
        tokens = normalize_sequence(tokens, self.empty_id)
        corrupted_tokens = corrupt_sequence(tokens, len(self.tokenizer), self.empty_id, n_corruptions, self.p_insert, self.p_delete, self.p_substitute)

        optimal_replacements = optimal_replacement_policy(corrupted_tokens, tokens, len(self.tokenizer), self.empty_id)

        token_tensor = torch.tensor(corrupted_tokens) # (seq_len)
        label_tensor = torch.tensor(optimal_replacements) # (seq_len, vocab_size)
        # normalize the label tensor so rows sum to 1
        label_tensor = label_tensor / label_tensor.sum(dim=1, keepdim=True)

        attention_mask = torch.ones_like(token_tensor)
    
        return {
            "input_ids": token_tensor,
            "attention_mask": attention_mask,
            "labels": label_tensor
        }


def get_replacement_collate_fn(tokenizer):

    def collate_fn(batch):
        # labels = batch["labels"]
        features = [
            {
                "input_ids": x["input_ids"],
                "attention_mask": x["attention_mask"]
            }
            for x in batch
        ]

        padded_features = tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        seq_length = padded_features["input_ids"].shape[1]
        vocab_size = len(tokenizer)

        labels = [
            x["labels"]
            for x in batch
        ]

        # pad labels with zero vectors to be (batch_size, seq_length, vocab_size)

        padded_labels = torch.zeros((len(batch), seq_length, vocab_size))
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label
        
        padded_features["labels"] = padded_labels
        
        return padded_features

    return collate_fn