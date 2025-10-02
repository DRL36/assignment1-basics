"""
BPE (Byte Pair Encoding) tokenizer training implementation.
"""

import regex as re 
import os
from collections import defaultdict, Counter
from typing import BinaryIO
from .pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

def process_chunk(args):
    """Process a single chunk and return token counts."""
    input_path, start, end, special_tokens = args
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    # Split on special tokens to prevent merging across boundaries
    if special_tokens:
        # Create pattern: "token1|token2|token3"
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        segments = re.split(f"({special_pattern})", chunk)
    else:
        segments = [chunk]
    
    token_counts = Counter()
    for segment in segments:
        if segment in special_tokens:
            # This is a special token itself
            token_counts[segment] += 1
        elif segment:  # Non-empty regular text segment
            # Pre-tokenize this segment normally
            for match in re.finditer(PAT, segment):
                token = match.group()
                token_counts[token] += 1
    
    return token_counts

def merge_pair_in_sequence(byte_sequence, first_id, second_id, new_token_id, vocab, pair_counts, freq):
    """Optimized merge function with cached vocab lookups and efficient pair counting."""
    if len(byte_sequence) < 2:
        return byte_sequence[:]
    
    first_token_bytes = vocab[first_id]
    second_token_bytes = vocab[second_id] 
    merged_token_bytes = vocab[new_token_id]
    new_sequence = []
    i = 0
    
    while i < len(byte_sequence):
        # Check if current and next tokens form the pair to merge
        if (i < len(byte_sequence) - 1 and 
            byte_sequence[i] == first_id and 
            byte_sequence[i + 1] == second_id):
            
            # Update overlapping pair counts before merging
            # Left overlap: if there's a token before the merge
            if i > 0:
                left_token_bytes = vocab[byte_sequence[i-1]]
                left_pair = (left_token_bytes, first_token_bytes)
                pair_counts[left_pair] -= freq
                if pair_counts[left_pair] <= 0:
                    del pair_counts[left_pair]
                # Add new pair with merged token
                new_left_pair = (left_token_bytes, merged_token_bytes)
                pair_counts[new_left_pair] += freq
            
            # Right overlap: if there's a token after the merge
            if i + 2 < len(byte_sequence):
                right_token_bytes = vocab[byte_sequence[i+2]]
                right_pair = (second_token_bytes, right_token_bytes)
                pair_counts[right_pair] -= freq
                if pair_counts[right_pair] <= 0:
                    del pair_counts[right_pair]
                # Add new pair with merged token
                new_right_pair = (merged_token_bytes, right_token_bytes)
                pair_counts[new_right_pair] += freq
            
            # Merge: replace two tokens with the new merged token
            new_sequence.append(new_token_id)
            i += 2
        else:
            new_sequence.append(byte_sequence[i])
            i += 1
    
    return new_sequence

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input text.
    
    Args:
        input_path: Path to input text file
        vocab_size: Maximum vocabulary size (including initial bytes + special tokens + merges)
        special_tokens: List of special token strings to add to vocabulary
        
    Returns:
        vocab: Dictionary mapping token IDs to token bytes
        merges: List of merge operations in order of creation
    """
    ## init vocab
    vocab = {}
    special_token_to_id = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for special_token in special_tokens:
        special_token_id = len(vocab)
        vocab[special_token_id] = special_token.encode('utf-8')
        special_token_to_id[special_token] = special_token_id

    # pre-tokenizer
    with open(input_path, 'rb') as f:
        num_processes = 4
        # Use a common delimiter for chunking, or the first special token if available
        split_token = b"<|endoftext|>" if not special_tokens else special_tokens[0].encode('utf-8')
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((input_path, start, end, special_tokens))

    with Pool(num_processes) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)
    
    token_counts = Counter()
    for chunk_counts in chunk_results:
        token_counts.update(chunk_counts)

    # Convert pre-tokens to byte sequences for BPE training
    # also init pair_counts
    # token = "hello", freq = 10 => word_splits["hello"] = ([104, 101, 108, 108, 111], 10)
    # pair = (b'h', b'e')
    word_splits = {}
    pair_counts = defaultdict(int)
    for token, freq in token_counts.items():
        if token in special_tokens:
            # Special tokens should not be split into bytes
            special_token_id = special_token_to_id[token]
            word_splits[token] = ([special_token_id], freq)
        else:
            token_bytes = token.encode('utf-8')
            token_sequence = list(token_bytes)
            for i in range(len(token_sequence) - 1):
                # Get the actual bytes for each token ID
                first_token_bytes = vocab[token_sequence[i]]
                second_token_bytes = vocab[token_sequence[i + 1]]
                pair = (first_token_bytes, second_token_bytes)
                pair_counts[pair] += freq
            word_splits[token] = (token_sequence, freq)
    
    merges = []
    bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
    
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        
        # Find the pair with the highest count
        # In case of ties, use lexicographic ordering for deterministic results
        # best_pair = max(pair_counts.keys(), key=lambda pair: (pair_counts[pair], pair))
        best_count = -1
        best_pair = None
        for pair, count in pair_counts.items():
            if count > best_count or (count == best_count and pair > best_pair):
                best_count = count
                best_pair = pair
        
        merged_token = best_pair[0] + best_pair[1]
        new_token_id = len(vocab)
        vocab[new_token_id] = merged_token
        merges.append(best_pair)
        del pair_counts[best_pair]
        
        # Update reverse mapping with new token
        bytes_to_id[merged_token] = new_token_id
        
        # Find token IDs for the best pair
        first_id = bytes_to_id[best_pair[0]]
        second_id = bytes_to_id[best_pair[1]]
        
        # Only process words that contain the target pair (KEY OPTIMIZATION)
        new_word_splits = {}
        for token, (token_sequence, freq) in word_splits.items():
            if first_id in token_sequence and second_id in token_sequence:
                new_sequence = merge_pair_in_sequence(token_sequence, first_id, second_id, new_token_id, vocab, pair_counts, freq)
                new_word_splits[token] = (new_sequence, freq)
            else:
                new_word_splits[token] = (token_sequence, freq)
        word_splits = new_word_splits
    
    return vocab, merges


class BPETokenizer:
    """
    A simple BPE tokenizer class for encoding and decoding text.
    """
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Create reverse mapping from bytes to token IDs
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # Build merge rules in order
        self.merge_rules = {}
        for i, (first, second) in enumerate(merges):
            self.merge_rules[(first, second)] = first + second
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        # This is a simplified implementation
        # In practice, you'd need to handle the full GPT-2 tokenization process
        raise NotImplementedError("Encoding not implemented in skeleton")
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        # This is a simplified implementation
        raise NotImplementedError("Decoding not implemented in skeleton")
