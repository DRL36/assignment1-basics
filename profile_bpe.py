#!/usr/bin/env python3
# uv run python profile_bpe.py
"""
Profiling script for BPE training implementation.
"""

import cProfile
import pstats
import io
from pathlib import Path
from cs336_basics.bpe import train_bpe

def profile_bpe_training():
    """Profile the BPE training function."""
    input_path = Path("tests/fixtures/corpus.en")
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    
    # Run the BPE training
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    return vocab, merges

if __name__ == "__main__":
    # Create a profiler
    profiler = cProfile.Profile()
    
    print("Starting BPE profiling...")
    
    # Profile the function
    profiler.enable()
    vocab, merges = profile_bpe_training()
    profiler.disable()
    
    print(f"BPE training completed. Vocab size: {len(vocab)}, Merges: {len(merges)}")
    
    # Create a string buffer to capture the output
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Also sort by total time (self time)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('tottime')
    ps.print_stats(20)
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (Top 20 functions by self time)")
    print("="*80)
    print(s.getvalue())
    
    # Save detailed profile to file
    profiler.dump_stats('bpe_profile.prof')
    print("\nDetailed profile saved to 'bpe_profile.prof'")
    print("You can analyze it with: python -m pstats bpe_profile.prof")
