"""
Test TinyLlama - Compare speed with Qwen

TinyLlama is a 1.1B parameter model (vs Qwen's 7B).
Expected to be 5-10x faster on CPU.

Usage:
    python scripts/test_tinyllama.py
"""

import sys
import io
from pathlib import Path
import time
import psutil
import gc

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import AnswerGenerator


def get_memory_usage():
    """Get current RAM usage in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # Convert to GB


def test_tinyllama():
    """Test TinyLlama with the same prompts as Qwen test."""

    print("=" * 80)
    print("TinyLlama Speed Test - 1.1B Parameters")
    print("=" * 80)
    print()
    print("Goal: Find out if this is fast enough for ~1 minute answers")
    print()

    # Measure initial memory
    gc.collect()
    initial_memory = get_memory_usage()
    print(f"Initial RAM usage: {initial_memory:.2f} GB")
    print()

    # Initialize generator with TinyLlama
    print("Step 1: Initializing TinyLlama...")
    print("Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("Size: ~2.2 GB (vs Qwen's 15 GB)")
    print()

    generator = AnswerGenerator(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="auto",
        load_in_8bit=False
    )
    print()

    # Load model
    print("Step 2: Loading Model...")
    print("(First run: downloads ~2.2GB - much smaller than Qwen)")
    print("(Subsequent runs: loads from cache)")
    print()

    load_start = time.time()
    try:
        generator.load_model()
        load_end = time.time()
        load_time = load_end - load_start

        # Measure memory after loading
        loaded_memory = get_memory_usage()
        memory_increase = loaded_memory - initial_memory

        print()
        print(f"Model loaded in {load_time:.1f} seconds")
        print(f"RAM usage after loading: {loaded_memory:.2f} GB")
        print(f"Model size in RAM: {memory_increase:.2f} GB")
        print()

    except Exception as e:
        print(f"FAILED to load model: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check internet connection (first download)")
        print("2. Check available RAM")
        import traceback
        traceback.print_exc()
        return

    # Test 1: Simple question
    print("=" * 80)
    print("Test 1: Simple Question")
    print("=" * 80)
    print()

    test_question = "What is computer vision?"
    test_context = """Computer vision is a field of artificial intelligence
that trains computers to interpret and understand the visual world. Using
digital images from cameras and videos and deep learning models, machines
can accurately identify and classify objects."""

    print(f"Question: {test_question}")
    print()

    gen_start = time.time()
    try:
        result = generator.generate_answer(
            question=test_question,
            context=test_context,
            max_new_tokens=150,  # Shorter to test speed
            temperature=0.3
        )
        gen_end = time.time()
        gen_time = gen_end - gen_start

        print()
        print("ANSWER:")
        print(result['answer'])
        print()
        print("-" * 80)
        print()

        # Calculate tokens per second
        answer_words = len(result['answer'].split())
        tokens_approx = answer_words * 1.3
        tokens_per_sec = tokens_approx / gen_time

        print(f"Generation time: {gen_time:.2f} seconds")
        print(f"Approximate speed: {tokens_per_sec:.2f} tokens/second")
        print()

        # Compare with Qwen
        qwen_speed = 0.1  # From previous test
        speedup = tokens_per_sec / qwen_speed
        print(f"Speed comparison: {speedup:.1f}x faster than Qwen")
        print()

        # Estimate time for typical answer
        typical_answer_tokens = 200
        estimated_time = typical_answer_tokens / tokens_per_sec
        print(f"Estimated time for 200-token answer: {estimated_time:.1f} seconds")
        print()

        if estimated_time <= 60:
            print("SUCCESS: Can generate typical answers in under 1 minute!")
        elif estimated_time <= 120:
            print("GOOD: Can generate typical answers in 1-2 minutes")
        elif estimated_time <= 300:
            print("ACCEPTABLE: Can generate typical answers in 3-5 minutes")
        else:
            print("SLOW: Still takes 5+ minutes per answer")

        print()

    except Exception as e:
        print(f"FAILED to generate answer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Technical question
    print("=" * 80)
    print("Test 2: Technical Question (Citation Test)")
    print("=" * 80)
    print()

    tech_question = "How does Vision Transformer work?"
    tech_context = """[Chunk 1 - Paper ID: 2, Page: 3]
Vision Transformer (ViT) splits an image into fixed-size patches (typically 16x16 pixels).
Each patch is linearly embedded and position embeddings are added. The sequence of
embedded patches is then processed by a standard Transformer encoder.

[Chunk 2 - Paper ID: 2, Page: 5]
ViT achieved 88.55% top-1 accuracy on ImageNet when pre-trained on JFT-300M dataset.
The model uses 12 transformer layers with 768 hidden dimensions."""

    print(f"Question: {tech_question}")
    print()

    gen_start = time.time()
    result = generator.generate_answer(
        question=tech_question,
        context=tech_context,
        max_new_tokens=150,
        temperature=0.3
    )
    gen_end = time.time()

    print()
    print("ANSWER:")
    print(result['answer'])
    print()
    print(f"Generation time: {gen_end - gen_start:.2f} seconds")
    print()

    # Check if citations are present
    has_citations = "Paper ID" in result['answer'] or "Page:" in result['answer']
    if has_citations:
        print("Citation check: PASS - Model includes citations")
    else:
        print("Citation check: WARNING - Model may not follow citation format well")
        print("(TinyLlama is smaller, so citation quality may be lower)")

    print()

    # Final summary
    print("=" * 80)
    print("TINYLLAMA TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"Model: TinyLlama-1.1B-Chat-v1.0")
    print(f"Size: 1.1B parameters (vs Qwen's 7B)")
    print(f"Load time: {load_time:.1f} seconds")
    print(f"Model size in RAM: {memory_increase:.2f} GB")
    print(f"Generation speed: {tokens_per_sec:.2f} tokens/second")
    print(f"Speed improvement: {speedup:.1f}x faster than Qwen")
    print()

    if estimated_time <= 60:
        print("VERDICT: TinyLlama can hit the 1-minute target for typical answers!")
        print()
        print("Trade-offs:")
        print("  + Much faster generation")
        print("  + Smaller download (2GB vs 15GB)")
        print("  + Less RAM usage")
        print("  - Lower quality (less sophisticated reasoning)")
        print("  - May struggle with complex technical questions")
        print("  - Citations may be less accurate")
    else:
        print(f"VERDICT: TinyLlama is {speedup:.1f}x faster but still needs {estimated_time:.0f}s")
        print("         for typical answers. Not quite at 1-minute target.")

    print()
    print("Recommendation:")
    if speedup >= 5:
        print("- Use TinyLlama for quick answers where speed matters")
        print("- Keep Qwen for complex questions where quality matters")
        print("- You can switch between them in the code easily")
    else:
        print("- Consider trying Phi-3.5-mini (3.8B) for better balance")
        print("- Or accept slower answers with better quality")

    print()

    # Cleanup
    print("Unloading model from memory...")
    generator.unload_model()
    gc.collect()

    final_memory = get_memory_usage()
    print(f"Final RAM usage: {final_memory:.2f} GB")
    print()


if __name__ == "__main__":
    test_tinyllama()
