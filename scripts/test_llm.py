"""
Test LLM - Download model and verify it works

This script:
1. Downloads the Qwen2.5-7B-Instruct model (~15GB, one-time)
2. Loads it into memory
3. Tests it with a simple prompt
4. Measures performance (speed, RAM usage)

Usage:
    python scripts/test_llm.py
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


def test_llm():
    """Test the LLM with a simple prompt and measure performance."""

    print("=" * 80)
    print("LLM Test - Qwen2.5-7B-Instruct")
    print("=" * 80)
    print()

    # Measure initial memory
    gc.collect()  # Clean up before measuring
    initial_memory = get_memory_usage()
    print(f"Initial RAM usage: {initial_memory:.2f} GB")
    print()

    # Initialize generator
    print("Step 1: Initializing Answer Generator...")
    generator = AnswerGenerator(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="auto",  # Auto-detect GPU/CPU
        load_in_8bit=False  # Set to True if you have <12GB RAM
    )
    print()

    # Load model (this is the heavy part)
    print("Step 2: Loading Model...")
    print("(First run: downloads ~15GB - may take 5-30 minutes depending on internet)")
    print("(Subsequent runs: loads from cache - takes 10-30 seconds)")
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
        print("2. Check available RAM (need 10-16GB free)")
        print("3. Try setting load_in_8bit=True in the code")
        print("4. Close other applications to free memory")
        return

    # Test with simple prompt
    print("=" * 80)
    print("Step 3: Testing with Simple Prompt")
    print("=" * 80)
    print()

    test_question = "What is computer vision?"
    test_context = """Computer vision is a field of artificial intelligence
that trains computers to interpret and understand the visual world. Using
digital images from cameras and videos and deep learning models, machines
can accurately identify and classify objects."""

    print(f"Question: {test_question}")
    print()
    print("Context (simplified):")
    print(test_context)
    print()

    # Generate answer
    print("-" * 80)
    print("Generating answer...")
    print("-" * 80)
    print()

    gen_start = time.time()
    try:
        result = generator.generate_answer(
            question=test_question,
            context=test_context,
            max_new_tokens=200,
            temperature=0.3
        )
        gen_end = time.time()
        gen_time = gen_end - gen_start

        # Display result
        print("ANSWER:")
        print(result['answer'])
        print()
        print("-" * 80)
        print()

        # Calculate tokens per second (approximate)
        answer_words = len(result['answer'].split())
        tokens_approx = answer_words * 1.3  # Rough estimate: 1 word ≈ 1.3 tokens
        tokens_per_sec = tokens_approx / gen_time

        print(f"Generation time: {gen_time:.2f} seconds")
        print(f"Approximate speed: {tokens_per_sec:.1f} tokens/second")
        print()

        # Interpret speed
        if tokens_per_sec >= 10:
            print("Speed assessment: EXCELLENT (GPU detected)")
        elif tokens_per_sec >= 3:
            print("Speed assessment: GOOD (acceptable for production)")
        elif tokens_per_sec >= 1:
            print("Speed assessment: ACCEPTABLE (usable but slow)")
        else:
            print("Speed assessment: SLOW (consider using GPU or smaller model)")

        print()

    except Exception as e:
        print(f"FAILED to generate answer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with a technical question
    print("=" * 80)
    print("Step 4: Testing with Technical Question")
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
    print("Context:")
    print(tech_context)
    print()

    print("-" * 80)
    print("Generating answer...")
    print("-" * 80)
    print()

    gen_start = time.time()
    result = generator.generate_answer(
        question=tech_question,
        context=tech_context,
        max_new_tokens=300,
        temperature=0.3
    )
    gen_end = time.time()

    print("ANSWER:")
    print(result['answer'])
    print()
    print(f"Generation time: {gen_end - gen_start:.2f} seconds")
    print()

    # Final summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"Model: {generator.model_name}")
    print(f"Device: {'GPU' if 'cuda' in str(generator.model.device) else 'CPU'}")
    print(f"Load time: {load_time:.1f} seconds")
    print(f"Model size in RAM: {memory_increase:.2f} GB")
    print(f"Generation speed: {tokens_per_sec:.1f} tokens/second")
    print()

    print("Status: LLM is working correctly!")
    print()
    print("Next steps:")
    print("1. The model is now cached and won't need to download again")
    print("2. You can use this LLM in the full Q&A pipeline")
    print("3. Try: python scripts/ask.py 'your question here'")
    print()

    # Cleanup
    print("Unloading model from memory...")
    generator.unload_model()
    gc.collect()

    final_memory = get_memory_usage()
    print(f"Final RAM usage: {final_memory:.2f} GB")
    print()


if __name__ == "__main__":
    test_llm()
