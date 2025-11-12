"""
Test Citation Prompt Engineering

Try different prompt strategies to force TinyLlama to include citations.
"""

import sys
import io
from pathlib import Path
import time

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import AnswerGenerator


def test_prompt_strategy(generator, strategy_name, system_prompt, user_prompt, context, question):
    """Test a specific prompt strategy."""
    print("=" * 80)
    print(f"Testing: {strategy_name}")
    print("=" * 80)
    print()

    # Temporarily modify the generator's system prompt
    # We'll do this by directly calling the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(context=context, question=question)}
    ]

    prompt = generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = generator.tokenizer(prompt, return_tensors="pt")

    print("Generating...")
    start = time.time()

    import torch
    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=generator.tokenizer.eos_token_id
        )

    full_output = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output.split("assistant")[-1].strip()
    if answer.startswith(":"):
        answer = answer[1:].strip()

    elapsed = time.time() - start

    print(f"Answer ({elapsed:.1f}s):")
    print(answer)
    print()

    # Check for citations
    has_paper_id = "Paper ID" in answer or "paper id" in answer.lower()
    has_page = "Page" in answer or "page" in answer.lower()
    has_brackets = "(" in answer and ")" in answer

    print("Citation Check:")
    print(f"  Contains 'Paper ID': {'✓' if has_paper_id else '✗'}")
    print(f"  Contains 'Page': {'✓' if has_page else '✗'}")
    print(f"  Contains parentheses: {'✓' if has_brackets else '✗'}")

    if has_paper_id and has_page and has_brackets:
        print("  VERDICT: ✓ Good citations!")
    elif has_paper_id or has_page:
        print("  VERDICT: ⚠ Partial citations")
    else:
        print("  VERDICT: ✗ No citations")

    print()
    return answer


def main():
    print("=" * 80)
    print("Citation Prompt Engineering Test")
    print("=" * 80)
    print()
    print("Goal: Find the best prompt to make TinyLlama include citations")
    print()

    # Load TinyLlama
    print("Loading TinyLlama...")
    generator = AnswerGenerator(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="auto"
    )
    generator.load_model()
    print()

    # Test context
    context = """[Chunk 1 - Paper ID: 2, Page: 3]
Vision Transformer (ViT) splits an image into fixed-size patches (typically 16x16 pixels).
Each patch is linearly embedded and position embeddings are added.

[Chunk 2 - Paper ID: 2, Page: 5]
ViT achieved 88.55% top-1 accuracy on ImageNet when pre-trained on JFT-300M dataset."""

    question = "How does Vision Transformer work?"

    # Strategy 1: Original (from current code)
    system_1 = """You are a research assistant analyzing computer vision papers.

CRITICAL RULES:
1. Answer ONLY using information from the Context below
2. ALWAYS cite sources using format: (Paper ID: X, Page: Y)
3. Cite EVERY factual claim
4. If information is NOT in context, say "Not found in your papers"
5. Do NOT invent information
6. Use EXACT page numbers from context metadata"""

    user_1 = """Context from research papers:
{context}

Question: {question}

Answer based ONLY on context above. Include citations."""

    test_prompt_strategy(generator, "Strategy 1: Original", system_1, user_1, context, question)

    # Strategy 2: More explicit with examples
    system_2 = """You are a research assistant. You MUST cite sources.

CITATION FORMAT: (Paper ID: 2, Page: 3)

Example:
Q: What is ViT?
A: ViT is Vision Transformer (Paper ID: 2, Page: 3). It uses patches (Paper ID: 2, Page: 3).

RULES:
- Put citations after EVERY fact
- Use format: (Paper ID: X, Page: Y)
- NO answers without citations"""

    user_2 = """Context:
{context}

Question: {question}

Answer with citations after every fact:"""

    test_prompt_strategy(generator, "Strategy 2: Example-driven", system_2, user_2, context, question)

    # Strategy 3: In-context example in user message
    system_3 = """You are a helpful research assistant."""

    user_3 = """Context:
{context}

Question: {question}

IMPORTANT: Cite sources like this: (Paper ID: 2, Page: 3)

Example answer format:
"ViT splits images into patches (Paper ID: 2, Page: 3). It achieved 88.55% accuracy (Paper ID: 2, Page: 5)."

Your answer with citations:"""

    test_prompt_strategy(generator, "Strategy 3: In-context example", system_3, user_3, context, question)

    # Strategy 4: Very simple and direct
    system_4 = """You answer questions and cite sources.

ALWAYS use this format: (Paper ID: X, Page: Y)"""

    user_4 = """Information:
{context}

Question: {question}

Answer (cite EVERY fact with Paper ID and Page):"""

    test_prompt_strategy(generator, "Strategy 4: Simple & direct", system_4, user_4, context, question)

    # Strategy 5: Threaten consequences (sometimes works!)
    system_5 = """You are a research assistant.

WARNING: You will FAIL if you don't cite sources.
REQUIRED FORMAT: (Paper ID: X, Page: Y)
You must cite EVERY statement."""

    user_5 = """Context:
{context}

Question: {question}

Answer (YOU MUST include citations or your answer is WRONG):"""

    test_prompt_strategy(generator, "Strategy 5: Strong warning", system_5, user_5, context, question)

    # Strategy 6: Step-by-step instruction
    system_6 = """You are a research assistant."""

    user_6 = """Context:
{context}

Question: {question}

Instructions:
1. Read the context above
2. Answer the question
3. After EACH fact, add a citation: (Paper ID: X, Page: Y)
4. Use the Paper ID and Page from the [Chunk] headers

Your answer (with citations):"""

    test_prompt_strategy(generator, "Strategy 6: Step-by-step", system_6, user_6, context, question)

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("Check which strategy produced the best citations above.")
    print("We'll use that strategy in the main generator code.")
    print()

    generator.unload_model()


if __name__ == "__main__":
    main()
