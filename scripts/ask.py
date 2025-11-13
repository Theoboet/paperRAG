"""
Ask Questions - Main Q&A Interface

Usage:
    python scripts/ask.py "How does Vision Transformer work?"
    python scripts/ask.py  # Interactive mode
    python scripts/ask.py "What is attention?" --model qwen --verbose
"""

import sys
import io
from pathlib import Path
import argparse

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qa_pipeline import QAPipeline


def format_answer(result: dict, verbose: bool = False):
    """Pretty-print the answer and sources."""

    print("\n" + "="*70)
    print("ANSWER")
    print("="*70)
    print()
    print(result['answer'])
    print()

    if result['sources']:
        print("="*70)
        print("SOURCES")
        print("="*70)
        print()

        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['paper_title']}")
            if 'authors' in source and source['authors']:
                authors = ", ".join(source['authors'][:3])
                if len(source['authors']) > 3:
                    authors += " et al."
                print(f"   Authors: {authors}")
            if 'year' in source:
                print(f"   Year: {source['year']}")
            print(f"   Page: {source['page']}")

            if verbose:
                print(f"   Relevance: {source['relevance_score']:.3f}")
                print(f"   Preview: {source['text_preview']}")

            print()

    print("="*70)
    print(f"Time: {result['time']:.1f}s")

    if verbose:
        print(f"  Retrieval: {result['retrieval_time']:.1f}s")
        print(f"  Generation: {result['generation_time']:.1f}s")

    print("="*70)
    print()


def interactive_mode(pipeline: QAPipeline, args):
    """Interactive question-answering mode."""

    print("\n" + "="*70)
    print("📚 paperRAG - Your Research Assistant")
    print("="*70)

    # Show library stats
    stats = pipeline.get_stats()
    print(f"\nLibrary: {stats['paper_count']} papers")
    print(f"Indexed: {stats['indexed_count']} papers")
    print(f"Model: {stats['model_name'].split('/')[-1]}")
    print()
    print("Type your question (or 'quit' to exit)")
    print("="*70)

    while True:
        try:
            # Get question
            question = input("\n❓ Question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Get answer
            result = pipeline.ask(
                question,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose
            )

            # Display
            format_answer(result, verbose=args.verbose)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about your computer vision papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ask.py "How does Vision Transformer work?"
  python scripts/ask.py "What is attention?" --model qwen
  python scripts/ask.py --interactive --verbose
  python scripts/ask.py "Compare ResNet and ViT" --top-k 10
        """
    )

    # Question argument
    parser.add_argument(
        "question",
        nargs="?",
        help="Your question (omit for interactive mode)"
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=["tinyllama", "qwen"],
        default="tinyllama",
        help="LLM to use (default: tinyllama - fast, 1-2 min)"
    )

    # Retrieval options
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3, reduced for TinyLlama's 2048 token limit)"
    )

    # Generation options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum answer length in tokens (default: 300)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="LLM temperature 0.0-1.0 (default: 0.5, lower=more focused)"
    )

    parser.add_argument(
        "--quantized", "--quant",
        action="store_true",
        help="Use 8-bit quantized model (2x faster, slight quality loss)"
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode (ask multiple questions)"
    )

    args = parser.parse_args()

    # Model mapping
    model_map = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen": "Qwen/Qwen2.5-7B-Instruct"
    }

    # Initialize pipeline
    if args.verbose:
        print(f"\nInitializing paperRAG...")
        print(f"Model: {args.model} ({model_map[args.model].split('/')[-1]})")
        print()

    try:
        pipeline = QAPipeline(
            model_name=model_map[args.model],
            load_in_8bit=args.quantized
        )
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Interactive or single question mode
    if args.interactive or not args.question:
        interactive_mode(pipeline, args)
    else:
        # Single question mode
        question = args.question

        try:
            result = pipeline.ask(
                question,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose
            )

            format_answer(result, verbose=args.verbose)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
