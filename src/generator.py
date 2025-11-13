"""
Generator Module

Handles LLM-based answer generation with citations using transformers library.
"""

from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress transformer warnings
warnings.filterwarnings('ignore', category=UserWarning)


class AnswerGenerator:
    """Generates answers with citations using a local LLM via transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 device: str = "auto",
                 load_in_8bit: bool = False):
        """
        Initialize the answer generator with a transformers LLM.

        Args:
            model_name: HuggingFace model name (default: Qwen2.5-7B-Instruct)
            device: Device to run on ("auto", "cpu", "cuda")
            load_in_8bit: Use 8-bit quantization to save memory (requires bitsandbytes)
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load_model(self):
        """
        Load the LLM model and tokenizer.

        This is separated from __init__ so you can control when the heavy
        loading happens (e.g., show a loading message first).
        """
        if self._is_loaded:
            print("Model already loaded!")
            return

        print("=" * 60)
        print(f"Loading LLM: {self.model_name}")
        print("=" * 60)
        print()
        print("This will download ~15GB on first run (one-time)...")
        print("Subsequent runs will use cached model.")
        print()
        print("Loading model into memory (this takes 10-30 seconds)...")
        print()

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Determine device and dtype
            if self.device == "auto":
                device_map = "auto"
                if torch.cuda.is_available():
                    dtype = torch.float16
                    print("GPU detected! Using GPU acceleration.")
                else:
                    dtype = torch.float32
                    print("No GPU detected. Using CPU (slower).")
            else:
                device_map = self.device
                dtype = torch.float32

            # Load model with optional 8-bit quantization
            if self.load_in_8bit:
                print("Loading with 8-bit quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    load_in_8bit=True,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    dtype=dtype,
                    trust_remote_code=True
                )

            self._is_loaded = True
            print()
            print("Model loaded successfully!")
            print("=" * 60)
            print()

        except Exception as e:
            print(f"Error loading model: {e}")
            print()
            print("Troubleshooting:")
            print("1. Make sure you have internet connection (first download)")
            print("2. Check you have enough RAM (need 10-16GB free)")
            print("3. Try load_in_8bit=True to reduce memory usage")
            raise

    def generate_answer(self, question: str, context: str,
                       chunks_metadata: Optional[List[Dict]] = None,
                       max_new_tokens: int = 512,
                       temperature: float = 0.3) -> Dict[str, any]:
        """
        Generate an answer to a question based on provided context.

        Args:
            question: User's question
            context: Retrieved context from papers
            chunks_metadata: Metadata of retrieved chunks for citations
            max_new_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (0.1-1.0, lower=more focused)

        Returns:
            Dictionary containing:
                - answer: Generated answer
                - raw_output: Full LLM output (for debugging)
                - prompt: The prompt that was sent to LLM
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded! Call load_model() first.")

        # Build the prompt - Focus on clear, comprehensive answers
        system_prompt = """You are a helpful research assistant that explains computer vision concepts clearly and thoroughly.

Your goal: Provide comprehensive, easy-to-understand answers based on the provided research papers.

Guidelines:
- Use simple, clear language that's easy to understand
- Explain technical terms when you use them
- Be thorough - include key details, numbers, and results from the papers
- Structure your answer logically (overview → details → results)
- If the context doesn't have enough information, say so honestly
- Focus on being helpful and informative"""

        user_prompt = f"""Based on these research papers:

{context}

Question: {question}

Provide a clear, comprehensive answer:"""

        # Format messages for chat models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to same device as model
        if torch.cuda.is_available() and self.device != "cpu":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        print("Generating answer...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer (remove the prompt part)
        # The answer is everything after the last user message
        answer = full_output.split("assistant")[-1].strip()

        # Clean up common artifacts
        if answer.startswith(":"):
            answer = answer[1:].strip()
        if answer.startswith("\n"):
            answer = answer.strip()

        return {
            'answer': answer,
            'raw_output': full_output,
            'prompt': prompt,
            'question': question
        }

    def generate_stream(self, question: str, context: str,
                       max_new_tokens: int = 512,
                       temperature: float = 0.3):
        """
        Generate answer with streaming output (yields tokens as they're generated).

        Args:
            question: User's question
            context: Retrieved context from papers
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated tokens one by one
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded! Call load_model() first.")

        # Build prompt (same as generate_answer)
        # Focus on clear, comprehensive answers
        system_prompt = """You are a helpful research assistant that explains computer vision concepts clearly and thoroughly.

Your goal: Provide comprehensive, easy-to-understand answers based on the provided research papers.

Guidelines:
- Use simple, clear language that's easy to understand
- Explain technical terms when you use them
- Be thorough - include key details, numbers, and results from the papers
- Structure your answer logically (overview → details → results)
- If the context doesn't have enough information, say so honestly
- Focus on being helpful and informative"""

        user_prompt = f"""Based on these research papers:

{context}

Question: {question}

Provide a clear, comprehensive answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available() and self.device != "cpu":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate with streaming
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they come
        for text in streamer:
            yield text

        thread.join()

    def format_citations(self, citations: List[Dict]) -> str:
        """
        Format citations in a readable way.

        Args:
            citations: List of citation dictionaries

        Returns:
            Formatted citation string
        """
        if not citations:
            return "No citations"

        formatted = []
        for i, cite in enumerate(citations, 1):
            paper_id = cite.get('paper_id', 'Unknown')
            page = cite.get('page_num', 'Unknown')
            title = cite.get('title', 'Unknown Title')
            formatted.append(
                f"{i}. Paper ID {paper_id}: {title[:60]}... (Page {page})"
            )

        return "\n".join(formatted)

    def unload_model(self):
        """Unload model from memory to free up RAM."""
        if self._is_loaded:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._is_loaded = False
            print("Model unloaded from memory.")
