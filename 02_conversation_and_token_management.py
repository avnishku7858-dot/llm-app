"""
Objective 2: Manage Conversations, Chat History, System/User/Assistant Messages,
and Optimize Model Parameters and Token Usage.

Key Concepts Covered:
1. Message Roles: System Instruction (Persona & Boundaries), User (Prompts), Assistant/Model (Responses).
2. Chat History Management: Stateful multi-turn conversation loops.
3. Hyperparameter Optimization:
   - Temperature: 0.0 (Deterministic, Code/Math) vs 1.0+ (Creative, Brainstorming)
   - Top-P (Nucleus Sampling) & Top-K: Probability filtering for token selection
   - Max Output Tokens: Hard cutoff to constrain length and budget
4. Token Counting & Optimization:
   - Tracking prompt tokens vs candidate/output tokens
   - Context Window Pruning (Sliding Window Strategy to keep within limits)
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class ConversationManager:
    """
    Manages multi-turn conversation memory, context pruning,
    and parameter optimization for LLM interactions.
    """

    def __init__(
        self,
        system_instruction: str = "You are a helpful, concise AI technical mentor.",
        max_history_turns: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_output_tokens: int = 500,
    ):
        self.system_instruction = system_instruction
        self.max_history_turns = max_history_turns
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

        # History stores list of {"role": "user"|"assistant", "content": "..."}
        self.history: List[Dict[str, str]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add_message(self, role: str, content: str):
        """Appends a new turn to the history."""
        self.history.append({"role": role, "content": content})

    def prune_history(self) -> List[Dict[str, str]]:
        """
        Implements Sliding Window context management.
        Keeps only the most recent N turns to prevent context blowup and runaway token usage.
        """
        # Each turn is a pair of (user, assistant), so max messages = max_history_turns * 2
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            # Retain the most recent messages
            return self.history[-max_messages:]
        return self.history

    def estimate_tokens(self, text: str) -> int:
        """
        Heuristic estimation: ~4 characters per token for English text.
        Useful for pre-flight budget checks before making network calls.
        """
        return max(1, len(text) // 4)

    def get_context_stats(self) -> Dict[str, Any]:
        """Calculates current context token footprint."""
        pruned = self.prune_history()
        history_text = " ".join([m["content"] for m in pruned])
        system_tokens = self.estimate_tokens(self.system_instruction)
        history_tokens = self.estimate_tokens(history_text)
        return {
            "total_messages_stored": len(self.history),
            "active_messages_in_context": len(pruned),
            "estimated_system_tokens": system_tokens,
            "estimated_history_tokens": history_tokens,
            "estimated_total_prompt_tokens": system_tokens + history_tokens,
        }

    def chat_turn(self, user_prompt: str, client=None, model: str = "gemini-3.8-flash") -> str:
        """
        Executes one full conversation turn with system instruction,
        pruned history, and parameter controls.
        """
        # 1. Record User Message
        self.add_message("user", user_prompt)
        pruned_history = self.prune_history()

        api_key = os.getenv("GEMINI_API_KEY")

        if client and api_key and api_key != "your_gemini_api_key_here":
            try:
                # Use Google GenAI Client with config parameters
                from google.genai import types

                config = types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_output_tokens,
                )

                # Format multi-turn contents for the SDK
                contents = []
                for msg in pruned_history:
                    contents.append(
                        types.Content(
                            role="user" if msg["role"] == "user" else "model",
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )

                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                assistant_reply = response.text or "No response received."

                # Update token usage from API metadata if available
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    self.total_prompt_tokens += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    self.total_completion_tokens += getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                else:
                    self.total_prompt_tokens += self.estimate_tokens(user_prompt)
                    self.total_completion_tokens += self.estimate_tokens(assistant_reply)

            except Exception as e:
                assistant_reply = f"[API Error: {e}]"
        else:
            # Educational fallback simulation if no API key is provided
            est_tokens = self.estimate_tokens(user_prompt)
            assistant_reply = (
                f"[Simulated AI ({model}) | temp={self.temperature} | max_tokens={self.max_output_tokens}]: "
                f"Acknowledging: '{user_prompt[:40]}...'. Persona applied: '{self.system_instruction[:30]}...'"
            )
            self.total_prompt_tokens += est_tokens
            self.total_completion_tokens += self.estimate_tokens(assistant_reply)

        # 2. Record Assistant Message in history
        self.add_message("assistant", assistant_reply)
        return assistant_reply


def main():
    print("=" * 70)
    print("  Objective 2: Conversation Management, History & Token Optimization")
    print("=" * 70)

    # Initialize manager with custom persona & optimized parameters
    system_persona = (
        "You are an expert Python tutor. Answer concisely in under 2 sentences with code snippets."
    )
    manager = ConversationManager(
        system_instruction=system_persona,
        max_history_turns=3,       # Sliding window: keep only last 3 exchanges
        temperature=0.2,           # Low temperature for precise, deterministic technical answers
        top_p=0.9,
        max_output_tokens=200      # Prevent verbose token exhaustion
    )

    print(f"[Config] System Instruction: {manager.system_instruction}")
    print(f"[Config] Temperature: {manager.temperature} (Focused / Low Randomness)")
    print(f"[Config] Max Output Tokens: {manager.max_output_tokens}")
    print(f"[Config] Sliding Window: Last {manager.max_history_turns} turns retained")
    print("-" * 70)

    # Optional Live Client
    api_key = os.getenv("GEMINI_API_KEY")
    client = None
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            print("[+] Live Gemini Client loaded for multi-turn chat.")
        except Exception as e:
            print(f"[-] Could not initialize live client: {e}")
    else:
        print("[*] Running in educational simulation mode (Set GEMINI_API_KEY in .env for live calls).")

    print("-" * 70)

    # Multi-Turn Simulation / Live Demonstration
    test_queries = [
        "What is a Python list comprehension?",
        "Can you show an example filtering only even numbers?",
        "Now convert that into a dictionary comprehension where key is the number and value is its square.",
        "What was my very first question?",  # Tests conversational memory
    ]

    for turn_idx, query in enumerate(test_queries, 1):
        print(f"\n--- Turn {turn_idx} ---")
        print(f"User: {query}")
        reply = manager.chat_turn(query, client=client)
        print(f"AI  : {reply}")

        stats = manager.get_context_stats()
        print(f"[Context Footprint] Stored msgs: {stats['total_messages_stored']} | "
              f"Active in window: {stats['active_messages_in_context']} | "
              f"Est prompt tokens: {stats['estimated_total_prompt_tokens']}")

    print("\n" + "=" * 70)
    print("Token Optimization Summary:")
    print(f"Cumulative Prompt Tokens    : {manager.total_prompt_tokens}")
    print(f"Cumulative Completion Tokens: {manager.total_completion_tokens}")
    print(f"Total Tokens Consumed       : {manager.total_prompt_tokens + manager.total_completion_tokens}")
    print("Sliding window successfully pruned older exchanges to maintain token budgets!")
    print("=" * 70)


if __name__ == "__main__":
    main()
