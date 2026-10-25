"""
Objective 3: Implement Structured Responses, Robust Error Handling,
and Basic API Cost Management.

Key Concepts Covered:
1. Structured Outputs:
   - Utilizing Pydantic models for deterministic, type-safe JSON schema enforcement
   - Automated parsing into Python dataclasses/models
2. Robust Error Handling:
   - Handling Rate Limits (HTTP 429) with Exponential Backoff and Jitter
   - Graceful trapping for Authentication (401/403) and Invalid Argument (400) errors
3. API Cost Management:
   - Multi-tier pricing calculation (Prompt Tokens vs Candidate/Output Tokens)
   - Real-time spend tracking and budget alerts
"""

import os
import time
import json
import random
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# =====================================================================
# 1. Structured Response Schema (Pydantic)
# Matching our ATS / Resume Analysis Theme
# =====================================================================
class ATSResumeEvaluation(BaseModel):
    candidate_name: str = Field(description="Name of the candidate extracted from profile")
    target_role: str = Field(description="Target job title or specialization")
    match_score: int = Field(description="ATS compatibility score from 0 to 100", ge=0, le=100)
    strengths: List[str] = Field(description="Top 3 technical or professional strengths")
    missing_keywords: List[str] = Field(description="Crucial industry keywords missing from profile")
    recommended_improvements: List[str] = Field(description="Actionable bullet point suggestions")
    summary: str = Field(description="High-level 2-sentence executive summary")


# =====================================================================
# 2. API Cost Calculator & Budget Tracker
# =====================================================================
class CostManager:
    """
    Tracks token consumption and calculates exact USD costs across models.
    """
    # Pricing per 1M tokens (USD) - Current benchmark rates
    MODEL_PRICING = {
        "gemini-3.8-flash": {"input_per_million": 0.15, "output_per_million": 0.60},
        "gemini-3.5-flash-lite": {"input_per_million": 0.075, "output_per_million": 0.30},
        "gemini-4-argon": {"input_per_million": 1.25, "output_per_million": 5.00},
        "gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.60},
    }

    def __init__(self, budget_limit_usd: float = 1.00):
        self.budget_limit_usd = budget_limit_usd
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def calculate_cost(self, model: str, prompt_tokens: int, output_tokens: int) -> float:
        """Computes cost for a single transaction."""
        pricing = self.MODEL_PRICING.get(
            model, {"input_per_million": 0.15, "output_per_million": 0.60}
        )
        input_cost = (prompt_tokens / 1_000_000.0) * pricing["input_per_million"]
        output_cost = (output_tokens / 1_000_000.0) * pricing["output_per_million"]
        return input_cost + output_cost

    def record_usage(self, model: str, prompt_tokens: int, output_tokens: int) -> dict:
        """Records token usage and updates running balance with budget alerts."""
        cost = self.calculate_cost(model, prompt_tokens, output_tokens)
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        budget_warning = self.total_cost_usd >= (self.budget_limit_usd * 0.8)
        budget_exceeded = self.total_cost_usd >= self.budget_limit_usd

        return {
            "transaction_cost_usd": cost,
            "cumulative_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "budget_warning": budget_warning,
            "budget_exceeded": budget_exceeded,
        }


# =====================================================================
# 3. Robust Error Handling & Retry Logic
# =====================================================================
def execute_with_retry(func, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Executes an LLM API call with Exponential Backoff and Full Jitter
    to gracefully handle transient rate limits (429) or connection hiccups.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "quota" in err_str or "rate limit" in err_str
            is_transient = "503" in err_str or "timeout" in err_str or "connection" in err_str

            if (is_rate_limit or is_transient) and attempt < max_retries:
                # Exponential backoff: base * (2 ^ attempt) + random jitter
                backoff = initial_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                print(f"[!] Transient error encountered ({e}). Retrying in {backoff:.2f}s (Attempt {attempt}/{max_retries})...")
                time.sleep(backoff)
            else:
                # Unrecoverable error (e.g. 401 Unauthorized or exhausted retries)
                print(f"[-] Execution halted after attempt {attempt}. Error: {e}")
                raise e


# =====================================================================
# 4. Structured Output Generation Function
# =====================================================================
def generate_structured_ats_review(
    raw_resume_text: str,
    target_role: str = "Senior AI Engineer",
    model: str = "gemini-3.8-flash",
    cost_manager: Optional[CostManager] = None
) -> ATSResumeEvaluation:
    """
    Uses Gemini API structured output enforcement via Pydantic schema.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key and api_key != "your_gemini_api_key_here":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = f"""
        Analyze the following candidate resume for the role: {target_role}.
        Return a strict structured assessment matching the required schema.

        Resume Content:
        {raw_resume_text}
        """

        def api_call():
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ATSResumeEvaluation,
                temperature=0.1,  # Low temperature for strict factual evaluation
            )
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

        response = execute_with_retry(api_call, max_retries=3)

        # Track Cost
        if cost_manager and hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_toks = getattr(response.usage_metadata, "prompt_token_count", 0) or 300
            resp_toks = getattr(response.usage_metadata, "candidates_token_count", 0) or 200
            cost_manager.record_usage(model, prompt_toks, resp_toks)

        parsed_json = json.loads(response.text)
        return ATSResumeEvaluation(**parsed_json)

    else:
        # High-fidelity simulation mode for testing & educational study
        print("[*] Simulating structured Pydantic response (Set GEMINI_API_KEY in .env for live API call)...")
        if cost_manager:
            cost_manager.record_usage(model, 350, 180)

        return ATSResumeEvaluation(
            candidate_name="Alex Mercer",
            target_role=target_role,
            match_score=88,
            strengths=[
                "Strong Python and PyTorch foundation",
                "Hands-on experience deploying LLM APIs and containerized microservices",
                "Demonstrated optimization of inference latency and token overhead"
            ],
            missing_keywords=[
                "Retrieval-Augmented Generation (RAG)",
                "Vector Databases (Pinecone/Milvus)",
                "Kubernetes Orchestration"
            ],
            recommended_improvements=[
                "Quantify business impact on data pipelines (e.g., reduced query latency by 35%)",
                "Highlight prompt engineering evaluation benchmarks (e.g., Ragas, TruLens)",
                "Add explicit mention of modern SDKs (google-genai, openai)"
            ],
            summary="Candidate shows strong foundational AI engineering and API integration skills with clear project deliverables."
        )


def main():
    print("=" * 70)
    print("  Objective 3: Structured Responses, Error Handling & Cost Tracking")
    print("=" * 70)

    cost_tracker = CostManager(budget_limit_usd=0.05)

    sample_resume = """
    Alex Mercer - Software & AI Developer
    Email: alex.mercer@example.com | GitHub: github.com/alex-mercer

    Professional Summary:
    Full-stack developer with 3+ years experience building backend microservices and integrating
    large language model APIs. Skilled in Python, FastAPI, Docker, and prompt engineering.

    Experience:
    AI Software Engineer | Nexus Tech (2024 - Present)
    - Integrated LLM APIs (Gemini, OpenAI) for automated customer support triage.
    - Reduced token consumption by 40% using sliding-window history pruning.
    - Built structured output pipelines with Pydantic for deterministic JSON payloads.

    Skills:
    Python, REST APIs, Git, Docker, PostgreSQL, LLM Integrations, Linux.
    """

    print("[*] Evaluating Resume with Structured Pydantic Schema...")
    review = generate_structured_ats_review(
        raw_resume_text=sample_resume,
        target_role="Lead AI Application Engineer",
        cost_manager=cost_tracker
    )

    print("\n--- Structured Output Result (Validated Pydantic Object) ---")
    print(f"Candidate   : {review.candidate_name}")
    print(f"Target Role : {review.target_role}")
    print(f"Match Score : {review.match_score}/100")
    print("\nTop Strengths:")
    for s in review.strengths:
        print(f"  + {s}")
    print("\nMissing Keywords:")
    for k in review.missing_keywords:
        print(f"  - {k}")
    print("\nRecommended Improvements:")
    for imp in review.recommended_improvements:
        print(f"  * {imp}")
    print(f"\nExecutive Summary: {review.summary}")

    print("\n" + "-" * 70)
    print("API Cost & Budget Monitoring:")
    print(f"Total Input Tokens  : {cost_tracker.total_input_tokens}")
    print(f"Total Output Tokens : {cost_tracker.total_output_tokens}")
    print(f"Total Session Spend : ${cost_tracker.total_cost_usd:.6f} USD")
    print(f"Budget Limit        : ${cost_tracker.budget_limit_usd:.2f} USD")
    print(f"Budget Status       : {'Within limits' if cost_tracker.total_cost_usd < cost_tracker.budget_limit_usd else 'EXCEEDED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
