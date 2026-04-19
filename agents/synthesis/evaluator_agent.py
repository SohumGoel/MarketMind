"""
Evaluator agent — final inference stage of the MarketMind pipeline.

Supports three backends:
  1. CMU AI Gateway (OpenAI-compatible) — recommended for demo, no GPU needed
  2. Local finetuned checkpoint — for ablation comparison
  3. HuggingFace Hub checkpoint — remote model
"""

import os
import torch
from evaluation.metrics import extract_direction_from_output


_GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
_DEFAULT_GATEWAY_MODEL = "claude-sonnet-4-20250514-v1:0"
_GATEWAY_MODEL_FALLBACKS = [
    "claude-opus-4-20250514-v1:0",
    "gpt-5.4",
    "gemini-2.5-pro",
]

_SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Your task is to list the positive "
    "developments and potential concerns for companies based on relevant news and "
    "basic financial data from the past weeks, then make a prediction about the "
    "companies' stock price movement for the upcoming week.\n\n"
    "Your answer format must be exactly:\n\n"
    "[Positive Developments]:\n1. ...\n\n"
    "[Potential Concerns]:\n1. ...\n\n"
    "[Prediction & Analysis]:\n"
    "Conclude with whether the stock will move **up**, **down**, or remain **neutral**."
)


class EvaluatorAgent:
    """
    Wraps model inference for single-ticker analysis.

    Usage (gateway — recommended for demo):
        agent = EvaluatorAgent(backend="gateway", api_key="your_cmu_key")
        result = agent.predict(prompt_dict)

    Usage (local finetuned model):
        agent = EvaluatorAgent(backend="local", model_path="/path/to/checkpoint")
        result = agent.predict(prompt_dict)
    """

    def __init__(
        self,
        backend: str = "gateway",
        api_key: str = None,
        model_path: str = None,
        gateway_model: str = _DEFAULT_GATEWAY_MODEL,
        device: str = "auto",
    ):
        self.backend = backend

        if backend == "gateway":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.environ.get("CMU_AI_GATEWAY_KEY"),
                base_url=_GATEWAY_BASE_URL,
            )
            self.gateway_model = gateway_model

        elif backend in ("local", "huggingface"):
            if not model_path:
                raise ValueError("model_path required for local/huggingface backend")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading model from {model_path} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            print("Model loaded.")

        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'gateway', 'local', or 'huggingface'.")

    def predict(self, prompt_dict: dict, max_new_tokens: int = 600) -> dict:
        """
        Run inference given a prompt dict from agents/pipeline.build_prompt().

        Args:
            prompt_dict:    Output of build_prompt() — must contain 'prompt' key
            max_new_tokens: Max tokens to generate

        Returns:
            dict with keys: ticker, start_date, end_date, direction, reasoning
        """
        if self.backend == "gateway":
            full_output = self._predict_gateway(prompt_dict, max_new_tokens)
        else:
            full_output = self._predict_local(prompt_dict, max_new_tokens)

        direction = extract_direction_from_output(full_output)

        return {
            "ticker":     prompt_dict.get("ticker"),
            "start_date": prompt_dict.get("start_date"),
            "end_date":   prompt_dict.get("end_date"),
            "direction":  direction,
            "reasoning":  full_output,
        }

    def _predict_gateway(self, prompt_dict: dict, max_tokens: int) -> str:
        user_content = _extract_user_content(prompt_dict["prompt"])
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        models_to_try = [self.gateway_model] + _GATEWAY_MODEL_FALLBACKS
        last_exc = None
        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                continue
        raise RuntimeError(f"All gateway models failed. Last error: {last_exc}")

    def _predict_local(self, prompt_dict: dict, max_new_tokens: int) -> str:
        inputs = self.tokenizer(
            prompt_dict["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _extract_user_content(chatml_prompt: str) -> str:
    """Pull out the user turn content from a ChatML-formatted string."""
    import re
    match = re.search(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
        chatml_prompt,
        re.DOTALL,
    )
    return match.group(1).strip() if match else chatml_prompt
