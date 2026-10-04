"""Local Qwen LoRA policy utilities for a real Minecraft rollout.

This module can load a PEFT policy and sample typed actions, but it is *not* a
GRPO trainer.  Strict GRPO needs the rollout engine (veRL) to retain the exact
prompt/completion token ids and behaviour-policy log probabilities, then apply
the clipped token-level objective against a reference policy.  Re-encoding a
parsed JSON action with the current model is not equivalent: it changes the
tokenisation boundary, loses the sampled completion, and uses a post-update
log-probability as the old-policy term.  ``update`` therefore fails closed so
the real runner cannot silently produce a pseudo-GRPO checkpoint.

The API-only Qwen server is deliberately not used here because it cannot
receive gradients.  Use the official veRL trainer for train-time updates and
keep this class for model loading/checkpoint interoperability and rollout
smoke tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class QwenLoRAPolicy:
    def __init__(self, model_path: str, checkpoint: str | None = None,
                 *, device: str = "auto", max_input_tokens: int = 1536,
                 max_new_tokens: int = 128, lr: float = 1e-5,
                 lora_r: int = 32, lora_alpha: int = 32,
                 temperature: float = 0.8, top_p: float = 0.95,
                 clip_ratio: float = 0.2):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        self.torch = torch
        self.model_path = model_path
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        # Kept for checkpoint/config compatibility.  This local class does not
        # implement the GRPO clipping objective; veRL owns that computation.
        self.clip_ratio = clip_ratio
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        map_location = device if device != "auto" else ("auto" if torch.cuda.is_available() else None)
        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if map_location is not None:
            kwargs["device_map"] = map_location
        base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if checkpoint:
            self.model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
        else:
            config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                                lora_dropout=0.05, bias="none",
                                task_type=TaskType.CAUSAL_LM,
                                target_modules="all-linear")
            self.model = get_peft_model(base, config)
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _prompt(observation: Mapping[str, Any], agent: str) -> str:
        return ("You are a Minecraft agent. Emit exactly one JSON object with "
                "keys type and typed action arguments. Never claim success without "
                "an executable action.\nagent=" + agent + "\nobservation=" +
                json.dumps(observation, ensure_ascii=False, sort_keys=True))

    def _encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=self.max_input_tokens).to(self.device)

    def sample_action(self, observation: Mapping[str, Any], agent: str, *, deterministic=False):
        prompt = self._prompt(observation, agent)
        inputs = self._encode_prompt(prompt)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                         do_sample=not deterministic,
                                         temperature=self.temperature if not deterministic else 1.0,
                                         top_p=self.top_p if not deterministic else 1.0,
                                         return_dict_in_generate=True,
                                         output_scores=True,
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = output.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {"type": "observe", "_valid": False}
        try:
            action = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {"type": "observe", "_valid": False}
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            return {"type": "observe", "_valid": False}
        action["_valid"] = True
        if output.scores:
            # These are generation-time scores, retained only as rollout
            # diagnostics.  They must not be fed back as GRPO ``old_logprob``:
            # official veRL obtains behaviour-policy token log-probs from the
            # same sampled token sequence and stores them in its trajectory.
            score_logp = []
            for logits, token in zip(output.scores, generated):
                score_logp.append(self.torch.log_softmax(logits[0], dim=-1)[int(token)])
            if score_logp:
                action["_sampling_logprob_sum"] = float(self.torch.stack(score_logp).sum().cpu())
                action["_sampling_logprobs"] = [float(value.detach().cpu()) for value in score_logp]
        # Preserve the sampled completion for an adapter that hands the
        # trajectory to veRL.  ``villageragent_real`` strips private metadata
        # before invoking a Minecraft tool, so these fields never reach the
        # environment as action arguments.
        action["_prompt"] = prompt
        action["_completion_text"] = text
        action["_completion_token_ids"] = [int(token) for token in generated.detach().cpu().tolist()]
        return action

    def update(self, trajectories: Sequence[Mapping[str, Any]], advantages: Sequence[float]):
        """Reject the generic learner hook instead of running pseudo-GRPO.

        The repository-level coordinator exposes a small ``policy.update``
        hook for contract tests.  Calling it with this policy would tempt a
        caller to treat a re-encoded JSON teacher-forcing loss as GRPO.  That
        objective is not valid for a real Minecraft experiment, so fail with a
        direct remediation message.  Configure the official veRL GRPO trainer
        with this model/checkpoint instead.
        """
        del trajectories, advantages
        raise RuntimeError(
            "QwenLoRAPolicy.update is disabled: this class is a rollout/model "
            "adapter, not a strict GRPO trainer. Use the official veRL GRPO "
            "worker so sampled completion token ids, behaviour-policy "
            "log-probs, reference-policy KL, and world-reset groups are "
            "computed from the same real rollout."
        )

    def save_checkpoint(self, path: Path, metadata: Mapping[str, Any]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "rl_metadata.json").write_text(json.dumps(dict(metadata), indent=2, sort_keys=True))

    def load_checkpoint(self, path: Path):
        from peft import set_peft_model_state_dict
        del set_peft_model_state_dict
        self.model.load_adapter(str(path), adapter_name="default", is_trainable=False)
        self.model.eval()
