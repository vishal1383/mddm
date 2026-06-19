from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import ConfidenceResult, EncodedExample, ProbeExample


MODEL_IDS = {
    "llada-8b": "GSAI-ML/LLaDA-8B-Instruct",
    "dream-7b": "Dream-org/Dream-v0-Instruct-7B",
}


@dataclass
class MDMProbeModel:
    name: str
    tokenizer: object
    model: object
    mask_token_id: int
    device: str
    logit_shift: int = 0
    prompt_format: str = "auto"

    @classmethod
    def load(
        cls,
        name: str,
        *,
        model_id: str | None = None,
        mask_token_id: int | None = None,
        mask_token: str | None = None,
        device: str = "auto",
        prompt_format: str = "auto",
    ) -> "MDMProbeModel":
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        _patch_transformers_tied_weights_compat()
        resolved_id = model_id or MODEL_IDS.get(name, name)
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        config = AutoConfig.from_pretrained(resolved_id, trust_remote_code=True)
        _patch_remote_model_compat(config, resolved_id)
        tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            resolved_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if resolved_device == "cuda" else torch.float32,
        ).to(resolved_device)
        model.eval()

        resolved_mask_token_id = _mask_id(tokenizer, config, mask_token_id, mask_token)
        return cls(
            name=name,
            tokenizer=tokenizer,
            model=model,
            mask_token_id=resolved_mask_token_id,
            device=resolved_device,
            logit_shift=1 if "dream" in f"{name} {resolved_id}".lower() else 0,
            prompt_format=_resolve_prompt_format(tokenizer, name, resolved_id, prompt_format),
        )

    def encode_example(
        self,
        example: ProbeExample,
        max_completion_tokens: int | None = None,
    ) -> EncodedExample:
        prompt_ids = self._encode_prompt(example.prompt)
        completion_ids = self.tokenizer(example.completion, add_special_tokens=False)["input_ids"]
        if max_completion_tokens is not None:
            completion_ids = completion_ids[:max_completion_tokens]

        return EncodedExample(
            example_id=example.example_id,
            prompt_token_ids=list(map(int, prompt_ids)),
            completion_token_ids=list(map(int, completion_ids)),
            completion_token_texts=[self.tokenizer.decode([t]) for t in completion_ids],
            prompt_text=example.prompt,
            completion_text=example.completion,
            metadata=example.metadata,
            anchor_positions=_anchor_positions(self.tokenizer, completion_ids, self.mask_token_id),
        )

    def compute_p_gt_batch(
        self,
        encoded: EncodedExample,
        anchor_sets: Iterable[Iterable[int]],
        batch_size: int = 1,
    ) -> list[ConfidenceResult]:
        import torch

        anchor_sets = [_clean_anchors(a, encoded.completion_length) for a in anchor_sets]
        target_ids = torch.tensor(encoded.completion_token_ids, device=self.device)
        results: list[ConfidenceResult] = []

        for start in range(0, len(anchor_sets), batch_size):
            batch_anchors = anchor_sets[start : start + batch_size]
            input_ids = [self._input_ids(encoded, anchors) for anchors in batch_anchors]
            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)

            with torch.no_grad():
                # These MDMs attend bidirectionally and our batches have no padding
                # (every row is the same example at one fixed length), so we pass no
                # attention mask. This matches LLaDA's official usage and avoids Dream
                # feeding an int mask into scaled_dot_product_attention.
                outputs = self.model(input_ids=input_ids_t)
                if not hasattr(outputs, "logits"):
                    raise TypeError("Model output has no .logits; use an LM-head model for this probe.")
                logits = outputs.logits
                start_pos = encoded.prompt_length - self.logit_shift
                end_pos = encoded.prompt_length + encoded.completion_length - self.logit_shift
                if start_pos < 0:
                    raise ValueError("logit_shift is larger than the prompt length")
                completion_logits = logits[
                    :,
                    start_pos:end_pos,
                    :,
                ].float()
                log_probs = torch.log_softmax(completion_logits, dim=-1)
                probs = log_probs.exp()
                gold_probs = probs.gather(
                    -1,
                    target_ids.view(1, -1, 1).expand(len(batch_anchors), -1, 1),
                ).squeeze(-1)
                entropy = -(probs * log_probs).sum(dim=-1)

            for row, anchors in enumerate(batch_anchors):
                results.append(
                    ConfidenceResult(
                        anchor_positions=anchors,
                        p_gt=[float(v) for v in gold_probs[row].cpu().tolist()],
                        entropy=[float(v) for v in entropy[row].cpu().tolist()],
                    )
                )
        return results

    def _input_ids(self, encoded: EncodedExample, anchors: tuple[int, ...]) -> list[int]:
        anchor_set = set(anchors)
        completion = [
            token_id if i in anchor_set else self.mask_token_id
            for i, token_id in enumerate(encoded.completion_token_ids)
        ]
        return encoded.prompt_token_ids + completion

    def _encode_prompt(self, prompt: str) -> list[int]:
        if self.prompt_format == "chat":
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
            return _as_list(ids)
        return self.tokenizer(prompt, add_special_tokens=True)["input_ids"]


def load_probe_model(
    model_name: str,
    *,
    model_id: str | None = None,
    mask_token_id: int | None = None,
    mask_token: str | None = None,
    device: str = "auto",
    prompt_format: str = "auto",
) -> MDMProbeModel:
    return MDMProbeModel.load(
        model_name,
        model_id=model_id,
        mask_token_id=mask_token_id,
        mask_token=mask_token,
        device=device,
        prompt_format=prompt_format,
    )


def _clean_anchors(anchor_set: Iterable[int], T: int) -> tuple[int, ...]:
    anchors = tuple(sorted({int(i) for i in anchor_set}))
    if any(i < 0 or i >= T for i in anchors):
        raise ValueError(f"anchor out of range for completion length {T}: {anchors}")
    return anchors


def _as_list(ids) -> list[int]:
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(map(int, ids))


def _patch_rope_init_functions() -> None:
    try:
        from transformers import modeling_rope_utils as rope_utils
    except Exception:
        return
    funcs = getattr(rope_utils, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(funcs, dict) or "default" in funcs:
        return
    funcs["default"] = getattr(
        rope_utils,
        "_compute_default_rope_parameters",
        _compute_default_rope_parameters,
    )


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
    del seq_len
    import torch

    if config is None:
        base = rope_kwargs.get("base", 10000.0)
        dim = rope_kwargs["dim"]
    else:
        base = getattr(config, "rope_theta", 10000.0)
        head_dim = getattr(
            config,
            "head_dim",
            getattr(config, "hidden_size") // getattr(config, "num_attention_heads"),
        )
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv_freq, 1.0


def _patch_transformers_tied_weights_compat() -> None:
    from transformers.modeling_utils import PreTrainedModel

    existing = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if isinstance(existing, property) and existing.fset is not None:
        return

    @property
    def all_tied_weights_keys(self):
        keys = getattr(self, "_mdm_probe_all_tied_weights_keys", None)
        if keys is None:
            keys = getattr(self, "_tied_weights_keys", None) or []
        return _tied_weights_dict(keys)

    @all_tied_weights_keys.setter
    def all_tied_weights_keys(self, value):
        self.__dict__["_mdm_probe_all_tied_weights_keys"] = value

    PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys


def _tied_weights_dict(keys) -> dict:
    if isinstance(keys, dict):
        return keys
    return {key: None for key in keys}


def _patch_remote_generation_config_validate(model_cls) -> None:
    """Newer transformers calls GenerationConfig.validate(user_set_attributes=...),
    but Dream's vendored *GenerationConfig.validate() predates that kwarg. Wrap the
    remote class's validate to ignore unexpected keyword arguments. Scoped to the
    dynamic remote module so base transformers classes are left untouched."""
    import sys

    module = sys.modules.get(getattr(model_cls, "__module__", "") or "")
    if module is None:
        return
    for attr in dir(module):
        if not attr.endswith("GenerationConfig"):
            continue
        gen_cls = getattr(module, attr, None)
        if not isinstance(gen_cls, type):
            continue
        if not getattr(gen_cls, "__module__", "").startswith("transformers_modules"):
            continue
        validate = getattr(gen_cls, "validate", None)
        if validate is None or getattr(validate, "_mdm_probe_compat", False):
            continue

        def make_validate(original):
            def validate(self, *args, **kwargs):
                kwargs.pop("user_set_attributes", None)
                try:
                    return original(self, *args, **kwargs)
                except TypeError:
                    return original(self)

            validate._mdm_probe_compat = True
            return validate

        gen_cls.validate = make_validate(validate)


def _patch_remote_model_compat(config, model_id: str) -> None:
    _patch_rope_init_functions()

    # LLaDA's custom config omits `use_cache`, but modeling_llada.forward reads it.
    # The probe runs single bidirectional passes (no KV cache), so default it off.
    if getattr(config, "use_cache", None) is None:
        config.use_cache = False

    auto_map = getattr(config, "auto_map", {}) or {}
    class_ref = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if isinstance(class_ref, (list, tuple)):
        class_ref = next((ref for ref in reversed(class_ref) if ref), None)
    if not class_ref:
        return
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        try:
            model_cls = get_class_from_dynamic_module(class_ref, model_id, trust_remote_code=True)
        except TypeError:
            model_cls = get_class_from_dynamic_module(class_ref, model_id)
    except Exception:
        return

    _patch_remote_generation_config_validate(model_cls)

    original = getattr(model_cls, "tie_weights", None)
    if original is None or getattr(original, "_mdm_probe_compat", False):
        return

    def tie_weights(self, *args, **kwargs):
        del args, kwargs
        return original(self)

    tie_weights._mdm_probe_compat = True
    model_cls.tie_weights = tie_weights


def _mask_id(tokenizer, config, mask_token_id: int | None, mask_token: str | None) -> int:
    if mask_token_id is not None:
        return int(mask_token_id)
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return int(tokenizer.mask_token_id)
    if getattr(config, "mask_token_id", None) is not None:
        return int(config.mask_token_id)
    if mask_token is not None:
        token_id = tokenizer.convert_tokens_to_ids(mask_token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return int(token_id)
    raise ValueError("Set --mask-token-id or --mask-token for this tokenizer.")


def _resolve_prompt_format(tokenizer, name: str, model_id: str, prompt_format: str) -> str:
    if prompt_format not in {"auto", "raw", "chat"}:
        raise ValueError("prompt_format must be auto, raw, or chat")
    has_template = bool(getattr(tokenizer, "chat_template", None))
    if prompt_format == "chat":
        if not has_template:
            raise ValueError("Tokenizer has no chat_template; use --prompt-format raw.")
        return "chat"
    if prompt_format == "auto":
        return "chat" if has_template and "instruct" in f"{name} {model_id}".lower() else "raw"
    return "raw"


def _anchor_positions(tokenizer, token_ids: list[int], mask_token_id: int) -> list[int]:
    banned = set(getattr(tokenizer, "all_special_ids", []) or [])
    banned.add(int(mask_token_id))
    for name in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "mask_token_id"]:
        token_id = getattr(tokenizer, name, None)
        if token_id is not None:
            banned.add(int(token_id))
    for text in [" ", "\n", "\t", "\r"]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            banned.add(int(ids[0]))
    return [pos for pos, token_id in enumerate(token_ids) if int(token_id) not in banned]
