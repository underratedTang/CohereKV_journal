"""Install KeepKV runtime patch for Hugging Face LLaMA.

Supports both module-level and class-based eager attention implementations.
"""
from models.common.keepkv_runtime_common import (
    keepkv_eager_attention_forward,
    make_class_forward_wrapper,
)

_PATCHED = False
_PATCH_KIND = None
_ORIGINAL = None


def apply_keepkv_runtime_patch():
    global _PATCHED, _PATCH_KIND, _ORIGINAL
    if _PATCHED:
        return

    from transformers.models.llama import modeling_llama

    if hasattr(modeling_llama, "eager_attention_forward"):
        _ORIGINAL = modeling_llama.eager_attention_forward
        modeling_llama.eager_attention_forward = keepkv_eager_attention_forward
        _PATCH_KIND = "module_function"
    elif hasattr(modeling_llama, "LlamaAttention"):
        _ORIGINAL = modeling_llama.LlamaAttention.forward
        modeling_llama.LlamaAttention.forward = make_class_forward_wrapper(_ORIGINAL)
        _PATCH_KIND = "class_forward"
    else:
        raise RuntimeError(
            "Unsupported Transformers LLaMA implementation: neither "
            "modeling_llama.eager_attention_forward nor LlamaAttention was found."
        )

    _PATCHED = True


def remove_keepkv_runtime_patch():
    global _PATCHED, _PATCH_KIND, _ORIGINAL
    if not _PATCHED:
        return

    from transformers.models.llama import modeling_llama

    if _PATCH_KIND == "module_function":
        modeling_llama.eager_attention_forward = _ORIGINAL
    elif _PATCH_KIND == "class_forward":
        modeling_llama.LlamaAttention.forward = _ORIGINAL

    _PATCHED = False
    _PATCH_KIND = None
    _ORIGINAL = None
