"""Install KeepKV runtime patch for Hugging Face Qwen2/Qwen2.5.

Supports both Transformers layouts:
1. module-level eager_attention_forward (newer releases), and
2. attention computation inside Qwen2Attention.forward (older releases).
"""
from models.common.keepkv_runtime_common import (
    keepkv_eager_attention_forward,
    make_class_forward_wrapper,
)

_PATCHED = False
_PATCH_KIND = None
_ORIGINAL = None


def apply_keepkv_qwen_runtime_patch():
    global _PATCHED, _PATCH_KIND, _ORIGINAL
    if _PATCHED:
        return

    from transformers.models.qwen2 import modeling_qwen2

    if hasattr(modeling_qwen2, "eager_attention_forward"):
        _ORIGINAL = modeling_qwen2.eager_attention_forward
        modeling_qwen2.eager_attention_forward = keepkv_eager_attention_forward
        _PATCH_KIND = "module_function"
    elif hasattr(modeling_qwen2, "Qwen2Attention"):
        _ORIGINAL = modeling_qwen2.Qwen2Attention.forward
        modeling_qwen2.Qwen2Attention.forward = make_class_forward_wrapper(_ORIGINAL)
        _PATCH_KIND = "class_forward"
    else:
        raise RuntimeError(
            "Unsupported Transformers Qwen2 implementation: neither "
            "modeling_qwen2.eager_attention_forward nor Qwen2Attention was found."
        )

    _PATCHED = True


def remove_keepkv_qwen_runtime_patch():
    global _PATCHED, _PATCH_KIND, _ORIGINAL
    if not _PATCHED:
        return

    from transformers.models.qwen2 import modeling_qwen2

    if _PATCH_KIND == "module_function":
        modeling_qwen2.eager_attention_forward = _ORIGINAL
    elif _PATCH_KIND == "class_forward":
        modeling_qwen2.Qwen2Attention.forward = _ORIGINAL

    _PATCHED = False
    _PATCH_KIND = None
    _ORIGINAL = None
