import os
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "meta-llama/Llama-3.2-3B"
# MODEL_PATH = "Qwen/Qwen2.5-3B"

DEVICE = "cuda"
DTYPE = torch.bfloat16

SEQ_LEN = 1024
RATIO = 0.8


def get_fake_cache(model, seq_len):

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    inputs = tokenizer(
        "hello " * 10000,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len
    ).to(DEVICE)

    with torch.no_grad():
        out = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

    return (
        out.past_key_values,
        out.attentions,
        out.hidden_states
    )


def kv_len(cache):
    # qwen legacy cache
    return cache[0][0].shape[-2]


def test_method(name, cache, attn, hidden, model):

    print("\n====", name, "====")

    before = kv_len(cache)

    if name == "None":
        new_cache = cache

    elif name == "KeepKV":

        from task_evaluation.qwen.merge_keepkv_qwen import Repair_keepkv

        t=time.time()

        new_cache = Repair_keepkv(
            model,
            cache,
            attn,
            hidden,
            RATIO,
            0.8
        )

        cost=time.time()-t


    elif name == "GroupLogitKV":

        from task_evaluation.qwen.merge_grouplogitkv_qwen import Repair_qwen

        t=time.time()

        new_cache = Repair_qwen(
            model,
            cache,
            attn,
            hidden,
            RATIO,
            0.8
        )

        cost=time.time()-t


    elif name == "SimCalKV":

        from task_evaluation.qwen.merge_simcalkv_qwen import Repair

        t=time.time()

        new_cache = Repair(
            model,
            cache,
            attn,
            hidden,
            RATIO,
            -1
        )

        cost=time.time()-t


    elif name == "CaM":

        from task_evaluation.qwen.merge_cam_qwen import Repair as Repair_cam

        t = time.time()

        new_cache = Repair_cam(
            model,
            cache,
            attn,
            hidden,
            compress_ratio=RATIO,
            stochastic=False,
        )

        cost = time.time() - t


    else:
        raise ValueError(name)


    after = kv_len(new_cache)


    print(
        f"""
before={before}
after={after}
actual_ratio={after/before:.4f}
removed={before-after}
time={cost if name!='None' else 0:.4f}s
"""
    )



def main():

    torch.manual_seed(0)

    print("Loading model")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()


    print("Building KV cache")

    cache, attn, hidden = get_fake_cache(
        model,
        SEQ_LEN
    )


    print(
        "original KV length:",
        kv_len(cache)
    )


    for m in [
        "None",
        "GroupLogitKV",
        "KeepKV",
        "SimCalKV",
        "CaM"
    ]:

        test_method(
            m,
            cache,
            attn,
            hidden,
            model
        )


if __name__=="__main__":
    main()