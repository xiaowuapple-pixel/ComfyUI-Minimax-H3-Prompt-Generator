"""Qwen-Image-2.1 prompt enhancer node.

Thin ComfyUI wrapper around the official prompt-rewrite contract:
https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite

The two prompt-enhancer checkpoints (Qwen-Image-2.1-PE-T2I / -PE-I2I) are
fine-tuned Qwen3.5-VL 9B models. They turn a short request into the long prompt
the 2.1 model expects, and they answer with a small JSON object that also
carries the canvas decision (wh_ratio / ratio_follow).

Task profiles, the thinking split and the answer parser mirror
`prompt_rewrite/pe_core.py`. The task-specific system prompts are the ones that
ship inside the checkpoints, kept next to this file so they cannot drift away
from the weights they belong to.
"""

from __future__ import annotations

import json
import inspect
import hashlib
import math
import os
import random
import re
import time

import numpy as np

from .nodes import (
    CONTEXT_LENGTH_OPTIONS,
    DEFAULT_CONTEXT_LENGTH,
    _VisionRuntime,
    _completion_budget,
    _input_value,
    _language_models,
    _tensor_to_data_url,
    _vision_models,
)

# json_repair fixes answers that are almost valid JSON (trailing comma, unescaped
# quote). Optional: the balanced-brace scan below already handles well-formed
# answers, it just gives up a little sooner.
try:
    import json_repair  # type: ignore
except ImportError:
    json_repair = None


PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pe_prompts")

# Qwen-Image-2.1 accepts up to ten reference images. The frontend shows one empty
# socket at a time and grows the list as you connect, so a fresh node stays small.
MAX_INPUT_IMAGES = 10

# Which runtime turns the request into text. The local safetensors path uses
# ComfyUI's own text-encoder inference (that is what the Comfy-Org repacks of the
# PE models are for), so it gets the same int8/kv-cache handling as everything
# else in the graph.
SOURCE_GGUF = "Local GGUF"
SOURCE_AUTO = "Local safetensors (auto)"

TASK_AUTO = "Auto (by images)"

# The official contract lets the model choose the canvas; this lets you override
# it. Forcing a ratio changes two things: the model is told about it (so the
# composition it describes matches the frame) and the Width/Height outputs use it.
ASPECT_AUTO = "Auto (model decides)"
ASPECT_OPTIONS = [ASPECT_AUTO, "1:1", "4:3", "3:2", "16:9", "21:9", "9:16", "3:4", "2:3"]

# The enhancer consumes whatever the loader picked, so the weights and the
# sampling knobs each live on their own node instead of crowding the one you
# actually run every time.
PE_MODEL_TYPE = "QWE_PE_MODEL"
PE_SETTINGS_TYPE = "QWE_PE_SETTINGS"

SAMPLING_PRESET_DEFAULT = "Official defaults"

_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# How the assistant turn is pre-filled when the plan is switched off.
#
# These models were trained to plan first and answer second, and an empty think
# block does not reliably stop that: measured on a five-image edit request, the
# model ignored the pre-closed block and wrote a 5930-token plan anyway (83 s at
# 76 tok/s). Opening the answer for it settles the question -- the same request
# then produced 404 tokens in 9.6 s, at full speed and with the JSON contract
# intact. Forcing the same thing with a GBNF grammar also worked but crushed
# decoding to 9 tok/s, so the nudge is the one to keep.
ANSWER_PREFILL = '{"rewritten_prompt": "'
# The native path renders its template with `str.format`, so a literal brace has
# to be doubled there or the tokenizer raises "unmatched '{' in format spec".
ANSWER_PREFILL_TEMPLATE = ANSWER_PREFILL.replace("{", "{{").replace("}", "}}")

# Sampling values are the production inference settings of each task, copied from
# pe_core.py. They are not interchangeable: presence_penalty is 1.5 for t2i and
# 0 for edit, and a wrong penalty does not fail loudly, it quietly changes the
# distribution you sample from.
PE_PROFILES = {
    "t2i": {
        "label": "Text to Image (t2i)",
        "prompt_file": "system_prompt_t2i.txt",
        "takes_images": False,
        "has_ratio_follow": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_new_tokens": 16256,
        "image_max_pixels": 1024 * 1024,
    },
    "edit": {
        "label": "Image Edit (edit)",
        "prompt_file": "system_prompt_edit.txt",
        "takes_images": True,
        "has_ratio_follow": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_new_tokens": 24000,
        "image_max_pixels": 1024 * 1024,
    },
}

TASK_LABELS = [profile["label"] for profile in PE_PROFILES.values()]
_LABEL_TO_TASK = {profile["label"]: key for key, profile in PE_PROFILES.items()}
ALL_TASK_LABELS = [TASK_AUTO] + TASK_LABELS


def _task_key(label):
    return _LABEL_TO_TASK.get(label, "t2i")


def _resolve_task(label, has_images):
    """Pick the task. Auto reads the graph: images in means an edit request.

    'edit' with no image and 't2i' with images are both errors in the official
    tooling, so the presence of images is the whole decision -- there is nothing
    else the model could be asked to do.
    """
    if label == TASK_AUTO:
        return "edit" if has_images else "t2i"
    return _task_key(label)


def _text_encoder_choices():
    import folder_paths

    return folder_paths.get_filename_list("text_encoders") or ["No text encoders found"]


def _pe_language_models():
    """Only PE checkpoints are offered for the local GGUF path.

    A stock Qwen3.5 (or any other instruct model) does follow the system prompt's
    structure, but it was never trained against its answer contract, so it cannot
    produce the JSON this node hands downstream. Listing it only invites a run
    that costs minutes and returns a fallback. If no PE checkpoint is installed
    yet the full list is shown, so the node stays usable while you download one.
    """
    import folder_paths

    models = list(_language_models())
    try:
        # Comfy-Org ships the PE checkpoints as text encoders, so a GGUF dropped
        # next to them should be found too, not only the ones under models/LLM.
        models += [
            name
            for name in folder_paths.get_filename_list("text_encoders")
            if name.lower().endswith(".gguf")
        ]
    except Exception:
        pass
    unique = sorted(set(models))
    pe = [name for name in unique if "pe" in os.path.basename(name).lower()]
    return pe or unique


def _default_encoder(names, marker):
    for name in names:
        if marker in name.lower():
            return name
    return names[0] if names else "No text encoders found"


_ENCODER_CACHE = {}


def _release_encoder(name=None):
    """Drop a loaded PE encoder and hand the VRAM back.

    Dropping the last reference is enough -- measured 14.6 GB -> 1.4 GB with
    nothing else resident -- so this deliberately does NOT call
    `unload_all_models()`, which would also evict the diffusion model the rest of
    the workflow is about to use.
    """
    import gc

    import comfy.model_management

    if name is not None and name in _ENCODER_CACHE:
        _ENCODER_CACHE.pop(name, None)
    else:
        _ENCODER_CACHE.clear()
    gc.collect()
    comfy.model_management.soft_empty_cache()


def _load_encoder(filename):
    """Load one PE encoder, keeping at most one of them resident.

    The two checkpoints are 8.8 GB each, so holding both would not fit a 16 GB
    card. Only the one this run needs is loaded, and switching tasks drops the
    other one first.
    """
    import comfy.sd
    import folder_paths

    for other in [name for name in _ENCODER_CACHE if name != filename]:
        _release_encoder(other)
    cached = _ENCODER_CACHE.get(filename)
    if cached is not None:
        return cached

    path = folder_paths.get_full_path_or_raise("text_encoders", filename)
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
    )
    _ENCODER_CACHE[filename] = clip
    return clip


def _load_system_prompt(task, override_path):
    """Resolve the system prompt: an explicit file wins, else the bundled copy.

    The answer contract is part of what the weights were trained on, so the
    prompt has to travel with the checkpoint. Swapping tasks and forgetting to
    swap the prompt fails silently: fluent output with the wrong contract.

    A missing override falls back to the bundled prompt instead of refusing to
    run. Saved workflows break this field by no fault of their own: the loaders
    gained and lost widgets over time, and a workflow stores widget values by
    position, so an older file can hand a Context Length value to this one
    (measured: "24576"). Falling back keeps that workflow working, and the
    warning says where the stray value came from.
    """
    path = (override_path or "").strip()
    if path and not os.path.isfile(path):
        print(
            f"[Qwen Image 2.1 PE] 警告：System Prompt File 指定的文件不存在（{path}），"
            "已改用官方的系统提示词。这种情况通常来自旧工作流：加载节点上的控件顺序变过，"
            "把加载节点删掉、重新添加一个就对上了。"
        )
        path = ""
    if not path:
        path = os.path.join(PROMPT_DIR, PE_PROFILES[task]["prompt_file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"系统提示词文件不存在：{path}")
    with open(path, encoding="utf-8") as handle:
        prompt = handle.read().strip()
    # Logged on every run so the question "did the rules actually reach the model"
    # can be answered from the console instead of from the source.
    print(
        f"[Qwen Image 2.1 PE] 系统提示词：{os.path.basename(path)}"
        f"（{len(prompt)} 字符，sha256 {hashlib.sha256(prompt.encode()).hexdigest()[:12]}）"
    )
    return prompt


def _split_thinking(text):
    """Split a decoded answer into (thinking, answer).

    The chat template pre-fills `<think>` before generation, so the decoded text
    normally starts inside the thinking block and closes it with `</think>`.
    """
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        if "<think>" in think:
            think = think.partition("<think>")[2]
        return think.strip(), answer.strip()
    if "<think>" in text:
        # Unterminated thinking block: the generation hit the token budget.
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


def _balanced_spans(answer):
    """Every balanced top-level ``{...}`` span, in order.

    A single greedy `\\{.*\\}` is not enough: a brace in the prose after the
    object stretches the match past its real end and the parse fails silently.
    Braces inside string literals are skipped.
    """
    spans = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(answer):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(answer[start:index + 1])
    return spans


def _as_object(candidate):
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        if json_repair is None:
            return None
        obj = json_repair.repair_json(candidate, return_objects=True)
        if isinstance(obj, list):
            obj = obj[0] if obj else None
    return obj if isinstance(obj, dict) else None


def _parse_answer(answer, task):
    """Parse the answer section into the task's declared fields.

    On failure the raw answer is returned as the prompt so nothing is lost, and
    `parse_ok` is False, which is the only way to tell a fallback from a clean
    parse downstream.
    """
    answer = (answer or "").strip()
    for candidate in reversed(_balanced_spans(answer)):
        obj = _as_object(candidate)
        if obj is None:
            continue
        # Some training runs mis-typed the key as `rewrited_prompt`; accept both.
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        ratio_follow = ""
        if PE_PROFILES[task]["has_ratio_follow"]:
            ratio_follow = str(obj.get("ratio_follow") or "").strip()
        return {
            "positive_prompt": rewritten.strip(),
            "wh_ratio": str(obj.get("wh_ratio") or "").strip(),
            "ratio_follow": ratio_follow,
            "parse_ok": True,
        }
    return {"positive_prompt": answer, "wh_ratio": "", "ratio_follow": "", "parse_ok": False}


def _restore_prefill(text):
    """Put the pre-filled opening back when the model continued from it.

    Depending on how the chat template closes the pre-filled turn, the model
    either repeats `{"rewritten_prompt": "` or carries straight on from it. Only
    the second case needs the opening glued back on, and the guard matters: if
    the model did plan anyway, prepending would swallow the real JSON and turn a
    slow success into a parse failure.
    """
    if '"rewritten_prompt"' in text:
        return text
    return ANSWER_PREFILL + text


def _answer_complete(text, thinking):
    """True once the answer's top-level JSON object has closed.

    Only the part after the thinking block is scanned, because the private plan
    mentions JSON examples of its own and an early `{` in there would look like
    the start of the answer.
    """
    marker = text.rfind("</think>")
    if marker >= 0:
        body = text[marker + len("</think>"):]
    elif thinking:
        # Inside the private plan: the answer has not started yet.
        return False
    else:
        body = text
    depth = 0
    started = False
    in_string = False
    escaped = False
    for char in body:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                started = True
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and started:
                return True
    return False


def _reset_runtime(llm):
    """Return a resident runtime to a clean state before the next generation.

    This llama-cpp-python build keeps the previous turn in the KV cache and its
    multimodal handler only truncates that cache when the new prompt is shorter
    than the history. Feed it an identical prompt and it finds a full match, so
    nothing is re-evaluated and the model keeps writing where it left off -- an
    answer that opens by closing the previous one's JSON. Plain `reset()` is not
    enough either: it only zeroes the counter, and decoding position 0 into a
    cache that still holds a prompt fails outright. Only a memory clear leaves
    the state both layers agree on.
    """
    manager = getattr(llm, "_hybrid_cache_mgr", None)
    if manager is not None:
        try:
            manager.clear()
        except Exception:
            pass
    context = getattr(llm, "_ctx", None)
    clear = getattr(context, "memory_clear", None) if context is not None else None
    if callable(clear):
        try:
            clear(True)
        except TypeError:
            clear()
    reset = getattr(llm, "reset", None)
    if callable(reset):
        reset()
    try:
        llm.n_tokens = 0
    except Exception:
        pass


def _stream_plan(llm, messages, budget, **parameters):
    """Stream the private plan and cut it off once it has run past `budget`.

    The plan is where the minute goes: on a five-image request it measured 7038
    tokens against a 350-token answer. There is no way to ask for less plan --
    the model keeps deliberating -- so the only lever is to stop reading and
    hand the plan it has already written back to it as context (see
    `ANSWER_PREFILL` and the second pass in `_enhance_one`).

    Returns (text, truncated). `truncated` means the model was still planning
    when the budget ran out, so the caller owes it a second pass; if False the
    stream ended on its own and `text` is a finished answer.
    """
    started = time.perf_counter()
    pieces = []
    tokens = 0
    truncated = False
    planning = True
    print(f"[Qwen Image 2.1 PE] 开始内部规划（上限 {budget} tokens）...")
    _reset_runtime(llm)
    stream = llm.create_chat_completion(messages=messages, stream=True, **parameters)
    for chunk in stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if not delta:
            continue
        pieces.append(delta)
        tokens += 1
        text = "".join(pieces)
        if planning:
            if "</think>" in text:
                # It finished planning inside the budget; let it write normally.
                planning = False
            elif tokens >= budget:
                truncated = True
                break
        elif _answer_complete(text, False):
            break
    elapsed = time.perf_counter() - started
    if truncated:
        print(
            f"[Qwen Image 2.1 PE] 规划到 {tokens} tokens 收住（{elapsed:.1f}s），"
            "现在把这段规划交还给模型，让它直接写答案。"
        )
    else:
        print(f"[Qwen Image 2.1 PE] 模型在预算内自己完成了规划（{tokens} tokens，{elapsed:.1f}s）。")
    return "".join(pieces), truncated


def _stream_answer(llm, messages, stage, thinking, **parameters):
    """Stream a completion, stopping the moment the JSON answer is closed.

    The official budgets are huge (16256 / 24000 new tokens) because these are
    general-purpose tools. A prompt rewrite is a few hundred tokens of JSON, so
    anything written after the closing brace is pure waiting -- and on a bad
    input the ceiling alone is several minutes.
    """
    started = time.perf_counter()
    pieces = []
    tokens = 0
    first = None
    print(f"[Qwen Image 2.1 PE] 开始{stage}...")
    _reset_runtime(llm)
    stream = llm.create_chat_completion(messages=messages, stream=True, **parameters)
    for chunk in stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if not delta:
            continue
        if first is None:
            first = time.perf_counter() - started
        pieces.append(delta)
        tokens += 1
        if _answer_complete("".join(pieces), thinking):
            break
    text = "".join(pieces)
    elapsed = time.perf_counter() - started
    decode = elapsed - (first or 0)
    rate = tokens / decode if decode > 0 else 0.0
    print(
        f"[Qwen Image 2.1 PE] {stage}完成：预填充 {first or elapsed:.1f}s，"
        f"生成 {tokens} tokens / {decode:.1f}s（{rate:.0f} tok/s），合计 {elapsed:.1f}s"
    )
    if not _answer_complete(text, thinking):
        print(
            "[Qwen Image 2.1 PE] 提示：到达上限时 JSON 还没闭合，本次回答可能被截断。"
            "可以调大 Max New Tokens 或上下文长度。"
        )
    return text


def _adapt_sampling(llm, sampling):
    """Fit the official sampling names to whatever this llama-cpp-python build takes.

    Builds differ, and the differences are silent if you do not look: the bundled
    one exposes llama.cpp's own `present_penalty` instead of `presence_penalty`,
    and has no `chat_template_kwargs` because local thinking is controlled by the
    chat handler (`enable_thinking` at load time), not by a request field.
    """
    try:
        accepted = set(inspect.signature(llm.create_chat_completion).parameters)
    except (TypeError, ValueError):
        return dict(sampling)
    if any(param.kind == param.VAR_KEYWORD for param in inspect.signature(llm.create_chat_completion).parameters.values()):
        return dict(sampling)
    adapted = {}
    for name, value in sampling.items():
        if name in accepted:
            adapted[name] = value
        elif name == "presence_penalty" and "present_penalty" in accepted:
            adapted["present_penalty"] = value
    return adapted


def _build_llama_template(system_prompt, image_count, thinking=False):
    """Chat template that carries the official system prompt.

    ComfyUI renders this with `str.format`, so literal braces in the system
    prompt (the official ones contain JSON examples) must be escaped before the
    single `{}` placeholder for the user text.

    The assistant turn opens the thinking block itself. The official models were
    trained with `<think>` pre-filled by the chat template, so leaving it out
    would quietly change the contract even though the text still looks fine.

    With thinking off the template emits an empty block instead, mirroring what
    llama.cpp's own Qwen3.5 handler does. Measured on a 4080 that is 7 s vs 19 s
    for t2i and 8.5 s vs 54 s for edit, with the JSON contract intact -- the
    answer is just written without the private plan. That is why "off" is the
    default and the loader exposes the switch.
    """
    safe = system_prompt.replace("{", "{{").replace("}", "}}")
    vision = _VISION_BLOCK * max(0, int(image_count))
    return (
        "<|im_start|>system\n"
        f"{safe}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{vision}{{}}<|im_end|>\n"
        "<|im_start|>assistant\n"
        + ("<think>\n" if thinking else "<think>\n\n</think>\n\n" + ANSWER_PREFILL_TEMPLATE)
    )


def _run_native_clip(clip, prompt, images, profile, system_prompt, sampling, thinking=False):
    """Generate through ComfyUI's own text encoder, with the official chat template."""
    import torch

    image_tensor = None
    image_count = 0
    if images:
        try:
            image_tensor = torch.cat([image.reshape(-1, *image.shape[-3:]) for image in images], dim=0)
        except Exception as exc:  # mismatched sizes cannot share one batch
            raise ValueError(
                "edit 任务的输入图片尺寸必须一致（它们会被当作同一个批次送入模型）。"
            ) from exc
        image_count = int(image_tensor.shape[0])

    template = _build_llama_template(system_prompt, image_count, thinking)
    tokens = clip.tokenize(
        prompt,
        image=image_tensor,
        llama_template=template,
        thinking=True,
    )
    generated = clip.generate(
        tokens,
        do_sample=True,
        max_length=sampling["max_length"],
        temperature=sampling["temperature"],
        top_k=sampling["top_k"],
        top_p=sampling["top_p"],
        min_p=sampling["min_p"],
        repetition_penalty=1.0,
        presence_penalty=sampling["presence_penalty"],
        seed=sampling["seed"],
        # "auto": uses the checkpoint's MTP head when it has one, plain sampling
        # when it does not, which is what the official runners effectively do.
        mtp=True,
    )
    decoded = clip.decode(generated)
    # The template ends inside the answer, so what comes back is the text after
    # the opening we pre-filled.
    return decoded if thinking else _restore_prefill(decoded)


def _ratio_to_pair(text):
    """Parse the model's own ratio format ("16:9") into (width, height)."""
    match = re.match(r"\s*(\d+)\s*[:：xX×]\s*(\d+)\s*$", text or "")
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return (width, height) if width > 0 and height > 0 else None


def _canvas_from_pair(pair, megapixels, multiple=8):
    """Pixel size at the given ratio. Same math as the Resolution Selector."""
    w_ratio, h_ratio = pair
    scale = math.sqrt(float(megapixels) * 1024 * 1024 / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return max(multiple, int(width)), max(multiple, int(height))


def _canvas_from_image(image, multiple=8):
    """Framing of one source image, used by edit runs (ratio_follow names it)."""
    height, width = int(image.shape[-3]), int(image.shape[-2])
    return (
        max(multiple, round(width / multiple) * multiple),
        max(multiple, round(height / multiple) * multiple),
    )


def _image_index(text):
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) - 1 if match else -1


def _resolve_canvas(parsed, images, megapixels, multiple=8, forced_pair=None):
    """Pixel size for the render, so WH Ratio is usable without hand-copying.

    t2i: the model picked the ratio, the megapixel budget is yours.
    edit: `ratio_follow` names the source image whose framing the output keeps,
    so the answer is that image's own size -- rescaling it would defeat the point.
    """
    if forced_pair:
        return _canvas_from_pair(forced_pair, megapixels, multiple)
    pair = _ratio_to_pair(parsed.get("wh_ratio") or "")
    if parsed.get("parse_ok"):
        follow = (parsed.get("ratio_follow") or "").strip()
        if follow:
            index = _image_index(follow)
            if 0 <= index < len(images):
                return _canvas_from_image(images[index], multiple)
        if pair:
            return _canvas_from_pair(pair, megapixels, multiple)
    return _canvas_from_pair(pair or (1, 1), megapixels, multiple)


def _cache_dir():
    import folder_paths

    path = os.path.join(folder_paths.get_user_directory(), "qwen_image21_pe_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_key(task, prompt, images, system_prompt, sampling):
    """Identity of one expansion: same inputs must give the same answer.

    A prompt expansion is a pure function of its inputs, and re-running the same
    request while iterating on the image side is the common case, so the second
    run should not cost another two minutes.
    """
    digest = hashlib.sha256()
    digest.update(task.encode("utf-8"))
    digest.update(prompt.encode("utf-8"))
    digest.update(system_prompt.encode("utf-8"))
    digest.update(json.dumps(sampling, sort_keys=True, default=str).encode("utf-8"))
    for image in images:
        if hasattr(image, "detach"):
            array = image.detach().cpu().numpy()
        else:
            array = np.asarray(image)
        array = np.clip(array * 255.0, 0, 255).astype("uint8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()[:32]


def _cache_read(key):
    path = os.path.join(_cache_dir(), key + ".json")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "positive_prompt" not in data:
        return None
    return data


def _cache_write(key, task, prompt, result):
    record = {
        "task": task,
        "request": prompt[:200],
        "positive_prompt": result[0],
        "wh_ratio": result[1],
        "ratio_follow": result[2],
        "parse_ok": result[3],
    }
    path = os.path.join(_cache_dir(), key + ".json")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[Qwen Image 2.1 PE] 缓存写入失败（不影响本次结果）：{exc}")


def _pe_bundle(source, **values):
    """One shape for every loader, so the enhancer never asks where it came from."""
    bundle = {
        "source": source,
        "t2i_model": "",
        "i2i_model": "",
        "vision_model": "",
        "gpu_layers": -1,
        "context_length": DEFAULT_CONTEXT_LENGTH,
        "system_prompt_file": "",
        "thinking": False,
        "unload": True,
        # -1 keeps the official behaviour (plan until the model stops on its
        # own). The GGUF loader exposes this; the native path has no streaming,
        # so it never reads it.
        "plan_tokens": -1,
    }
    bundle.update(values)
    return bundle


def _report_bundle(bundle):
    """Print what the loader actually resolved, once per run.

    A workflow stores widget values by position, so a node whose controls were
    re-ordered can quietly hand one control another's value (the "24576" that
    ended up in System Prompt File came from exactly that). Printing the resolved
    settings turns this into a line you can read instead of a puzzle.
    """
    window = (
        f" | 上下文={bundle['context_length']}" if bundle["source"] == SOURCE_GGUF else ""
    )
    plan = bundle["plan_tokens"]
    plan_text = ""
    if bundle["thinking"] and bundle["source"] == SOURCE_GGUF:
        plan_text = " | 规划=不限" if plan < 0 else f" | 规划={plan} tokens"
    print(
        f"[Qwen Image 2.1 PE] 加载节点：{bundle['source']}{window}"
        f" | 系统提示词={'自定义文件' if bundle['system_prompt_file'] else '官方内置'}"
        f" | 思考={'开' if bundle['thinking'] else '关'}"
        f"{plan_text}"
        f" | 用完{'卸载' if bundle['unload'] else '保留'}"
    )


def _task_model(bundle, task):
    """The file this run should use: the two tasks have their own checkpoint.

    Both loaders offer a T2I and an I2I picker, so the task decides which one is
    loaded -- you never pick a model and a task separately and hope they match.
    """
    return bundle.get("t2i_model" if task == "t2i" else "i2i_model", "")


class QwenImage21PELoaderSafetensors:
    """Official PE safetensors: the reference weights.

    Two pickers, because the two tasks have their own checkpoint. The node loads
    whichever one the task needs and releases it afterwards; measured, that ends
    up identical to feeding a CLIPLoader in, only without keeping 8.8 GB parked in
    VRAM between runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        encoders = _text_encoder_choices()
        return {
            "required": {
                "T2I Encoder": (
                    encoders,
                    {
                        "default": _default_encoder(encoders, "pe_t2i"),
                        "tooltip": "文生图任务用的 PE 编码器（text_encoders 里选）。",
                    },
                ),
                "I2I Encoder": (
                    encoders,
                    {
                        "default": _default_encoder(encoders, "pe_i2i"),
                        "tooltip": "带图改写任务用的 PE 编码器（text_encoders 里选）。",
                    },
                ),
                "System Prompt File": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "留空则使用随权重配套的官方系统提示词。只有换权重时才需要指定。",
                    },
                ),
                "Thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Think",
                        "label_off": "Direct",
                        "tooltip": "要不要让模型先内部推理再写答案。"
                                   "关（Direct）：直接写答案，比 GGUF 快，实测带图改写约 29 秒，"
                                   "但没有分析画面和权衡需求的机会，出来的提示词会明显变浅。"
                                   "开（Think）：质量好得多，但原生推理拿不到流式输出、无法中途截断，"
                                   "所以只能是完整思考——实测一次带图改写要十分钟以上。"
                                   "要质量又要速度，请换上面的 GGUF 加载节点，那边有 Plan Tokens 可以限制规划长度。",
                    },
                ),
                "Unload Model After Generation": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (PE_MODEL_TYPE,)
    RETURN_NAMES = ("PE Model",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Prompt"
    DESCRIPTION = "Load the official Qwen-Image-2.1 PE encoders (text_encoders) for the enhancer node."

    def build(self, **inputs):
        bundle = _pe_bundle(
            SOURCE_AUTO,
            t2i_model=inputs.get("T2I Encoder", ""),
            i2i_model=inputs.get("I2I Encoder", ""),
            system_prompt_file=inputs.get("System Prompt File", ""),
            thinking=bool(inputs.get("Thinking", False)),
            unload=bool(inputs.get("Unload Model After Generation", True)),
        )
        _report_bundle(bundle)
        return (bundle,)


class QwenImage21PELoaderGGUF:
    """PE weights as GGUF, run by llama.cpp.

    Measured about five times faster than ComfyUI's int8 path on a 16 GB card, with
    the same answer contract. Needs the matching mmproj because the loader is
    built on the multimodal handler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _pe_language_models() or ["No language models found"]
        vision_models = _vision_models() or ["No vision models found"]
        return {
            "required": {
                "T2I GGUF": (
                    models,
                    {
                        "default": _default_encoder(models, "pe-t2i"),
                        "tooltip": "文生图任务用的 PE-T2I GGUF（列表只列 PE 检查点）。",
                    },
                ),
                "I2I GGUF": (
                    models,
                    {
                        "default": _default_encoder(models, "pe-i2i"),
                        "tooltip": "带图改写任务用的 PE-I2I GGUF。两个都选好，任务由主节点自动判断。",
                    },
                ),
                "Vision Model": (
                    vision_models,
                    {
                        "tooltip": "配套 mmproj 视觉模型。PE-I2I 的官方 mmproj 或任意 "
                                   "Qwen3.5-9B 的 mmproj 都可以。",
                    },
                ),
                "GPU Offload Layers": ("INT", {"default": -1, "min": -1, "max": 256, "step": 1}),
                "Context Length": (
                    list(CONTEXT_LENGTH_OPTIONS),
                    {
                        "default": "16384",
                        "tooltip": "模型上下文窗口。默认 16384：关掉思考后一次回复只有几百 token，"
                                   "这个窗口省下约 3GB 显存，也不会拖慢速度。"
                                   "只有打开 Think、或者提示词里塞了很长的参考素材时才需要调大。",
                    },
                ),
                "System Prompt File": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "留空则使用随权重配套的官方系统提示词。只有换权重时才需要指定。",
                    },
                ),
                "Thinking": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Think",
                        "label_off": "Direct",
                        "tooltip": "要不要让模型先内部推理再写答案。"
                                   "关（Direct）最快，实测 t2i 约 7 秒、带图改写约 8 秒，"
                                   "但它没有分析画面和权衡需求的机会，复杂请求出来的提示词会明显变浅。"
                                   "开（Think）质量好得多，规划长度由下面的 Plan Tokens 控制。",
                    },
                ),
                "Unload Model After Generation": ("BOOLEAN", {"default": True}),
                "Plan Tokens": (
                    "INT",
                    {
                        "default": 800,
                        "min": -1,
                        "max": 16384,
                        "step": 100,
                        "tooltip": "思考（规划）的 token 上限，只在 Thinking 打开时生效。"
                                   "推理是这里最贵的一段：实测一次五图改写写了 7038 个 token、89 秒，"
                                   "而答案只有 350 个 token。"
                                   "写到上限就收住，把已经写好的规划交还给模型直接写答案，"
                                   "所以越小越快：800 约 20 秒、400 约 15 秒。"
                                   "-1 = 不限，等于官方完整思考（约 90 秒，细节最全）。"
                                   "规划越短，越容易漏掉请求里的隐含要求。",
                    },
                ),
            },
        }

    RETURN_TYPES = (PE_MODEL_TYPE,)
    RETURN_NAMES = ("PE Model",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Prompt"
    DESCRIPTION = "Load a Qwen-Image-2.1 PE checkpoint as GGUF (llama.cpp) for the enhancer node."

    def build(self, **inputs):
        bundle = _pe_bundle(
            SOURCE_GGUF,
            t2i_model=inputs.get("T2I GGUF", ""),
            i2i_model=inputs.get("I2I GGUF", ""),
            vision_model=inputs.get("Vision Model", ""),
            gpu_layers=int(inputs.get("GPU Offload Layers", -1)),
            context_length=int(
                inputs.get("Context Length", DEFAULT_CONTEXT_LENGTH) or DEFAULT_CONTEXT_LENGTH
            ),
                system_prompt_file=inputs.get("System Prompt File", ""),
                thinking=bool(inputs.get("Thinking", True)),
                unload=bool(inputs.get("Unload Model After Generation", True)),
                plan_tokens=int(inputs.get("Plan Tokens", -1)),
            )
        _report_bundle(bundle)
        return (bundle,)


class QwenImage21PESettings:
    """Sampling knobs, split out so the enhancer shows only what changes per run.

    Leave it unconnected and every run uses the official per-task settings, which
    is what you want unless you are deliberately experimenting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Sampling Preset": (
                    [SAMPLING_PRESET_DEFAULT, "Custom"],
                    {
                        "default": SAMPLING_PRESET_DEFAULT,
                        "tooltip": "官方默认使用各任务的出厂参数（t2i 的 presence_penalty=1.5，edit 为 0）。"
                                   "选 Custom 才使用下面那些数值。",
                    },
                ),
                "Temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Top P": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Top K": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "Presence Penalty": ("FLOAT", {"default": 1.5, "min": -2.0, "max": 2.0, "step": 0.05}),
                "Max New Tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32768,
                        "step": 256,
                        "tooltip": "0 = 使用官方默认（t2i 16256 / edit 24000），"
                                   "并受上下文长度限制自动收敛。",
                    },
                ),
            },
        }

    RETURN_TYPES = (PE_SETTINGS_TYPE,)
    RETURN_NAMES = ("PE Settings",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Prompt"
    DESCRIPTION = "Optional sampling overrides for the Qwen-Image-2.1 prompt enhancer."

    def build(self, **inputs):
        return (
            {
                "preset": inputs.get("Sampling Preset", SAMPLING_PRESET_DEFAULT),
                "temperature": float(inputs.get("Temperature", 1.0)),
                "top_p": float(inputs.get("Top P", 0.95)),
                "top_k": int(inputs.get("Top K", 20)),
                "presence_penalty": float(inputs.get("Presence Penalty", 1.5)),
                "max_new_tokens": int(inputs.get("Max New Tokens", 0) or 0),
            },
        )


class QwenImage21PromptEnhancer:
    """Expand a short request into a Qwen-Image-2.1 prompt with the official PE models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "任何语言的简短需求，例如：一只在雨中弹吉他的柯基",
                    },
                ),
                "Task": (
                    ALL_TASK_LABELS,
                    {
                        "default": TASK_AUTO,
                        "tooltip": "Auto：连了图片就走 edit，没连图片就走 t2i，节点自己按这个选权重。"
                                   "t2i：把简短描述扩写成成片画面的长提示词（只要文字）。"
                                   "edit：把改图指令加上参考图写成精确指令。"
                                   "两个任务用的是不同权重和不同系统提示词，不能互换。",
                    },
                ),
                "Seed": ("INT", {"default": 42, "min": -1, "max": 0xFFFFFFFF, "step": 1}),
                "Use Cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "相同的请求 + 相同的种子会直接复用上次结果，跳过这次生成。"
                                   "注意 ComfyUI 自己也会缓存节点结果：输入没变时整张图都直接复用，"
                                   "这时要把 Seed 设成 -1 才会重算（-1 会绕过 ComfyUI 的缓存）。"
                                   "缓存目录：ComfyUI/user/qwen_image21_pe_cache。",
                    },
                ),
                "Target Megapixels": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Width/Height 两路输出按这个像素总量换算。"
                                   "t2i 用模型选的画幅比例；edit 直接沿用参考图的尺寸，不受这里影响。",
                    },
                ),
                "Aspect Ratio": (
                    ASPECT_OPTIONS,
                    {
                        "default": ASPECT_AUTO,
                        "tooltip": "默认让模型自己定画幅（官方行为）。想固定就选一个："
                                   "节点会把这个画幅同时写进请求交给模型，并让 Width/Height 按它计算，"
                                   "这样提示词描述的构图和实际画布是一致的。",
                    },
                ),
                "Prompt Count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "输出几条提示词（列表形式，下游会按条数各跑一次）。"
                                   "官方契约要求一次回答只给一条，所以每条都是一次完整生成，"
                                   "耗时基本线性叠加：关掉思考时 t2i 每条约 7 秒、edit 约 8.5 秒"
                                   "（打开 Think 则是 19 秒 / 52 秒）。"
                                   "同批各条用 Seed、Seed+1、Seed+2…，模型全程只载入一次。",
                    },
                ),
            },
            "optional": {
                "pe_model": (
                    PE_MODEL_TYPE,
                    {
                        "tooltip": "连接 Qwen Image 2.1 PE Loader：用哪个权重、上下文、"
                                   "卸载策略和 clip 都由那个节点决定。",
                    },
                ),
                "pe_settings": (
                    PE_SETTINGS_TYPE,
                    {
                        "tooltip": "可选。连接 Qwen Image 2.1 PE Settings 才使用里面的采样参数；"
                                   "不连则用官方出厂设置。",
                    },
                ),
                **{f"Image {index}": ("IMAGE",) for index in range(1, MAX_INPUT_IMAGES + 1)},
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("Positive Prompt", "WH Ratio", "Ratio Follow", "Parse OK", "Width", "Height")
    # Every per-prompt value is a list, so a Prompt Count above 1 feeds the graph
    # N times (ComfyUI runs a downstream node once per element) and the width and
    # height stay matched to their prompt. With the default count of 1 the lists
    # hold a single item, which behaves exactly like a scalar.
    OUTPUT_IS_LIST = (True, True, True, True, True, True)
    FUNCTION = "enhance"
    CATEGORY = "MiniMax H3/Prompt"
    DESCRIPTION = "Official Qwen-Image-2.1 prompt enhancer (PE-T2I / PE-I2I). Connect a PE loader for the weights."

    @classmethod
    def IS_CHANGED(cls, **inputs):
        """Make the -1 seed mode bypass ComfyUI's execution cache.

        Without this, queueing twice with the same widgets reuses the previous
        result without running anything, so -1 (a fresh roll every run) does not
        actually roll again.
        """
        try:
            seed = int(inputs.get("Seed", 42))
        except (TypeError, ValueError):
            seed = -1
        return time.time_ns() if seed < 0 else seed

    def enhance(self, **inputs):
        count = max(1, int(inputs.get("Prompt Count", 1) or 1))
        seed = int(inputs.get("Seed", 42))
        if seed < 0:
            seed = random.SystemRandom().randint(0, 0xFFFFFFFF)
        rows = [
            self._enhance_one(index, (seed + index) & 0xFFFFFFFF, index == count - 1, **inputs)
            for index in range(count)
        ]
        return tuple([row[column] for row in rows] for column in range(6))

    def _enhance_one(self, variant, seed, is_last, **inputs):
        model = inputs.get("pe_model")
        if not model:
            raise ValueError(
                "请连接“Qwen Image 2.1 PE Loader”：本节点只负责扩写，"
                "权重、上下文和 clip 都在那个加载节点上选。"
            )
        settings = inputs.get("pe_settings") or {}
        thinking = bool(model.get("thinking", False))
        prompt = (inputs.get("Prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt 不能为空。")

        images = [
            inputs[f"Image {index}"]
            for index in range(1, MAX_INPUT_IMAGES + 1)
            if inputs.get(f"Image {index}") is not None
        ]
        task = _resolve_task(inputs.get("Task", TASK_AUTO), bool(images))
        profile = PE_PROFILES[task]
        if inputs.get("Task", TASK_AUTO) == TASK_AUTO:
            print(f"[Qwen Image 2.1 PE] 自动判断任务：{'Image Edit (edit)' if task == 'edit' else 'Text to Image (t2i)'}")
        if profile["takes_images"] and not images:
            raise ValueError("edit 任务至少需要一张输入图片，请连接 Image 1。")
        if not profile["takes_images"] and images:
            # The official tooling treats this as an error rather than a warning:
            # dropping the images would look like a successful run of the wrong task.
            raise ValueError("t2i 任务不接受输入图片。要带图改写真，请把 Task 换成 Image Edit。")

        if settings.get("preset", SAMPLING_PRESET_DEFAULT) == SAMPLING_PRESET_DEFAULT:
            temperature = profile["temperature"]
            top_p = profile["top_p"]
            top_k = profile["top_k"]
            presence_penalty = profile["presence_penalty"]
        else:
            temperature = float(settings.get("temperature", profile["temperature"]))
            top_p = float(settings.get("top_p", profile["top_p"]))
            top_k = int(settings.get("top_k", profile["top_k"]))
            presence_penalty = float(settings.get("presence_penalty", profile["presence_penalty"]))

        context_length = int(model.get("context_length") or DEFAULT_CONTEXT_LENGTH)
        # Only an explicit override is worth reporting on. Left at 0 the ceiling
        # is still the official 16256 / 24000, which the context window trims to
        # half -- normal, and the stream stops at the closing brace anyway, so
        # warning about it on every run is just noise.
        override = int(settings.get("max_new_tokens", 0) or 0)
        requested = override or profile["max_new_tokens"]
        system_prompt = _load_system_prompt(task, model.get("system_prompt_file", ""))
        megapixels = float(inputs.get("Target Megapixels", 2.0) or 2.0)
        forced_ratio = inputs.get("Aspect Ratio", ASPECT_AUTO)
        forced_pair = None if forced_ratio == ASPECT_AUTO else _ratio_to_pair(forced_ratio)
        if forced_pair:
            # Bilingual on purpose: the edit task mirrors the request's language,
            # and a single-language marker could flip it.
            model_prompt = (
                f"{prompt}\n\n(输出画幅 / output aspect ratio: {forced_ratio})"
            )
        else:
            model_prompt = prompt
        source = model.get("source", SOURCE_AUTO)
        if source == SOURCE_AUTO:
            # ComfyUI grows the KV cache with the request, so there is no fixed
            # window to trim against here.
            max_tokens = requested
        else:
            max_tokens = _completion_budget(context_length, requested)
            if max_tokens < requested and override:
                print(
                    f"[Qwen Image 2.1 PE] 上下文 {context_length} 放不下你设定的 {requested} 个新 token，"
                    f"本次上限收敛为 {max_tokens}。需要更长输出请调大 Context Length。"
                )

        cache_key = None
        if bool(inputs.get("Use Cache", True)):
            model_hint = _task_model(model, task)
            cache_key = _cache_key(
                task,
                prompt,
                images,
                system_prompt,
                {
                    "source": source,
                    "model": model_hint,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "presence_penalty": presence_penalty,
                    "seed": seed,
                    "aspect": forced_ratio,
                    "thinking": thinking,
                    "plan_tokens": int(model.get("plan_tokens", -1)),
                },
            )
            cached = _cache_read(cache_key)
            if cached is not None:
                print(
                    "[Qwen Image 2.1 PE] 命中缓存（相同请求 + 相同种子），跳过生成。"
                    "需要重新生成请关掉 Use Cache 或清空 user/qwen_image21_pe_cache。"
                )
                width, height = _resolve_canvas(cached, images, megapixels, forced_pair=forced_pair)
                return (
                    cached.get("positive_prompt", ""),
                    forced_ratio if forced_pair else cached.get("wh_ratio", ""),
                    cached.get("ratio_follow", ""),
                    bool(cached.get("parse_ok", False)),
                    width,
                    height,
                )

        content = []
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _tensor_to_data_url(image, max_size=1024)},
                }
            )
        content.append({"type": "text", "text": model_prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        if not thinking:
            # See ANSWER_PREFILL: without this the model plans anyway on complex
            # requests, and the plan is where the minute goes.
            messages.append({"role": "assistant", "content": ANSWER_PREFILL})

        if source == SOURCE_AUTO:
            encoder = _task_model(model, task)
            if not encoder or encoder == "No text encoders found":
                raise ValueError(
                    "没有可用的文本编码器。请把官方 PE 权重放进 models/text_encoders，"
                    "或改用 Qwen Image 2.1 PE Loader (GGUF)。"
                )
            if variant == 0:
                print(
                    f"[Qwen Image 2.1 PE] 载入 {task} 编码器：{encoder}"
                    "（两个 PE 编码器各 8.8GB，本节点全程只驻留其中一个）"
                )
            clip = _load_encoder(encoder)
            try:
                raw = _run_native_clip(
                    clip,
                    model_prompt,
                    images,
                    profile,
                    system_prompt,
                    {
                        "max_length": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "min_p": profile["min_p"],
                        "presence_penalty": presence_penalty,
                        "seed": seed,
                    },
                    thinking,
                )
            finally:
                # A multi-prompt batch keeps the model until the last variant.
                if is_last and bool(model.get("unload", True)):
                    _release_encoder()
            thinking, answer = _split_thinking(raw)
            parsed = _parse_answer(answer, task)
            self._report(parsed)
            return self._store(
                parsed, cache_key, task, prompt, images, megapixels, forced_ratio, forced_pair
            )

        model_name = _task_model(model, task)
        vision_name = model.get("vision_model")
        if not model_name or model_name == "No language models found":
            raise FileNotFoundError(
                f"加载节点里没有为 {task} 任务指定模型，请把 PE-"
                f"{'T2I' if task == 't2i' else 'I2I'} 的 GGUF 选上。"
            )
        if not vision_name or vision_name == "No vision models found":
            raise FileNotFoundError("请在加载节点里选择配套的 mmproj 视觉模型。")
        load_started = time.perf_counter()
        resident = _VisionRuntime.llm is not None
        # Trained with the thinking block, but measured 6x slower with it on for
        # the edit task (54 s vs 8.5 s on a 4080): the plan is where almost all
        # the tokens go. Reusing the resident runtime keeps repeat runs cheap.
        llm = _VisionRuntime.ensure(
            model_name,
            vision_name,
            int(model.get("gpu_layers", -1)),
            context_length,
            thinking,
        )
        print(
            f"[Qwen Image 2.1 PE] {task} 模型就绪：{model_name}"
            f"（{'复用' if resident else '载入'} {time.perf_counter() - load_started:.1f}s）"
        )
        try:
            sampling = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": profile["min_p"],
                "presence_penalty": presence_penalty,
                "seed": seed,
            }
            # The local llama-cpp-python build exposes llama.cpp's own names, so
            # map them before the call.
            sampling = _adapt_sampling(llm, sampling)
            stage = f"Qwen Image 2.1 {task} 提示词扩写"
            plan_tokens = int(model.get("plan_tokens", -1))
            if thinking and plan_tokens >= 0:
                raw, truncated = _stream_plan(llm, messages, plan_tokens, **sampling)
                if truncated and raw.strip():
                    # Hand the half-written plan back as the assistant turn and
                    # turn the generation prompt off, so the model continues from
                    # inside the answer instead of opening a new turn.
                    messages = messages + [
                        {
                            "role": "assistant",
                            "content": raw.rstrip() + "\n</think>\n\n" + ANSWER_PREFILL,
                        }
                    ]
                    sampling["add_generation_prompt"] = False
                    raw = _stream_answer(llm, messages, f"{stage}（写答案）", False, **sampling)
                    raw = _restore_prefill(raw)
            else:
                raw = _stream_answer(llm, messages, stage, thinking, **sampling)
        finally:
            if is_last and bool(model.get("unload", True)):
                _VisionRuntime.close()

        if not thinking:
            raw = _restore_prefill(raw)
        thinking, answer = _split_thinking(raw)
        parsed = _parse_answer(answer, task)
        self._report(parsed)
        return self._store(
            parsed, cache_key, task, prompt, images, megapixels, forced_ratio, forced_pair
        )

    @staticmethod
    def _store(parsed, cache_key, task, prompt, images, megapixels, forced_ratio, forced_pair):
        width, height = _resolve_canvas(parsed, images, megapixels, forced_pair=forced_pair)
        if forced_pair and parsed.get("wh_ratio") and forced_ratio != parsed["wh_ratio"]:
            print(
                f"[Qwen Image 2.1 PE] 画幅按你的设定用 {forced_ratio}"
                f"（模型自己写的是 {parsed['wh_ratio']}）。"
            )
        result = (
            parsed["positive_prompt"],
            forced_ratio if forced_pair else parsed["wh_ratio"],
            parsed["ratio_follow"],
            parsed["parse_ok"],
            width,
            height,
        )
        if cache_key:
            # The canvas is derived from the answer, not generated, so only the
            # model's own four fields are worth storing.
            _cache_write(cache_key, task, prompt, result[:4])
        return result

    @staticmethod
    def _report(parsed):
        if not parsed["parse_ok"]:
            print(
                "[Qwen Image 2.1 PE] 警告：回答不是预期的 JSON，Positive Prompt 已回退为原始回答文本。"
                "常见的三个原因：选错了权重（要用 PE-T2I / PE-I2I，而不是基座模型）、"
                "系统提示词与权重不匹配、或输出被上下文长度截断。"
            )
        else:
            print(
                f"[Qwen Image 2.1 PE] 解析成功，画布："
                f"{parsed['wh_ratio'] or parsed['ratio_follow'] or '未指定'}"
            )
