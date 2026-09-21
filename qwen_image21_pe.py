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

import numpy as np

from .nodes import (
    CONTEXT_LENGTH_OPTIONS,
    DEFAULT_CONTEXT_LENGTH,
    _OnlineRuntime,
    _VisionRuntime,
    _completion_budget,
    _input_value,
    _language_models,
    _stream_completion,
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

MAX_INPUT_IMAGES = 4

# Which runtime turns the request into text. The local safetensors path uses
# ComfyUI's own text-encoder inference (that is what the Comfy-Org repacks of the
# PE models are for), so it gets the same int8/kv-cache handling as everything
# else in the graph.
SOURCE_GGUF = "Local GGUF"
SOURCE_CLIP = "Local safetensors (CLIP)"
SOURCE_AUTO = "Local safetensors (auto)"
SOURCE_ONLINE = "Online LLM"
MODEL_SOURCES = [SOURCE_GGUF, SOURCE_AUTO, SOURCE_CLIP, SOURCE_ONLINE]

TASK_AUTO = "Auto (by images)"

_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

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
    """
    path = (override_path or "").strip()
    if not path:
        path = os.path.join(PROMPT_DIR, PE_PROFILES[task]["prompt_file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"系统提示词文件不存在：{path}")
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


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


def _build_llama_template(system_prompt, image_count):
    """Chat template that carries the official system prompt.

    ComfyUI renders this with `str.format`, so literal braces in the system
    prompt (the official ones contain JSON examples) must be escaped before the
    single `{}` placeholder for the user text.

    The assistant turn opens the thinking block itself. The official models were
    trained with `<think>` pre-filled by the chat template, so leaving it out
    would quietly change the contract even though the text still looks fine.
    """
    safe = system_prompt.replace("{", "{{").replace("}", "}}")
    vision = _VISION_BLOCK * max(0, int(image_count))
    return (
        "<|im_start|>system\n"
        f"{safe}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{vision}{{}}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )


def _run_native_clip(clip, prompt, images, profile, system_prompt, sampling):
    """Generate through ComfyUI's own text encoder (CLIPLoader -> this node)."""
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

    template = _build_llama_template(system_prompt, image_count)
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
    return clip.decode(generated)


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


def _resolve_canvas(parsed, images, megapixels, multiple=8):
    """Pixel size for the render, so WH Ratio is usable without hand-copying.

    t2i: the model picked the ratio, the megapixel budget is yours.
    edit: `ratio_follow` names the source image whose framing the output keeps,
    so the answer is that image's own size -- rescaling it would defeat the point.
    """
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


class QwenImage21PromptEnhancer:
    """Expand a short request into a Qwen-Image-2.1 prompt with the official PE models."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _language_models() or ["No language models found"]
        vision_models = _vision_models() or ["No vision models found"]
        encoders = _text_encoder_choices()
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
                "Sampling Preset": (
                    ["Official defaults", "Custom"],
                    {
                        "default": "Official defaults",
                        "tooltip": "官方默认使用各任务的出厂采样参数（t2i 的 presence_penalty=1.5，"
                                   "edit 为 0）。选 Custom 才会使用下面那些数值。",
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
                "Seed": ("INT", {"default": 42, "min": -1, "max": 0xFFFFFFFF, "step": 1}),
                "Model Source": (
                    MODEL_SOURCES,
                    {
                        "default": SOURCE_GGUF,
                        "tooltip": "Local GGUF：本地 GGUF 权重（llama.cpp）。"
                                   "Local safetensors (CLIP)：用 CLIPLoader 加载官方 PE safetensors 后接到本节点的 clip 输入，"
                                   "走 ComfyUI 原生推理（int8_convrot 可用）。"
                                   "Online LLM：OpenAI 兼容接口（vLLM 起的 PE 服务等）。",
                    },
                ),
                "Language Model": (models,),
                "Vision Model": (vision_models,),
                "GPU Offload Layers": ("INT", {"default": -1, "min": -1, "max": 256, "step": 1}),
                "Context Length": (
                    list(CONTEXT_LENGTH_OPTIONS),
                    {
                        "default": "32768",
                        "tooltip": "模型上下文窗口。PE 模型带思考块，输出很长，"
                                   "官方 t2i 允许 16256 个新 token，所以不要设得太小。",
                    },
                ),
                "Online Request URL": ("STRING", {"default": "https://api.openai.com/v1", "multiline": False}),
                "Online API Key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "Online Model": ("STRING", {"default": "", "multiline": False}),
                "System Prompt File": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "留空则使用节点自带、随权重配套的官方系统提示词。"
                                   "只有换权重时才需要指定。",
                    },
                ),
                "Unload Model After Generation": ("BOOLEAN", {"default": True}),
                "Use Cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "相同的请求 + 相同的种子会直接复用上次结果，跳过这次生成。"
                                   "一次扩写要一两分钟，反复调图时这个开关能省掉绝大部分等待。"
                                   "缓存目录：ComfyUI/user/qwen_image21_pe_cache。",
                    },
                ),
                "T2I Encoder": (
                    encoders,
                    {
                        "default": _default_encoder(encoders, "pe_t2i"),
                        "tooltip": "模型来源选 Local safetensors (auto) 时，文生图任务用的 PE 编码器。",
                    },
                ),
                "I2I Encoder": (
                    encoders,
                    {
                        "default": _default_encoder(encoders, "pe_i2i"),
                        "tooltip": "模型来源选 Local safetensors (auto) 时，改图任务用的 PE 编码器。",
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
            },
            "optional": {
                **{f"Image {index}": ("IMAGE",) for index in range(1, MAX_INPUT_IMAGES + 1)},
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "模型来源选 Local safetensors (CLIP) 时，把 CLIPLoader 加载的 "
                                   "Qwen-Image-2.1 PE 编码器接到这里。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("Positive Prompt", "WH Ratio", "Ratio Follow", "Parse OK", "Width", "Height")
    FUNCTION = "enhance"
    CATEGORY = "MiniMax H3/Prompt"
    DESCRIPTION = "Official Qwen-Image-2.1 prompt enhancer (PE-T2I / PE-I2I) for local GGUF or online LLMs."

    def enhance(self, **inputs):
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

        seed = int(inputs.get("Seed", 42))
        if seed < 0:
            seed = random.SystemRandom().randint(0, 0xFFFFFFFF)

        if inputs.get("Sampling Preset", "Official defaults") == "Official defaults":
            temperature = profile["temperature"]
            top_p = profile["top_p"]
            top_k = profile["top_k"]
            presence_penalty = profile["presence_penalty"]
        else:
            temperature = float(inputs.get("Temperature", profile["temperature"]))
            top_p = float(inputs.get("Top P", profile["top_p"]))
            top_k = int(inputs.get("Top K", profile["top_k"]))
            presence_penalty = float(inputs.get("Presence Penalty", profile["presence_penalty"]))

        context_length = int(inputs.get("Context Length", DEFAULT_CONTEXT_LENGTH) or DEFAULT_CONTEXT_LENGTH)
        requested = int(inputs.get("Max New Tokens", 0) or 0) or profile["max_new_tokens"]
        system_prompt = _load_system_prompt(task, inputs.get("System Prompt File", ""))
        megapixels = float(inputs.get("Target Megapixels", 2.0) or 2.0)
        source = inputs.get("Model Source", SOURCE_GGUF)
        if source in (SOURCE_CLIP, SOURCE_AUTO):
            # ComfyUI grows the KV cache with the request, so there is no fixed
            # window to trim against here.
            max_tokens = requested
        else:
            max_tokens = _completion_budget(context_length, requested)
            if max_tokens < requested:
                print(
                    f"[Qwen Image 2.1 PE] 上下文 {context_length} 放不下官方 {requested} 个新 token，"
                    f"本次上限收敛为 {max_tokens}。需要更长输出请调大 Context Length。"
                )

        cache_key = None
        if bool(inputs.get("Use Cache", True)):
            model_hint = {
                SOURCE_GGUF: inputs.get("Language Model", ""),
                SOURCE_ONLINE: inputs.get("Online Model", ""),
            }.get(source, "clip")
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
                },
            )
            cached = _cache_read(cache_key)
            if cached is not None:
                print(
                    "[Qwen Image 2.1 PE] 命中缓存（相同请求 + 相同种子），跳过生成。"
                    "需要重新生成请关掉 Use Cache 或清空 user/qwen_image21_pe_cache。"
                )
                width, height = _resolve_canvas(cached, images, megapixels)
                return (
                    cached.get("positive_prompt", ""),
                    cached.get("wh_ratio", ""),
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
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        if source in (SOURCE_CLIP, SOURCE_AUTO):
            if source == SOURCE_AUTO:
                encoder = inputs.get("T2I Encoder" if task == "t2i" else "I2I Encoder", "")
                if not encoder or encoder == "No text encoders found":
                    raise ValueError(
                        "没有可用的文本编码器。请把官方 PE 权重放进 models/text_encoders，"
                        "或改用 Local GGUF / Online LLM。"
                    )
                print(
                    f"[Qwen Image 2.1 PE] 载入 {task} 编码器：{encoder}"
                    "（两个 PE 编码器各 8.8GB，本节点全程只驻留其中一个）"
                )
                clip = _load_encoder(encoder)
            else:
                clip = inputs.get("clip")
                if clip is None:
                    raise ValueError(
                        "模型来源选了 Local safetensors (CLIP)，但没有连接 clip。"
                        "请用 CLIPLoader 加载 PE 编码器后接到本节点的 clip 输入，"
                        "或把模型来源改成 Local safetensors (auto) 让节点自己载入。"
                    )
            try:
                raw = _run_native_clip(
                    clip,
                    prompt,
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
                )
            finally:
                if source == SOURCE_AUTO and bool(inputs.get("Unload Model After Generation", True)):
                    _release_encoder()
            thinking, answer = _split_thinking(raw)
            parsed = _parse_answer(answer, task)
            self._report(parsed)
            return self._store(parsed, cache_key, task, prompt, images, megapixels)

        online = source == SOURCE_ONLINE
        if online:
            key = (inputs.get("Online API Key") or "").strip()
            if not key:
                raise ValueError("在线模式必须填写 API Key。")
            llm = _OnlineRuntime(
                inputs.get("Online Request URL", ""),
                key,
                inputs.get("Online Model", ""),
            )
        else:
            model_name = inputs.get("Language Model")
            vision_name = inputs.get("Vision Model")
            if model_name in {"No language models found", None} or vision_name in {"No vision models found", None}:
                raise FileNotFoundError("请先选择语言模型与配套的 mmproj 视觉模型。")
            llm = _VisionRuntime.load(
                model_name,
                vision_name,
                inputs.get("GPU Offload Layers", -1),
                context_length,
                # Both PE models were trained with the thinking block and degrade
                # without it, so it is always on here.
                True,
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
            if online:
                # vLLM / SGLang extensions: the served model keeps its thinking
                # block on, exactly like the official client does.
                sampling["chat_template_kwargs"] = {"enable_thinking": True}
            else:
                sampling = _adapt_sampling(llm, sampling)
            raw = _stream_completion(
                llm,
                messages,
                f"Qwen Image 2.1 {task} 提示词扩写",
                **sampling,
            )
        finally:
            if not online and bool(inputs.get("Unload Model After Generation", True)):
                _VisionRuntime.close()

        thinking, answer = _split_thinking(raw)
        parsed = _parse_answer(answer, task)
        self._report(parsed)
        return self._store(parsed, cache_key, task, prompt, images, megapixels)

    @staticmethod
    def _store(parsed, cache_key, task, prompt, images, megapixels):
        width, height = _resolve_canvas(parsed, images, megapixels)
        result = (
            parsed["positive_prompt"],
            parsed["wh_ratio"],
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
