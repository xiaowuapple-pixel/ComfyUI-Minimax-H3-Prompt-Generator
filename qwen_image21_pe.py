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
import os
import random

from .nodes import (
    CONTEXT_LENGTH_OPTIONS,
    DEFAULT_CONTEXT_LENGTH,
    _OnlineRuntime,
    _VisionRuntime,
    _completion_budget,
    _input_value,
    _is_online_source,
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


def _task_key(label):
    return _LABEL_TO_TASK.get(label, "t2i")


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


class QwenImage21PromptEnhancer:
    """Expand a short request into a Qwen-Image-2.1 prompt with the official PE models."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _language_models() or ["No language models found"]
        vision_models = _vision_models() or ["No vision models found"]
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
                    TASK_LABELS,
                    {
                        "default": TASK_LABELS[0],
                        "tooltip": "t2i：把简短描述扩写成成片画面的长提示词（只要文字）。"
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
                "Model Source": ("BOOLEAN", {"default": False, "label_on": "Online LLM", "label_off": "Local Model"}),
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
            },
            "optional": {f"Image {index}": ("IMAGE",) for index in range(1, MAX_INPUT_IMAGES + 1)},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("Positive Prompt", "WH Ratio", "Ratio Follow", "Parse OK")
    FUNCTION = "enhance"
    CATEGORY = "MiniMax H3/Prompt"
    DESCRIPTION = "Official Qwen-Image-2.1 prompt enhancer (PE-T2I / PE-I2I) for local GGUF or online LLMs."

    def enhance(self, **inputs):
        task = _task_key(inputs.get("Task", TASK_LABELS[0]))
        profile = PE_PROFILES[task]
        prompt = (inputs.get("Prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt 不能为空。")

        images = [
            inputs[f"Image {index}"]
            for index in range(1, MAX_INPUT_IMAGES + 1)
            if inputs.get(f"Image {index}") is not None
        ]
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
        max_tokens = _completion_budget(context_length, requested)
        if max_tokens < requested:
            print(
                f"[Qwen Image 2.1 PE] 上下文 {context_length} 放不下官方 {requested} 个新 token，"
                f"本次上限收敛为 {max_tokens}。需要更长输出请调大 Context Length。"
            )

        system_prompt = _load_system_prompt(task, inputs.get("System Prompt File", ""))
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

        online = _is_online_source(inputs.get("Model Source", False))
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
        if not parsed["parse_ok"]:
            print(
                "[Qwen Image 2.1 PE] 警告：回答不是预期的 JSON，Positive Prompt 已回退为原始回答文本。"
                "常见的三个原因：选错了权重（要用 PE-T2I / PE-I2I，而不是基座模型）、"
                "系统提示词与权重不匹配、或输出被上下文长度截断。"
            )
        else:
            print(f"[Qwen Image 2.1 PE] 解析成功，画布：{parsed['wh_ratio'] or parsed['ratio_follow'] or '未指定'}")
        return (
            parsed["positive_prompt"],
            parsed["wh_ratio"],
            parsed["ratio_follow"],
            parsed["parse_ok"],
        )
