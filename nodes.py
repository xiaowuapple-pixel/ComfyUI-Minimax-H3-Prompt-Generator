import base64
import gc
import io
import re
import time
import json
import urllib.request
import os
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import comfy.model_management as mm
import folder_paths
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import (
        Gemma3ChatHandler,
        Gemma4ChatHandler,
        MTMDChatHandler,
        Qwen35ChatHandler,
        Qwen3VLChatHandler,
    )
    _LLAMA_CPP_IMPORT_ERROR = None
except ImportError as exc:
    # Online API mode and node registration do not require llama-cpp-python.
    Llama = None
    _LLAMA_CPP_IMPORT_ERROR = exc


SECTION_NAMES = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


SYSTEM_PROMPT = """你是 MiniMax H3 全参考模式（Ref2VA）视频提示词编写专家。依据已经完成的参考图
视觉分析和用户描述，输出可以直接用于 MiniMax H3 的完整中文视频提示词。以下规则来自官方
MiniMax H3 h3-prompt-writing Skill 的 Ref2VA 完整规范，不得压缩为故事梗概。

只输出提示词正文，不要解释、分析过程、Markdown 标题或代码块。必须严格按以下六个字段及顺序输出：
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

除上述官方字段名和下面规定的格式标记外，描述语言由节点的“输出中文提示词”开关决定。

规则：
1. 为画面中会被复用的人物、动物、物体、场景、服装或视觉风格建立稳定的 <Subject N> 标签，
   并在定义中注明来源 <Picture N>。同一个主体出现在多张图中时合并定义，不要重复编号。
2. 只有图片被指定为首帧、尾帧、关键帧、构图锚点或分镜参考时，才单独定义 <Picture N>；
   仅用于身份、场景或风格参考时，把图片来源写入对应的 <Subject N> 定义中。
3. summary 必须是一小段概述，并以官方任务类型 [reference generation]、
   [keyframe completion] 或两者用“ + ”连接作为开头。
4. retention_analysis 每个已定义标签各占一行。视觉关系标记只能使用
   fully_preserved、partially_preserved、attribute_transfer、weak_reference。
5. detailed_description 开头先用一至两句确定整体视觉风格。随后按播放顺序写镜头：
   [Shot 1] 不写时间戳；后续镜头必须使用 [Shot N] At MM:SS.mmm, 标明切换时间。
6. 镜头时间线必须严格适配用户指定的总时长。具体描述构图、主体动作、表情、动作连续性、
   景别、机位、运镜幅度与速度、光线、环境变化和声音同步，避免物理上互相冲突的动作。
7. detailed_description 是输出主体，长度以能清楚表达完整动作和镜头为准，通常约 250 至 800 个汉字，
   不要为了凑字数强行增加镜头或重复描述。每个镜头明确交代必要的构图、主体动作、环境、运镜和声音；
   短视频可以只有一个镜头。
8. 有真实对白时，按首次发声顺序分配稳定的 (S1)、(S2) 编号，并使用
   <d>[Chinese]对白原文</d>。用户没有要求对白时不要擅自添加。
9. overall_soundscape 只总结环境声和物理音效，不重复对白，也不写观众才能听见的配乐。
10. non_diegetic_music 描述非画内配乐的乐器、速度和动态发展；不需要配乐时写 N/A。
11. 不虚构图片中无法支持的身份、品牌、文字或关键外貌。用户描述含糊时，将其补全为连贯、
    可拍摄、符合指定时长和画幅的镜头方案。
12. subject_definitions 中每个主体必须写清来源、可辨识外貌或视觉特征及参考作用。summary 要交代
    主体关系、完整动作走向、镜头数量和参考用途。retention_analysis 每行必须写出现镜头、固定关系
    标记，并在连字符后具体解释保留或迁移了哪些特征，不能只写一个标记。
13. 人物身份、服装、比例、场景空间关系和光色必须跨镜头一致。后续镜头使用同一标签，不重新定义。
14. 最后自行检查：六个字段齐全且有内容；所有主体标签在镜头正文中实际出现；镜头时间严格递增且
    小于总时长；每个动作在分配时间内可完成；声音和配乐没有被放错字段。
15. 禁止用改写近义词的方式反复描述同一外貌、同一动作或同一构图。每个镜头必须推进新的动作状态、
    机位信息或声音事件；已经在 subject_definitions 确立的静态特征，正文只在首次出场时完整描述。
"""


ANALYSIS_PROMPT = """你是电影分镜策划和视觉分析师。逐张仔细观察参考图片，为后续编写 MiniMax H3
全参考视频提示词制作一份详尽的内部创作资料。这一步不要写最终六段提示词。

必须包含：
1. 每张 <Picture N> 中可见人物、动物、物体、服装、姿态、表情、材质、颜色、文字、场景、光线、
   构图、画风和镜头视角；不确定的信息明确标注，不要猜测身份。
2. 判断跨图片是否为同一主体；列出应该建立的 <Subject N>、各自图片来源和必须保持的一致特征。
3. 结合用户描述设计完整时间线：每个镜头的起止时间、景别、主体位置、连续动作、表情变化、运镜、
   转场、环境变化、同步音效和配乐发展。动作量必须适配总时长。
4. 指出每张图片只是身份/风格参考，还是具体首帧、关键帧、尾帧或分镜锚点。
5. 给出需要避免的身份漂移、服装变化、空间跳变、动作冲突和无依据细节。

写得具体、完整，供下一阶段直接扩写，不要输出客套话。"""


GENERATION_TYPE_INSTRUCTIONS = {
    "自动判别": "根据输入图片数量、图片用途和用户描述自动判别最合适的 H3 生成类型。",
    "文生视频": "按文生视频处理，不把参考图片当作必须复现的首帧或尾帧；若提供图片，只将其作为风格或身份参考。",
    "图生视频": "按图生视频处理，将第一张参考图作为主要视觉起点，保持主体、构图和风格连续。",
    "首尾帧生成": "按首尾帧生成处理：第一张参考图作为首帧，最后一张参考图作为尾帧，中间动作和镜头必须连贯过渡。",
    "尾帧生成": "按尾帧生成处理：最后一张参考图作为目标尾帧，设计动作和镜头使视频自然收束到该画面。",
    "多参考生成": "按多参考生成处理：综合全部参考图片中的主体、场景、风格和细节，保持跨镜头一致，不擅自将图片解释为首帧或尾帧。",
}
GENERATION_TYPE_INSTRUCTIONS.update({
    "Auto Detect": GENERATION_TYPE_INSTRUCTIONS["自动判别"],
    "Text-to-Video": GENERATION_TYPE_INSTRUCTIONS["文生视频"],
    "Image-to-Video": GENERATION_TYPE_INSTRUCTIONS["图生视频"],
    "First/Last Frame": GENERATION_TYPE_INSTRUCTIONS["首尾帧生成"],
    "Last Frame": GENERATION_TYPE_INSTRUCTIONS["尾帧生成"],
    "Multi-Reference": GENERATION_TYPE_INSTRUCTIONS["多参考生成"],
})

CREATIVE_SKILL_INSTRUCTIONS = {
    "自动判别": "根据用户描述和参考图片，自动选择最适合的创作技能，并保持 H3 提示词格式完整。",
    "通用 H3 提示词": "使用通用 MiniMax H3 视频提示词方法，优先保证主体一致、动作连贯、镜头可执行和声音完整。",
    "3D 动画短片": "采用 3D 动画短片技能：明确角色与材质、三维空间关系、灯光、镜头运动和可执行的动画节奏。",
    "品牌宣传片": "采用品牌宣传片技能：突出品牌或产品主体、卖点视觉化、品牌调性、商业镜头语言和清晰的结尾展示。",
    "合作游戏片头": "采用合作游戏片头技能：突出两名角色的协作关系、能力互补、动作节奏、冲突升级和具有辨识度的片头收束。",
    "手绘实拍融合": "采用手绘实拍融合技能：保持真实场景的摄影质感，同时设计手绘线条、涂鸦或发光笔触与实拍动作的互动。",
    "极简产品广告": "采用极简产品广告技能：使用干净构图、克制背景、明确产品材质与功能展示、精确运镜和高级商业光线。",
    "音乐字幕视频": "采用音乐字幕视频技能：根据音乐情绪安排画面节奏、歌词或字幕出现时机、排版位置、转场和可读性。",
    "纸张拼贴科普": "采用纸张拼贴科普技能：用纸张、剪纸、拼贴和手工材质表达知识点，保持层次清楚、动作连续且易于理解。",
    "纸艺定格科普": "采用纸艺定格科普技能：设计纸艺角色和场景的逐格运动、手工纹理、镜头节奏，并把知识讲解转化为可视化动作。",
}
CREATIVE_SKILL_INSTRUCTIONS.update({
    "Auto Detect": CREATIVE_SKILL_INSTRUCTIONS["自动判别"],
    "General H3 Prompt": CREATIVE_SKILL_INSTRUCTIONS["通用 H3 提示词"],
    "3D Animated Short": CREATIVE_SKILL_INSTRUCTIONS["3D 动画短片"],
    "Brand Promo": CREATIVE_SKILL_INSTRUCTIONS["品牌宣传片"],
    "Co-op Game Intro": CREATIVE_SKILL_INSTRUCTIONS["合作游戏片头"],
    "Hand-drawn Live Action": CREATIVE_SKILL_INSTRUCTIONS["手绘实拍融合"],
    "Minimalist Product Ad": CREATIVE_SKILL_INSTRUCTIONS["极简产品广告"],
    "Music Subtitle Video": CREATIVE_SKILL_INSTRUCTIONS["音乐字幕视频"],
    "Paper Collage Explainer": CREATIVE_SKILL_INSTRUCTIONS["纸张拼贴科普"],
    "Papercraft Stop-motion Explainer": CREATIVE_SKILL_INSTRUCTIONS["纸艺定格科普"],
})

OFFICIAL_SKILL_PATHS = {
    "通用 H3 提示词": "h3-prompt-writing",
    "3D 动画短片": "3d-animation-short-generator",
    "品牌宣传片": "brand-promo-video-generator",
    "合作游戏片头": "co-op-game-intro-generator",
    "手绘实拍融合": "handdrawn-live-video-generator",
    "极简产品广告": "minimalist-product-ad-generator",
    "音乐字幕视频": "music-video-subtitle-generator",
    "纸张拼贴科普": "paper-collage-explainer-generator",
    "纸艺定格科普": "papercraft-stop-motion-explainer",
}


def _official_skill_instruction(skill_name):
    """Read the vendored MiniMax-H3 skill snapshot; never fetch at runtime."""
    folder = OFFICIAL_SKILL_PATHS.get(skill_name)
    if not folder:
        return CREATIVE_SKILL_INSTRUCTIONS.get(skill_name, CREATIVE_SKILL_INSTRUCTIONS["自动判别"])
    skill_file = Path(__file__).parent / "skills" / folder / "SKILL.md"
    try:
        text = skill_file.read_text(encoding="utf-8").strip()
        if text:
            return "以下是本地安装的 MiniMax-H3 官方技能文件内容，请严格按其要求执行：\n\n" + text
    except OSError:
        pass
    return CREATIVE_SKILL_INSTRUCTIONS.get(skill_name, CREATIVE_SKILL_INSTRUCTIONS["自动判别"])


def _is_online_source(value):
    """Accept the new boolean switch and legacy saved string values."""
    if isinstance(value, str):
        return value.strip().lower() in {"online", "true", "1", "在线 openai 兼容 api", "在线llm"}
    return bool(value)


def _input_value(inputs, name, legacy_name=None, default=None):
    if name in inputs:
        return inputs[name]
    if legacy_name and legacy_name in inputs:
        return inputs[legacy_name]
    return default


def _llm_directories():
    try:
        roots = folder_paths.get_folder_paths("LLM")
    except Exception:
        roots = []
    local = Path(folder_paths.models_dir) / "LLM"
    result = [Path(p) for p in roots] if roots else [local]
    if local not in result:
        result.insert(0, local)
    return [p for p in result if p.exists()]


def _resolve_llm_path(relative_name):
    candidate = Path(relative_name)
    for root in _llm_directories():
        path = root / candidate
        if path.is_file():
            return path
    return None


def _language_models():
    models = []
    for root in _llm_directories():
        for path in root.rglob("*.gguf"):
            if "mmproj" not in path.name.lower():
                models.append(path.relative_to(root).as_posix())
    return sorted(models, key=str.lower)


def _vision_models():
    models = []
    for root in _llm_directories():
        for path in root.rglob("*.gguf"):
            if "mmproj" in path.name.lower():
                models.append(path.relative_to(root).as_posix())
    return sorted(models, key=str.lower)


def _create_chat_handler(model_name, mmproj_path):
    if _LLAMA_CPP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "本地 GGUF 模式需要安装 llama-cpp-python>=0.3.46；"
            "在线 OpenAI 兼容 API 模式无需此依赖。"
        ) from _LLAMA_CPP_IMPORT_ERROR
    name = Path(model_name).name.lower()
    common = {
        "clip_model_path": str(mmproj_path),
        "image_min_tokens": 0,
        "image_max_tokens": 0,
        "verbose": False,
    }
    if "qwen3.6" in name or "qwen3.5" in name:
        return Qwen35ChatHandler(enable_thinking=False, **common)
    if "qwen3-vl" in name or "qwen3_vl" in name:
        return Qwen3VLChatHandler(force_reasoning=False, **common)
    if "gemma-4" in name or "gemma4" in name:
        return Gemma4ChatHandler(enable_thinking=False, **common)
    if "gemma-3" in name or "gemma3" in name:
        return Gemma3ChatHandler(**common)
    return MTMDChatHandler(use_gpu=True, **common)


class _VisionRuntime:
    llm = None
    chat_handler = None

    @classmethod
    def load(cls, model_relative_path, vision_relative_path, gpu_layers):
        if _LLAMA_CPP_IMPORT_ERROR is not None:
            raise RuntimeError(
                "本地 GGUF 模式需要安装 llama-cpp-python>=0.3.46；"
                "在线 OpenAI 兼容 API 模式无需此依赖。"
            ) from _LLAMA_CPP_IMPORT_ERROR
        cls.close()
        mm.unload_all_models()
        model_path = _resolve_llm_path(model_relative_path)
        if model_path is None:
            raise FileNotFoundError(f"语言模型不存在（已搜索全部 LLM 路径）：{model_relative_path}")
        mmproj_path = _resolve_llm_path(vision_relative_path)
        if mmproj_path is None:
            raise FileNotFoundError(f"视觉模型不存在（已搜索全部 LLM 路径）：{vision_relative_path}")

        print(f"[H3 中文提示词] 语言模型：{model_path.name}")
        print(f"[H3 中文提示词] 视觉模型：{mmproj_path.name}")
        print(f"[H3 中文提示词] GPU 卸载层数：{gpu_layers}")
        try:
            cls.chat_handler = _create_chat_handler(model_relative_path, mmproj_path)
            cls.llm = Llama(
                model_path=str(model_path),
                chat_handler=cls.chat_handler,
                n_gpu_layers=gpu_layers,
                n_ctx=12288,
                verbose=False,
            )
        except Exception:
            cls.close()
            raise
        return cls.llm

    @classmethod
    def close(cls):
        if cls.llm is not None:
            try:
                cls.llm.close()
            except Exception:
                pass
        if cls.chat_handler is not None:
            try:
                cls.chat_handler._exit_stack.close()
            except Exception:
                pass
        cls.llm = None
        cls.chat_handler = None
        gc.collect()
        mm.soft_empty_cache()


class _OnlineRuntime:
    """Minimal OpenAI-compatible client for hosted text/vision models."""
    def __init__(self, base_url, api_key, model):
        self.base_url = (base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise ValueError("在线模式必须填写请求地址。")
        # Accept common OpenAI-compatible base URL forms without creating
        # duplicate /v1 segments (for example /api/v1).
        normalized = self.base_url.lower()
        if not (normalized.endswith("/v1") or normalized.endswith("/api/v1")):
            self.base_url += "/v1"
        self.api_key = api_key.strip()
        self.model = model.strip() or self._first_model()

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
        }, method="GET" if payload is None else "POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")

    def _first_model(self):
        if not self.api_key:
            raise ValueError("在线模式必须填写 API Key，或直接填写模型名。")
        try:
            result = json.loads(self._request("/models"))
            models = result.get("data", [])
            if not models:
                raise ValueError("在线平台 /models 没有返回可用模型。")
            return models[0].get("id") or models[0].get("name")
        except Exception as exc:
            raise RuntimeError(f"无法从在线平台拉取模型：{exc}") from exc

    def create_chat_completion(self, messages, stream=True, **parameters):
        payload = {"model": self.model, "messages": messages, "stream": bool(stream)}
        payload.update({k: v for k, v in parameters.items() if k in {
            "temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed"
        }})
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
        }, method="POST")
        response = urllib.request.urlopen(request, timeout=600)
        if not stream:
            body = json.loads(response.read().decode("utf-8"))
            return iter([body])
        def chunks():
            with response:
                for raw in response:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
        return chunks()


def _tensor_to_data_url(tensor, max_size=512):
    frame = tensor[0] if tensor.ndim == 4 else tensor
    if frame.ndim != 3:
        raise ValueError(f"图片张量格式无效：{tuple(frame.shape)}")
    array = np.clip(frame.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    width, height = image.size
    scale = min(float(max_size) / max(width, height), 1.0)
    if scale < 1.0:
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(target, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _collect_images(inputs, allow_empty=False):
    images = []
    for index in range(1, 10):
        tensor = inputs.get(f"Image {index}", inputs.get(f"Reference Image {index}", inputs.get(f"图片_{index}")))
        if tensor is None:
            continue
        if tensor.ndim == 3:
            images.append(tensor)
        elif tensor.ndim == 4:
            images.extend(tensor[item] for item in range(tensor.shape[0]))
        else:
            raise ValueError(f"图片_{index} 的张量格式无效：{tuple(tensor.shape)}")
    if not images and not allow_empty:
        raise ValueError("至少需要输入一张参考图片。")
    if len(images) > 9:
        raise ValueError(f"最多支持 9 张图片，当前收到 {len(images)} 张。")
    return images


def _clean_output(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _section_pattern(section):
    # Accept Markdown emphasis, headings, Chinese colons, and content on the heading line.
    return rf"(?im:^\s*(?:[#>*-]+\s*)?(?:\*\*|__)?{re.escape(section)}(?:\*\*|__)?\s*[:\uFF1A](?:\*\*|__)?)"


def _normalize_sections(text):
    normalized = text
    for section in SECTION_NAMES:
        normalized = re.sub(_section_pattern(section), f"{section}:", normalized, count=1)

    present = [section for section in SECTION_NAMES if re.search(_section_pattern(section), text)]
    if present and len(present) < len(SECTION_NAMES):
        for section in SECTION_NAMES:
            if section not in present:
                normalized = normalized.rstrip() + f"\n\n{section}:\nN/A"
    return normalized.strip()


def _h3_section_count(text):
    return sum(bool(re.search(_section_pattern(section), text)) for section in SECTION_NAMES)


def _looks_like_internal_text(text):
    lowered = text.lower()
    markers = (
        "note for internal use",
        "internal use",
        "preliminary discussion",
        "inner logic",
        "chain of thought",
        "your request (the",
    )
    return sum(marker in lowered for marker in markers) >= 2


def _stream_completion(llm, messages, stage, **parameters):
    started = time.perf_counter()
    pieces = []
    token_count = 0
    print(f"[H3 中文提示词] 开始{stage}...")
    stream = llm.create_chat_completion(messages=messages, stream=True, **parameters)
    for chunk in stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if not delta:
            continue
        pieces.append(delta)
        token_count += 1
        if token_count % 128 == 0:
            elapsed = time.perf_counter() - started
            print(f"[H3 中文提示词] {stage}已生成约 {token_count} tokens，耗时 {elapsed:.1f} 秒")
    elapsed = time.perf_counter() - started
    print(f"[H3 中文提示词] {stage}完成，共约 {token_count} tokens，耗时 {elapsed:.1f} 秒")
    return _clean_output("".join(pieces))


def _format_error(text):
    positions = []
    for section in SECTION_NAMES:
        match = re.search(_section_pattern(section), text)
        if match is None:
            return f"缺少字段 {section}"
        positions.append(match.start())
    if positions != sorted(positions):
        return "六个字段的顺序不正确"
    return None


def _section_text(text, section):
    index = SECTION_NAMES.index(section)
    next_section = SECTION_NAMES[index + 1] if index + 1 < len(SECTION_NAMES) else None
    pattern = r"(?s:" + _section_pattern(section) + r"\s*(.*?))"
    pattern += rf"(?={_section_pattern(next_section)}|\Z)" if next_section else r"\Z"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _replace_section(text, section, replacement):
    index = SECTION_NAMES.index(section)
    next_section = SECTION_NAMES[index + 1] if index + 1 < len(SECTION_NAMES) else None
    pattern = r"(?s:" + _section_pattern(section) + r"\s*.*?)"
    pattern += rf"(?={_section_pattern(next_section)}|\Z)" if next_section else r"\Z"
    return re.sub(pattern, f"{section}:\n{replacement.strip()}\n\n", text, count=1).strip()


def _detail_is_short(text, duration):
    details = _section_text(text, "detailed_description")
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", details))
    target = max(180, min(700, round(float(duration) * 45)))
    shots = len(re.findall(r"\[Shot\s+\d+\]", details, flags=re.IGNORECASE))
    minimum_shots = 1 if duration <= 6 else 2 if duration <= 15 else 3
    return chinese_count < target or shots < minimum_shots


def _quality_errors(text, duration):
    errors = []
    format_error = _format_error(text)
    if format_error:
        return [format_error]

    summary = _section_text(text, "summary")
    retention = _section_text(text, "retention_analysis")
    details = _section_text(text, "detailed_description")
    soundscape = _section_text(text, "overall_soundscape")

    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", details))
    minimum_chinese = max(180, min(700, round(float(duration) * 45)))
    if chinese_count < minimum_chinese:
        errors.append(f"detailed_description 只有约 {chinese_count} 个汉字，至少需要 {minimum_chinese} 个")

    shot_count = len(re.findall(r"\[Shot\s+\d+\]", details, flags=re.IGNORECASE))
    minimum_shots = 1 if duration <= 6 else 2 if duration <= 15 else 3
    if shot_count < minimum_shots:
        errors.append(f"只有 {shot_count} 个镜头，当前时长至少需要 {minimum_shots} 个有明确分工的镜头")

    if len(re.findall(r"[\u4e00-\u9fff]", summary)) < 60:
        errors.append("summary 没有完整概述主体关系、动作走向和参考用途")

    retention_lines = [line.strip() for line in retention.splitlines() if line.strip()]
    relationship = (
        r"(?:fully_preserved|partially_preserved|attribute_transfer|weak_reference|"
        r"fully_copy|partially_copy|reference)"
    )
    weak_retention = []
    for line in retention_lines:
        marker = re.search(relationship, line)
        explanation = line[marker.end():] if marker else ""
        explanation = re.sub(r"^[\s:：\-—–]+", "", explanation)
        if marker is None or len(re.findall(r"[\u4e00-\u9fff]", explanation)) < 8:
            weak_retention.append(line)
    if weak_retention:
        errors.append("retention_analysis 存在只写关系标记、没有具体保留说明的条目")

    subjects = set(re.findall(r"<Subject\s+\d+>", _section_text(text, "subject_definitions")))
    missing_subjects = sorted(subject for subject in subjects if subject not in details)
    if missing_subjects:
        errors.append("镜头正文没有实际使用这些主体标签：" + "、".join(missing_subjects))

    if soundscape.upper() != "N/A" and len(re.findall(r"[\u4e00-\u9fff]", soundscape)) < 25:
        errors.append("overall_soundscape 过于简略，没有覆盖持续环境声和关键物理音效")

    clauses = re.split(r"[。！？!?；;\n]+", details)
    seen_clauses = set()
    repeated_clauses = []
    for clause in clauses:
        normalized = re.sub(r"\s+", "", clause)
        normalized = re.sub(r"<Subject\s+\d+>|\[Shot\s+\d+\]|At\d{2}:\d{2}\.\d{3},?", "", normalized)
        if len(normalized) < 24:
            continue
        if normalized in seen_clauses:
            repeated_clauses.append(normalized[:36])
        seen_clauses.add(normalized)
    if repeated_clauses:
        errors.append("detailed_description 存在重复长句：" + "、".join(repeated_clauses[:3]))
    return errors


class H3SaveImage:
    """Save images as PNG/JPG/WEBP, with optional workflow sidecar JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "Filename Prefix": ("STRING", {"default": "H3", "multiline": False}),
                "Output Folder": ("STRING", {"default": "", "multiline": False, "placeholder": "Relative to ComfyUI/output"}),
                "Format": (["PNG", "JPG", "WEBP"], {"default": "PNG"}),
                "JPG Quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "Save Workflow JSON": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Output"

    @staticmethod
    def _safe_output_dir(folder):
        root = os.path.abspath(folder_paths.get_output_directory())
        folder = (folder or "").strip().replace("/", os.sep).replace("\\", os.sep)
        # Relative folders stay under ComfyUI/output. Absolute folders are
        # supported intentionally so users can save to another drive.
        target = os.path.abspath(folder) if os.path.isabs(folder) else (
            os.path.abspath(os.path.join(root, folder)) if folder else root
        )
        if not os.path.isabs(folder) and os.path.commonpath((root, target)) != root:
            raise ValueError("Relative Output Folder must stay inside ComfyUI/output; use an absolute path for another drive.")
        os.makedirs(target, exist_ok=True)
        return target

    @staticmethod
    def _prefix(value):
        value = (value or "H3").strip().replace("/", "_").replace("\\", "_")
        value = re.sub(r"[^\w .()\-\u4e00-\u9fff]", "_", value).strip(" .")
        return value or "H3"

    def save_images(self, images, **inputs):
        output_dir = self._safe_output_dir(inputs.get("Output Folder", ""))
        prefix = self._prefix(inputs.get("Filename Prefix", "H3"))
        fmt = inputs.get("Format", "PNG").upper()
        quality = int(inputs.get("JPG Quality", 95))
        save_json = bool(inputs.get("Save Workflow JSON", False))
        prompt = inputs.get("prompt")
        extra = inputs.get("extra_pnginfo") or {}
        workflow = extra.get("workflow") if isinstance(extra, dict) else None
        results = []
        extension = {"PNG": ".png", "JPG": ".jpg", "WEBP": ".webp"}[fmt]
        for index, tensor in enumerate(images, start=1):
            array = np.clip(tensor.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(array, "RGB")
            stem = f"{prefix}_{index:05d}"
            path = os.path.join(output_dir, stem + extension)
            counter = 1
            while os.path.exists(path):
                path = os.path.join(output_dir, f"{stem}_{counter:03d}{extension}")
                counter += 1
            save_kwargs = {}
            if fmt == "PNG":
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
                if isinstance(extra, dict):
                    for key, value in extra.items():
                        metadata.add_text(str(key), json.dumps(value, ensure_ascii=False))
                save_kwargs = {"pnginfo": metadata, "compress_level": 4}
            elif fmt == "JPG":
                save_kwargs = {"quality": quality, "optimize": True}
            else:
                save_kwargs = {"quality": quality, "method": 6}
            # Pillow registers the JPEG encoder as "JPEG", while the UI uses
            # the friendlier "JPG" label.
            image.save(path, format="JPEG" if fmt == "JPG" else fmt, **save_kwargs)
            try:
                subfolder = os.path.relpath(output_dir, folder_paths.get_output_directory())
            except ValueError:
                # Cross-drive paths have no Windows relative path. The file
                # is saved successfully; ComfyUI's preview route cannot serve
                # an external drive, so leave this field empty.
                subfolder = ""
            results.append({"filename": os.path.basename(path), "subfolder": subfolder, "type": "output"})
            if save_json and isinstance(workflow, dict):
                json_path = os.path.splitext(path)[0] + ".json"
                with open(json_path, "w", encoding="utf-8") as handle:
                    json.dump(workflow, handle, ensure_ascii=False, indent=2)
        return {"ui": {"images": results}}


class Qwen36MultiImageH3ChinesePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        models = _language_models() or ["未找到语言模型"]
        vision_models = _vision_models() or ["未找到视觉模型"]
        return {
            "required": {
                "Language Model": (models,),
                "Vision Model": (vision_models,),
                "GPU Offload Layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "-1=全部放入显存，0=全部使用内存/CPU。16GB 显存建议从 16-24 开始。",
                    },
                ),
                "Seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "可在生成后控制中选择随机、递增或固定。",
                    },
                ),
                "Description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "简单说明想要的情节、动作、运镜或声音。",
                    },
                ),
                "Video Duration": (
                    "FLOAT",
                    {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.5},
                ),
                "Aspect Ratio": (
                    ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    {"default": "16:9"},
                ),
                "Model Source": ("BOOLEAN", {"default": False, "label_on": "Online LLM", "label_off": "Local Model"}),
                "Online Request URL": ("STRING", {"default": "https://api.openai.com/v1", "multiline": False}),
                "Online API Key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "Online Model": ("STRING", {"default": "", "multiline": False, "placeholder": "Click refresh to load models"}),
                "Generation Type": (
                    ["Auto Detect", "Text-to-Video", "Image-to-Video", "First/Last Frame", "Last Frame", "Multi-Reference"],
                    {"default": "Auto Detect"},
                ),
                "Output Chinese Prompt": ("BOOLEAN", {"default": False}),
                "Creative Skill": (
                    [
                        "Auto Detect", "General H3 Prompt", "3D Animated Short", "Brand Promo", "Co-op Game Intro",
                        "Hand-drawn Live Action", "Minimalist Product Ad", "Music Subtitle Video", "Paper Collage Explainer", "Papercraft Stop-motion Explainer",
                    ],
                    {"default": "自动判别"},
                ),
                "Unload Model After Generation": ("BOOLEAN", {"default": True}),
            },
            "optional": {f"Image {index}": ("IMAGE",) for index in range(1, 10)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("H3 Prompt",)
    FUNCTION = "生成"
    CATEGORY = "MiniMax H3/Prompt"
    DESCRIPTION = "Generate MiniMax H3 prompts with local GGUF or online vision-language models."

    def 生成(self, **inputs):
        online_source = _is_online_source(_input_value(inputs, "Model Source", "模型来源", False))
        model_name = _input_value(inputs, "Language Model", "语言模型")
        vision_name = _input_value(inputs, "Vision Model", "视觉模型")
        gpu_layers = _input_value(inputs, "GPU Offload Layers", "GPU卸载层数", -1)
        seed = _input_value(inputs, "Seed", "种子", 0)
        generation_type = _input_value(inputs, "Generation Type", "生成类型", "Auto Detect")
        if not online_source and (model_name == "未找到语言模型" or vision_name == "未找到视觉模型"):
            raise FileNotFoundError("请将 GGUF 语言模型和对应 mmproj 视觉模型放入 models/LLM。")
        images = _collect_images(inputs, allow_empty=generation_type in {"自动判别", "文生视频", "Auto Detect", "Text-to-Video"})
        # With no reference image, automatic mode is a true text-to-video task.
        if not images and generation_type in {"自动判别", "Auto Detect"}:
            generation_type = "Text-to-Video"
        if generation_type in {"图生视频", "首尾帧生成", "尾帧生成", "多参考生成", "Image-to-Video", "First/Last Frame", "Last Frame", "Multi-Reference"} and not images:
            raise ValueError(f"生成类型“{generation_type}”至少需要输入一张参考图片。")
        if generation_type in {"首尾帧生成", "First/Last Frame"} and len(images) < 2:
            raise ValueError("首尾帧生成至少需要输入两张图片，分别作为首帧和尾帧。")
        duration = _input_value(inputs, "Video Duration", "视频时长", 10.0)
        aspect_ratio = _input_value(inputs, "Aspect Ratio", "画面比例", "16:9")
        generation_instruction = GENERATION_TYPE_INSTRUCTIONS.get(
            generation_type, GENERATION_TYPE_INSTRUCTIONS["自动判别"]
        )
        creative_skill = _input_value(inputs, "Creative Skill", "创意技能", "Auto Detect")
        creative_instruction = _official_skill_instruction(creative_skill)
        unload_after_generation = bool(_input_value(inputs, "Unload Model After Generation", "生成后卸载模型", True))
        output_chinese = bool(_input_value(inputs, "Output Chinese Prompt", "输出中文提示词", False))
        language_instruction = (
            "输出必须使用简体中文（官方字段名和格式标记保持不变）。"
            if output_chinese
            else "输出必须使用英文（官方字段名和格式标记保持不变）；不要翻译字段名。"
        )
        description = _input_value(inputs, "Description", "简单描述", "").strip() or (
            "根据文字描述创作连贯、自然、有电影感的视频。" if not images
            else "根据参考图片创作连贯、自然、有电影感的视频。"
        )

        analysis_request = (
            f"目标视频总时长：{duration:g} 秒\n目标画面比例：{aspect_ratio}\n"
            f"参考图片数量：{len(images)}\n用户的简单描述：{description}\n"
            f"指定生成类型：{generation_type}\n生成类型要求：{generation_instruction}"
            f"\n指定创意技能：{creative_skill}\n创意技能要求：{creative_instruction}"
        )
        content = [{"type": "text", "text": analysis_request}]
        if not images:
            content[0]["text"] += "\n当前没有任何参考图片，这是纯文生视频任务；请只依据文字描述规划镜头，不要虚构图片内容。"
        for index, image in enumerate(images, start=1):
            content.append({"type": "text", "text": f"下一张参考图片是 <Picture {index}>。"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _tensor_to_data_url(image)},
                }
            )

        if online_source:
            api_key = _input_value(inputs, "Online API Key", "在线APIKey", "")
            if not api_key.strip():
                raise ValueError("在线模式必须填写 API Key。")
            llm = _OnlineRuntime(
                _input_value(inputs, "Online Request URL", "在线请求地址", ""),
                api_key,
                _input_value(inputs, "Online Model", "在线模型", ""),
            )
        else:
            llm = _VisionRuntime.load(model_name, vision_name, gpu_layers)
        try:
            analysis_messages = [
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": content},
            ]
            visual_plan = _stream_completion(
                llm,
                messages=analysis_messages,
                stage="参考图分析",
                seed=seed,
                max_tokens=1024,
                temperature=0.25,
                top_p=0.9,
                repeat_penalty=1.08,
                frequency_penalty=0.08,
            )

            final_request = (
                f"目标视频总时长：{duration:g} 秒\n目标画面比例：{aspect_ratio}\n"
                f"指定生成类型：{generation_type}\n生成类型要求：{generation_instruction}\n"
                f"指定创意技能：{creative_skill}\n创意技能要求：{creative_instruction}\n"
                f"用户的简单描述：{description}\n\n"
                "长度原则：以清楚表达用户需求为准，保持自然、适中的篇幅；不要为了满足字数、镜头数量或技能模板而强行扩写、堆砌细节或重复内容。\n\n"
                "下面是已经根据全部参考图片完成的内部视觉分析和分镜策划。充分使用其中的具体视觉细节，"
                "但不要在最终输出中提及‘分析’或‘资料’：\n\n"
                f"{visual_plan}\n\n"
                "现在严格按照六段 Ref2VA 格式写出最终提示词。" + language_instruction
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n当前语言要求：" + language_instruction},
                {"role": "user", "content": final_request},
            ]
            prompt = _stream_completion(
                llm,
                messages=messages,
                stage="最终提示词",
                seed=(seed + 1) & 0xFFFFFFFFFFFFFFFF,
                max_tokens=3072,
                temperature=0.45,
                top_p=0.9,
                repeat_penalty=1.12,
                frequency_penalty=0.12,
            )
            section_count = _h3_section_count(prompt)
            if section_count < 2 or _looks_like_internal_text(prompt):
                raise RuntimeError(
                    "所选语言模型没有遵循 H3 写作指令，返回了内部说明或无关文本。"
                    "这通常是 Uncensored 微调模型的指令遵循问题，请更换 Instruct 模型或更换种子。"
                )
            prompt = _normalize_sections(prompt)
            if _detail_is_short(prompt, duration) and len(_section_text(prompt, "detailed_description")) < 120:
                expansion_request = (
                    "只重写下面两个字段：retention_analysis 和 detailed_description。"
                    "不要输出其他字段、解释、Markdown 或内部分析。\n\n"
                    "retention_analysis 必须逐个使用 <Subject N> 或 <Picture N>，写明出现镜头，"
                    "并严格使用 fully_preserved、partially_preserved、attribute_transfer 或 weak_reference，"
                    "格式为：<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - 中文具体说明。"
                    "禁止 [P1]、箭头、百分比或数字列表。\n\n"
                    f"detailed_description 写到约 {max(180, min(700, round(duration * 45)))} 个中文汉字即可，"
                    f"为 {duration:g} 秒视频安排自然数量的镜头（通常 1-3 个）。"
                    "只补充必要的构图、动作、环境、运镜和声音，不要为了长度重复或过度复杂化。\n\n"
                    f"目标画幅：{aspect_ratio}\n用户描述：{description}\n\n"
                    f"视觉资料：\n{visual_plan}\n\n"
                    f"主体定义：\n{_section_text(prompt, 'subject_definitions')}\n\n"
                    f"当前保留分析：\n{_section_text(prompt, 'retention_analysis')}\n\n"
                    f"当前过短正文：\n{_section_text(prompt, 'detailed_description')}"
                )
                expansion = _stream_completion(
                    llm,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": expansion_request},
                    ],
                    stage="正文补写",
                    seed=(seed + 2) & 0xFFFFFFFFFFFFFFFF,
                    max_tokens=2048,
                    temperature=0.4,
                    top_p=0.9,
                    repeat_penalty=1.15,
                    frequency_penalty=0.15,
                )
                expansion = _normalize_sections(expansion)
                new_retention = _section_text(expansion, "retention_analysis")
                new_details = _section_text(expansion, "detailed_description")
                if new_retention and new_retention.upper() != "N/A":
                    prompt = _replace_section(prompt, "retention_analysis", new_retention)
                if new_details and new_details.upper() != "N/A":
                    prompt = _replace_section(prompt, "detailed_description", new_details)
            errors = _quality_errors(prompt, duration)
            format_error = _format_error(prompt)
            if format_error:
                print("[H3 中文提示词] 格式检查提示：" + format_error)
            if errors:
                print("[H3 中文提示词] 质量检查提示：" + "；".join(errors))
            return (prompt,)
        finally:
            if not online_source and unload_after_generation:
                _VisionRuntime.close()


NODE_CLASS_MAPPINGS = {
    "H3Prompt": Qwen36MultiImageH3ChinesePrompt,
    "H3SaveImage": H3SaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Prompt": "H3 Prompt",
    "H3SaveImage": "H3 Save Image",
}
