import base64
import gc
import io
import re
import struct
import time
import json
import urllib.request
import os
import hashlib
import random
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

# 正文模式（T2VA / I2VA / FL2VA / L2VA）用官方三段式；全参考模式（Ref2VA）才用上面的六段式。
# 以前不分模式，正文请求也被写成 Ref2VA 六段结构，等于让模型按错的规范写提示词。
BASE_SECTION_NAMES = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

BASE_GENERATION_TYPES = frozenset({
    "文生视频", "图生视频", "首尾帧生成", "尾帧生成",
    "Text-to-Video", "Image-to-Video", "First/Last Frame", "Last Frame",
})


def _is_base_generation_type(generation_type):
    return str(generation_type or "").strip() in BASE_GENERATION_TYPES


def _sections_for(generation_type):
    return BASE_SECTION_NAMES if _is_base_generation_type(generation_type) else SECTION_NAMES


def _strip_type_heading(text, generation_type):
    """删掉模型偶尔写在前面的生成类型标题行（如单独一行 “Text-to-Video”）。"""
    lines = text.splitlines()
    while lines:
        head = lines[0].strip().strip("*_# ")
        if head and (head == str(generation_type).strip() or head in BASE_GENERATION_TYPES or head in {"Multi-Reference", "自动判别", "Auto Detect"}):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


#: 官方镜头时间戳写法是 [Shot N] At 00:05.000,（MM:SS.mmm，逗号结尾）。
#: 模型经常写成 00:1.5 / 1.5s / 5s，这里统一折算，不再指望模型自觉。
_SHOT_TIME = re.compile(r"\[(Shot\s+\d+)\]\s*(?:At\s*)?(\d{1,2}:\d{1,2}(?:\.\d{1,3})?|\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*([,，]?)", re.IGNORECASE)
_SHOT_PLAIN = re.compile(r"\[(Shot\s+\d+)\]\s*(?!At\b)(?=[^\]\n]{0,40}?[，,])", re.IGNORECASE)


def _stamp(seconds):
    total_ms = int(round(float(seconds) * 1000))
    minutes, remainder = divmod(total_ms, 60000)
    secs, ms = divmod(remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def _normalize_shot_times(text):
    def replace(match):
        shot, value = match.group(1), match.group(2)
        if ":" in value:
            minutes, secs = value.split(":", 1)
            seconds = int(minutes) * 60 + float(secs)
        else:
            seconds = float(value)
        if shot.lower().replace(" ", "") == "shot1":
            # 官方规范里第一个镜头不写时间戳
            return f"[{shot}]"
        return f"[{shot}] At {_stamp(seconds)},"

    return _SHOT_TIME.sub(replace, text)


def _body_section(sections):
    """正文主体字段名：正文模式是 integrated_multimodal_description，全参考是 detailed_description。"""
    return "integrated_multimodal_description" if tuple(sections) == BASE_SECTION_NAMES else "detailed_description"


def _strip_t2va_alignment(text):
    """文生视频没有首帧/尾帧，模型偶尔会照抄 I2VA 的首行对齐指令，这里删掉。

    只处理第一个字段之前的那段，且必须以 <Picture N> 开头才对；其他情况的正文一律不动。
    """
    first_section = re.search(_section_pattern("integrated_multimodal_description"), text)
    if not first_section:
        return text
    head, rest = text[:first_section.start()], text[first_section.start():]
    if "<Picture" not in head:
        return text
    kept = [
        line for line in head.splitlines()
        if line.strip() and "<Picture" not in line and "fully referenced" not in line.lower()
    ]
    return ("\n".join(kept) + "\n\n" + rest).strip() if kept else rest.strip()


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
   <d>[Chinese]对白原文</d>。用户输入引号内的对白必须逐字、完整复制，禁止省略号、概括、改写或截断；
   若篇幅不足，删减镜头细节而不是删减对白。用户没有要求对白时不要擅自添加。
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


BASE_SYSTEM_PROMPT = """你是 MiniMax H3 视频提示词编写专家。依据已经完成的画面分析和用户描述，
输出可以直接用于 MiniMax H3 的完整提示词。以下规则来自官方 MiniMax H3 h3-prompt-writing Skill
的正文模式（T2VA / I2VA / FL2VA / L2VA）规范，不得压缩为故事梗概。

只输出提示词正文，不要解释、分析过程、Markdown 标题或代码块。必须严格按以下字段及顺序输出：
integrated_multimodal_description:
overall_soundscape:
non_diegetic_music:

不要输出生成类型名称充当标题（例如单独一行 “Text-to-Video”）；除首行对齐指令外，
integrated_multimodal_description 之前不允许有任何行。

除上述官方字段名和下面规定的格式标记外，描述语言由节点的“输出中文提示词”开关决定。

规则：
1. 首行按生成类型写对齐指令，首行之后空一行再写字段。图生视频（I2VA）首行固定为
   `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`；
   首尾帧（FL2VA）首行说明 <Picture 1> 对齐 0.00 秒、<Picture 2> 对齐本段结束时刻，并给出两帧之间的连续路径；
   尾帧（L2VA）首行说明 <Picture 1> 对齐本段结束时刻。
   纯文生视频（T2VA）**不要写任何首行对齐指令**，也不要以“文生视频”“Text-to-Video”这类类型名开头，
   直接从 integrated_multimodal_description 字段写起。
2. integrated_multimodal_description 是正文主体：先用一两句定下整体视觉风格与画幅，再按播放顺序写镜头。
   [Shot 1] 不写时间戳；后续每一个切镜都必须写成 `[Shot N] At 00:05.000,` 这种格式（MM:SS.mmm，
   两位分钟、两位秒、三位毫秒，逗号结尾），时间严格递增且小于总时长。
   禁止写成 `[Shot 2] At 1.5s - 4.5s,` 这类区间或 `[Shot 2] At 5s,` 这类简写。
3. 每个镜头交代构图、主体外貌与位置、场景与关键道具、动作与反应、景别、机位、运镜和同步的画内声音；
   运镜幅度只用小幅、中幅、大幅，速度只用慢速、中速、快速。
4. 动作量必须适配总时长；短视频可以只有一个镜头。不要为凑长度重复描述、堆砌近义词或强行加镜头。
5. 有真实对白或歌唱时，按首次发声顺序分配稳定的 (S1)、(S2) 编号，并使用
   <d>[Chinese]对白原文</d>（语言标签按实际语种）。用户输入引号或 <d> 内的对白必须逐字完整复制，
   禁止省略号、概括、改写或截断；用户没有要求对白时不要擅自添加。
6. 画面中可见的文字（招牌、字幕、屏幕文字）用引号原样写出并保持原语言。
7. overall_soundscape 只总结持续环境声与关键画内音效，不重复对白；non_diegetic_music
   描述观众才能听见的配乐（乐器、速度、动态），不需要配乐时写 N/A。
8. 参考图只按它在本类型里的用途使用（身份/风格参考、首帧、尾帧或关键帧），不要凭空把它解释成别的角色或场景。
9. 不虚构素材无法支持的身份、品牌或外貌；用户描述含糊时补全为连贯、可拍摄、符合指定时长与画幅的方案。
10. 最后自查：三个字段齐全且有内容；镜头时间严格递增且小于总时长；每次切镜都有明确分工；
    声音与配乐没有写错字段。
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
# 节点面板和外部调用方用的是英文标签（INPUT_TYPES 里列的就是这几个），
# 以前只认中文键，于是按英文标签传进来时匹配不到技能文件、直接退回内置一行提示词。
OFFICIAL_SKILL_PATHS.update({
    "General H3 Prompt": "h3-prompt-writing",
    "3D Animated Short": "3d-animation-short-generator",
    "Brand Promo": "brand-promo-video-generator",
    "Co-op Game Intro": "co-op-game-intro-generator",
    "Hand-drawn Live Action": "handdrawn-live-video-generator",
    "Minimalist Product Ad": "minimalist-product-ad-generator",
    "Music Subtitle Video": "music-video-subtitle-generator",
    "Paper Collage Explainer": "paper-collage-explainer-generator",
    "Papercraft Stop-motion Explainer": "papercraft-stop-motion-explainer",
})


# skills/<技能>/references/ 下的官方规范原文：正文模式看 base-en，全参考模式看 ref-en。
# SKILL.md 里写着「read references/base-en.txt and follow its final prompt structure」，
# 但模型没有读文件的能力 —— 不把原文喂进去，它只能自己编结构（正文请求被写成 Ref2VA
# 六段式就是这么来的）。这里按模式把对应那份原文一起送进去。
_SKILL_GUIDES = {"base": "base-en.txt", "reference": "ref-en.txt"}

#: 注入上限（字符）。节点上下文默认 12288 token，技能文件 + 规范原文 + 分析 + 输出
#: 必须一起放得下，所以两个都限长（超出部分截断，并附一句说明）。
_SKILL_GUIDE_LIMIT = 8000
_SKILL_MD_LIMIT = 8000
_SKILL_REFERENCE_MENTION = re.compile(r"references/([A-Za-z0-9._\-]+)")


def _clip(text, limit, note):
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n（{note}）"


def _skill_reference_text(folder, skill_text, generation_type):
    """把 SKILL.md 点名要求读的 references 原文一并送给模型。

    h3-prompt-writing 按模式挑一份（正文 base-en / 全参考 ref-en）；其它技能按 SKILL.md
    里提到的顺序把 references 文件依次带上，总额受 _SKILL_GUIDE_LIMIT 限制。
    """
    root = Path(__file__).parent / "skills" / folder / "references"
    if folder == "h3-prompt-writing":
        names = [_SKILL_GUIDES["base" if _is_base_generation_type(generation_type) else "reference"]]
        limit = _SKILL_GUIDE_LIMIT
    else:
        names = []
        limit = 6000
        for name in _SKILL_REFERENCE_MENTION.findall(skill_text or ""):
            if name not in names:
                names.append(name)
    parts = []
    used = 0
    for name in names:
        try:
            guide = (root / name).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not guide:
            continue
        remaining = limit - used
        if remaining <= 400:
            break
        if len(guide) > remaining:
            guide = _clip(guide, remaining, f"references/{name} 过长，其余部分已省略")
        header = f"以下是官方规范原文 references/{name}，最终提示词必须严格按它的字段名、字段顺序和格式输出：\n\n"
        parts.append(header + guide)
        used += len(header) + len(guide)
    return "\n\n".join(parts)


def _official_skill_instruction(skill_name, generation_type=None):
    """Read the vendored MiniMax-H3 skill snapshot; never fetch at runtime."""
    folder = OFFICIAL_SKILL_PATHS.get(skill_name)
    if not folder:
        return CREATIVE_SKILL_INSTRUCTIONS.get(skill_name, CREATIVE_SKILL_INSTRUCTIONS["自动判别"])
    skill_file = Path(__file__).parent / "skills" / folder / "SKILL.md"
    try:
        text = skill_file.read_text(encoding="utf-8").strip()
        if text:
            # 扫描 references 提及要用完整原文（截断后可能正好把提及部分切掉）
            reference = _skill_reference_text(folder, text, generation_type)
            clipped = _clip(text, _SKILL_MD_LIMIT, "技能文件较长，其余部分已省略")
            instruction = "以下是本地安装的 MiniMax-H3 官方技能文件内容，请严格按其要求执行：\n\n" + clipped
            if reference:
                instruction += "\n\n" + reference
            return instruction
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


def _gguf_read_string(handle):
    size = struct.unpack("<Q", handle.read(8))[0]
    return handle.read(size).decode("utf-8", "replace")


def _gguf_read_value(handle, kind):
    fixed = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
    }
    if kind in fixed:
        fmt = fixed[kind]
        return struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]
    if kind == 8:
        return _gguf_read_string(handle)
    if kind == 9:
        element = struct.unpack("<I", handle.read(4))[0]
        for _ in range(struct.unpack("<Q", handle.read(8))[0]):
            _gguf_read_value(handle, element)
        return None
    raise ValueError(f"unknown GGUF metadata type {kind}")


def _gguf_architecture(model_path):
    """Read `general.architecture` from a GGUF header without loading weights.

    The chat handler has to be chosen before the model is loaded, and file names
    are unreliable (a Qwen3.8 file still reports architecture `qwen35`).
    """
    if not model_path:
        return ""
    try:
        with open(model_path, "rb") as handle:
            if handle.read(4) != b"GGUF":
                return ""
            struct.unpack("<I", handle.read(4))[0]
            struct.unpack("<Q", handle.read(8))[0]
            for _ in range(struct.unpack("<Q", handle.read(8))[0]):
                key = _gguf_read_string(handle)
                value = _gguf_read_value(handle, struct.unpack("<I", handle.read(4))[0])
                if key == "general.architecture":
                    return str(value or "").lower()
    except Exception:
        return ""
    return ""


def _create_chat_handler(model_name, mmproj_path, enable_thinking=False):
    if _LLAMA_CPP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "本地 GGUF 模式需要安装 llama-cpp-python>=0.3.46；"
            "在线 OpenAI 兼容 API 模式无需此依赖。"
        ) from _LLAMA_CPP_IMPORT_ERROR
    name = Path(model_name).name.lower()
    family = _gguf_architecture(_resolve_llm_path(model_name)) or name
    common = {
        "clip_model_path": str(mmproj_path),
        "image_min_tokens": 0,
        "image_max_tokens": 0,
        "verbose": False,
    }
    if "qwen35" in family or "qwen3_5" in family or any(
        version in name for version in ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8", "qwen3.9")
    ):
        return Qwen35ChatHandler(enable_thinking=enable_thinking, **common)
    if "qwen3vl" in family or "qwen3-vl" in name or "qwen3_vl" in name:
        return Qwen3VLChatHandler(force_reasoning=enable_thinking, **common)
    if "gemma-4" in name or "gemma4" in name:
        return Gemma4ChatHandler(enable_thinking=enable_thinking, **common)
    if "gemma-3" in name or "gemma3" in name:
        return Gemma3ChatHandler(**common)
    return MTMDChatHandler(use_gpu=True, **common)


# Context window sizes offered by the nodes. A larger window needs more VRAM for
# the KV cache, so a smaller one lets a bigger model stay on the GPU.
CONTEXT_LENGTH_OPTIONS = ("4096", "6144", "8192", "12288", "16384", "24576", "32768", "49152", "65536")
DEFAULT_CONTEXT_LENGTH = 12288


class _VisionRuntime:
    llm = None
    chat_handler = None
    # Which think setting the resident handler is built with. The prompt the
    # handler renders decides whether the model plans or answers, and one run can
    # need both (plan first, then write), so the handler is swappable.
    handler_thinking = None
    # What the resident model was loaded with. Without this, a second call with
    # the same settings pays the whole load again (~2-4 s from the OS cache,
    # much more on a cold read) for a model that is already in VRAM.
    signature = None

    @classmethod
    def load(
        cls,
        model_relative_path,
        vision_relative_path,
        gpu_layers,
        context_length=DEFAULT_CONTEXT_LENGTH,
        enable_thinking=False,
    ):
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
            n_ctx = int(context_length)
        except (TypeError, ValueError):
            n_ctx = DEFAULT_CONTEXT_LENGTH
        n_ctx = max(512, n_ctx)
        print(f"[H3 中文提示词] 上下文长度：{n_ctx}")
        print(f"[H3 中文提示词] 思考模式：{'开启' if enable_thinking else '关闭'}")
        try:
            cls.chat_handler = _create_chat_handler(model_relative_path, mmproj_path, enable_thinking)
            cls.llm = Llama(
                model_path=str(model_path),
                chat_handler=cls.chat_handler,
                n_gpu_layers=gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )
        except Exception:
            cls.close()
            raise
        cls.handler_thinking = bool(enable_thinking)
        cls.signature = cls._signature(
            model_relative_path, vision_relative_path, gpu_layers, n_ctx, enable_thinking
        )
        return cls.llm

    @classmethod
    def swap_thinking(cls, enable_thinking):
        """Rebuild the chat handler only, keeping the weights where they are.

        The rendered prompt tells the model whether to plan or to answer: with
        the think block left open it plans, with it closed it writes. A budgeted
        run needs the first for the plan and the second for the answer, and
        without this the second pass re-plans (measured 6169 tokens after an
        800-token cut). Only the handler is rebuilt, so no weights reload.
        """
        if cls.llm is None or cls.signature is None:
            return False
        if cls.handler_thinking is not None and bool(cls.handler_thinking) == bool(enable_thinking):
            return False
        vision_path = _resolve_llm_path(cls.signature[1])
        if vision_path is None:
            return False
        handler = _create_chat_handler(cls.signature[0], vision_path, bool(enable_thinking))
        previous = cls.chat_handler
        cls.chat_handler = handler
        cls.handler_thinking = bool(enable_thinking)
        cls.llm.chat_handler = handler
        if previous is not None:
            try:
                previous._exit_stack.close()
            except Exception:
                pass
        return True

    @staticmethod
    def _signature(model_relative_path, vision_relative_path, gpu_layers, n_ctx, enable_thinking):
        return (
            str(model_relative_path),
            str(vision_relative_path),
            int(gpu_layers),
            int(n_ctx),
            bool(enable_thinking),
        )

    @classmethod
    def ensure(
        cls,
        model_relative_path,
        vision_relative_path,
        gpu_layers,
        context_length=DEFAULT_CONTEXT_LENGTH,
        enable_thinking=False,
    ):
        """Load only when the resident model cannot serve this request.

        Every queue re-executes the node from scratch, so the default behaviour
        of reloading is a fixed tax on each run -- and a multi-prompt batch asked
        for the same weights N times over. Reusing the resident runtime removes
        that tax without changing what is loaded.
        """
        try:
            n_ctx = max(512, int(context_length))
        except (TypeError, ValueError):
            n_ctx = DEFAULT_CONTEXT_LENGTH
        signature = cls._signature(
            model_relative_path, vision_relative_path, gpu_layers, n_ctx, enable_thinking
        )
        if cls.llm is not None and cls.signature == signature:
            cls.swap_thinking(enable_thinking)
            return cls.llm
        return cls.load(
            model_relative_path,
            vision_relative_path,
            gpu_layers,
            n_ctx,
            enable_thinking,
        )

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
        cls.handler_thinking = None
        cls.signature = None
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
            "temperature", "top_p", "top_k", "min_p", "max_tokens",
            "frequency_penalty", "presence_penalty", "seed",
            # vLLM/SGLang extension used by the Qwen-Image-2.1 prompt enhancers
            # to keep the thinking block on. Other nodes never pass it.
            "chat_template_kwargs",
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
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # A reasoning block that was cut off at the token limit has no closing tag.
    # Drop it so private reasoning never reaches the prompt output.
    text = re.sub(r"<think\b.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _section_pattern(section):
    # Accept Markdown emphasis, headings, Chinese colons, and content on the heading line.
    return rf"(?im:^\s*(?:[#>*-]+\s*)?(?:\*\*|__)?{re.escape(section)}(?:\*\*|__)?\s*[:\uFF1A](?:\*\*|__)?)"


def _normalize_sections(text, sections=SECTION_NAMES):
    normalized = text
    for section in sections:
        normalized = re.sub(_section_pattern(section), f"{section}:", normalized, count=1)

    present = [section for section in sections if re.search(_section_pattern(section), text)]
    if present and len(present) < len(sections):
        for section in sections:
            if section not in present:
                normalized = normalized.rstrip() + f"\n\n{section}:\nN/A"
    return normalized.strip()


def _h3_section_count(text, sections=SECTION_NAMES):
    return sum(bool(re.search(_section_pattern(section), text)) for section in sections)


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


def _context_length(inputs, default=DEFAULT_CONTEXT_LENGTH):
    raw = _input_value(inputs, "Context Length", "上下文长度", default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 512 else default


def _completion_budget(context_length, requested):
    """Keep one reply inside the configured window.

    Half of the window is reserved for the instructions, the reference media and
    the material already produced, so a small context length trims the reply
    budget instead of letting a response run out of room halfway through.
    """
    limit = max(512, int(context_length) // 2)
    return max(256, min(int(requested), limit))


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


def _format_error(text, sections=SECTION_NAMES):
    positions = []
    for section in sections:
        match = re.search(_section_pattern(section), text)
        if match is None:
            return f"缺少字段 {section}"
        positions.append(match.start())
    if positions != sorted(positions):
        return "字段顺序不正确"
    return None


def _section_text(text, section, sections=SECTION_NAMES):
    index = sections.index(section)
    next_section = sections[index + 1] if index + 1 < len(sections) else None
    pattern = r"(?s:" + _section_pattern(section) + r"\s*(.*?))"
    pattern += rf"(?={_section_pattern(next_section)}|\Z)" if next_section else r"\Z"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _replace_section(text, section, replacement, sections=SECTION_NAMES):
    index = sections.index(section)
    next_section = sections[index + 1] if index + 1 < len(sections) else None
    pattern = r"(?s:" + _section_pattern(section) + r"\s*.*?)"
    pattern += rf"(?={_section_pattern(next_section)}|\Z)" if next_section else r"\Z"
    return re.sub(pattern, f"{section}:\n{replacement.strip()}\n\n", text, count=1).strip()


def _detail_is_short(text, duration, sections=SECTION_NAMES):
    details = _section_text(text, _body_section(sections), sections)
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", details))
    target = max(180, min(700, round(float(duration) * 45)))
    shots = len(re.findall(r"\[Shot\s+\d+\]", details, flags=re.IGNORECASE))
    minimum_shots = 1 if duration <= 6 else 2 if duration <= 15 else 3
    return chinese_count < target or shots < minimum_shots


def _quality_errors(text, duration, sections=SECTION_NAMES):
    errors = []
    format_error = _format_error(text, sections)
    if format_error:
        return [format_error]

    # 正文模式只有三个字段，没有 summary / retention_analysis / subject_definitions，
    # 那些检查只在全参考模式下做。
    reference_mode = tuple(sections) == SECTION_NAMES
    body_section = _body_section(sections)
    summary = _section_text(text, "summary", sections) if reference_mode else ""
    retention = _section_text(text, "retention_analysis", sections) if reference_mode else ""
    details = _section_text(text, body_section, sections)
    soundscape = _section_text(text, "overall_soundscape", sections)

    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", details))
    minimum_chinese = max(180, min(700, round(float(duration) * 45)))
    if chinese_count < minimum_chinese:
        errors.append(f"{body_section} 只有约 {chinese_count} 个汉字，至少需要 {minimum_chinese} 个")

    shot_count = len(re.findall(r"\[Shot\s+\d+\]", details, flags=re.IGNORECASE))
    minimum_shots = 1 if duration <= 6 else 2 if duration <= 15 else 3
    if shot_count < minimum_shots:
        errors.append(f"只有 {shot_count} 个镜头，当前时长至少需要 {minimum_shots} 个有明确分工的镜头")

    if reference_mode:
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

        subjects = set(re.findall(r"<Subject\s+\d+>", _section_text(text, "subject_definitions", sections)))
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
        errors.append(f"{body_section} 存在重复长句：" + "、".join(repeated_clauses[:3]))
    return errors


SCALE_BY_MULTIPLIER = "scale by multiplier"
TARGET_DIMENSIONS = "target dimensions"

# nvvfx ships more levels than the NVIDIA node exposes; DENOISE_* and DEBLUR_*
# are for footage, the plain ones are the ones that suit a still.
RTX_QUALITY_LEVELS = (
    "BICUBIC",
    "LOW",
    "MEDIUM",
    "HIGH",
    "ULTRA",
    "DENOISE_LOW",
    "DENOISE_MEDIUM",
    "DENOISE_HIGH",
    "DENOISE_ULTRA",
    "DEBLUR_LOW",
    "DEBLUR_MEDIUM",
    "DEBLUR_HIGH",
    "DEBLUR_ULTRA",
)


def _import_nvvfx():
    """NVIDIA's VFX runtime, which is what actually does the upscaling."""
    try:
        import nvvfx
    except ImportError as exc:
        raise RuntimeError(
            "RTX 超分需要 NVIDIA 的 VFX 运行时：pip install nvidia-vfx"
            "（装完重启 ComfyUI）。"
        ) from exc
    return nvvfx


class RTXImageSuperResolution:
    """RTX super resolution for stills, with or without an alpha channel.

    The NVIDIA node feeds three-channel frames to the engine, so a Qwen-Image
    result (RGBA) cannot go through it -- the alpha has nowhere to live. This one
    splits the alpha off, upscales the visible three channels with the same nvvfx
    engine, enlarges the alpha to match and puts the four channels back together,
    so a cut-out poster keeps its transparency.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "要放大的图像，RGB 或 RGBA 都行。"}),
                "resize_type": (
                    [SCALE_BY_MULTIPLIER, TARGET_DIMENSIONS],
                    {
                        "default": SCALE_BY_MULTIPLIER,
                        "tooltip": "按倍数放大，或者指定目标宽高。",
                    },
                ),
                "scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.01},
                ),
                "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8}),
                "quality": (
                    list(RTX_QUALITY_LEVELS),
                    {
                        "default": "ULTRA",
                        "tooltip": "ULTRA 最清晰也最慢；DENOISE_*/DEBLUR_* 是给视频帧用的档位。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale"
    CATEGORY = "Prompt Enhancer"
    DESCRIPTION = "RTX super resolution that keeps an alpha channel when the input has one."

    def upscale(self, images, resize_type, scale, width, height, quality):
        import comfy.utils
        import torch

        nvvfx = _import_nvvfx()

        batch, in_h, in_w, channels = images.shape
        if resize_type == TARGET_DIMENSIONS:
            out_w, out_h = int(width), int(height)
        else:
            out_w, out_h = int(in_w * float(scale)), int(in_h * float(scale))
        out_w = max(8, round(out_w / 8) * 8)
        out_h = max(8, round(out_h / 8) * 8)

        has_alpha = channels > 3
        rgb = images[..., :3]
        alpha = images[..., 3:4] if has_alpha else None

        level = getattr(nvvfx.effects.QualityLevel, quality, None)
        if level is None:
            level = nvvfx.effects.QualityLevel.ULTRA

        result = torch.empty((batch, out_h, out_w, 3), dtype=torch.float32)
        with nvvfx.VideoSuperRes(level) as engine:
            engine.output_width = out_w
            engine.output_height = out_h
            engine.load()
            for index in range(batch):
                frame = rgb[index].permute(2, 0, 1).float().cuda().contiguous()
                dlpack_image = engine.run(frame).image
                frame_out = torch.from_dlpack(dlpack_image)
                # The engine hands back CHW; keep the visible three channels in
                # case a future build appends something of its own.
                result[index] = frame_out.movedim(0, -1)[:, :, :3].to(result.device)

        if alpha is not None:
            # The engine has no alpha input, so the channel is enlarged with a
            # plain resample and reattached. Not comfy.utils.common_upscale: its
            # lanczos path assumes three channels and collapses a single one.
            enlarged = torch.nn.functional.interpolate(
                alpha.movedim(-1, 1).float(),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
            result = torch.cat([result, enlarged.to(result.device)], dim=-1)

        print(
            f"[Prompt Enhancer] RTX 超分：{in_w}x{in_h} → {out_w}x{out_h}"
            f"（{quality}，{'RGBA' if has_alpha else 'RGB'}）"
        )
        return (result.to(images.dtype),)


def _input_image_files():
    """Every image file under the input directory, subfolders included.

    The stock Load Image node lists the top level only, so images pasted into
    something like input/pasted/ never show up in its dropdown. Doing that by
    rewriting the stock node's INPUT_TYPES from here is the kind of
    cross-node interference the registry's standards forbid, so this pack offers
    the same convenience as its own node instead.
    """
    root = folder_paths.get_input_directory()
    files = []
    for current, _, names in os.walk(root):
        for name in names:
            full = os.path.join(current, name)
            if os.path.isfile(full):
                files.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


class PromptEnhancerLoadImage:
    """Load Image, with a dropdown that also covers input subfolders.

    Reading is delegated to the stock node's own loader, so EXIF rotation, alpha
    handling, animated formats and the upload button all behave exactly as they
    do in Load Image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    _input_image_files(),
                    {
                        "image_upload": True,
                        "tooltip": "input 目录下的图片（含子目录，例如 pasted/ 里的粘贴图）。",
                    },
                )
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load"
    CATEGORY = "Prompt Enhancer"
    DESCRIPTION = "Load an image from the input directory, including its subfolders."

    def load(self, image):
        import nodes as comfy_nodes

        return comfy_nodes.LoadImage().load_image(image)

    @classmethod
    def IS_CHANGED(cls, image):
        path = folder_paths.get_annotated_filepath(image)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read())
        return digest.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


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
    CATEGORY = "Prompt Enhancer/Output"

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
        # Continue the numeric sequence for this prefix. Include sidecar JSON
        # files in the scan so image/workflow pairs always share one number.
        prefix_pattern = re.compile(r"^" + re.escape(prefix) + r"_(\d+)(?:_\d+)?(?:\.[^.]+)$", re.IGNORECASE)
        next_number = 1
        for existing in os.listdir(output_dir):
            match = prefix_pattern.match(existing)
            if match:
                next_number = max(next_number, int(match.group(1)) + 1)
        for index, tensor in enumerate(images, start=1):
            array = np.clip(tensor.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(array, "RGB")
            stem = f"{prefix}_{next_number:05d}"
            path = os.path.join(output_dir, stem + extension)
            # A concurrent writer may claim the number between the directory
            # scan and save; advance the number instead of adding a suffix.
            while os.path.exists(path) or (save_json and os.path.exists(os.path.splitext(path)[0] + ".json")):
                next_number += 1
                stem = f"{prefix}_{next_number:05d}"
                path = os.path.join(output_dir, stem + extension)
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
            next_number += 1
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
                        "tooltip": "-1=全放显存，0=全用内存。显存不够就调小。",
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
                        "tooltip": "生成后控制：随机/递增/固定。",
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
                "Context Length": (
                    list(CONTEXT_LENGTH_OPTIONS),
                    {
                        "default": str(DEFAULT_CONTEXT_LENGTH),
                        "tooltip": "上下文窗口，也决定 KV 显存占用。默认 12288，调大更占显存。",
                    },
                ),
                "Enable Thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Thinking",
                        "label_off": "Direct",
                        "tooltip": "开启=先推理再输出（更慢、更细），推理内容不会写进提示词。",
                    },
                ),
            },
            "optional": {
                **{f"Image {index}": ("IMAGE",) for index in range(1, 10)},
                **{f"Video {index}": ("VIDEO",) for index in range(1, 4)},
                **{f"Audio {index}": ("AUDIO",) for index in range(1, 4)},
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("H3 Prompt",)
    FUNCTION = "生成"
    CATEGORY = "Prompt Enhancer"
    DESCRIPTION = "Generate MiniMax H3 prompts with local GGUF or online vision-language models."

    def 生成(self, **inputs):
        online_source = _is_online_source(_input_value(inputs, "Model Source", "模型来源", False))
        model_name = _input_value(inputs, "Language Model", "语言模型")
        vision_name = _input_value(inputs, "Vision Model", "视觉模型")
        gpu_layers = _input_value(inputs, "GPU Offload Layers", "GPU卸载层数", -1)
        context_length = _context_length(inputs)
        enable_thinking = bool(_input_value(inputs, "Enable Thinking", "启用思考", False))
        seed = _input_value(inputs, "Seed", "种子", 0)
        generation_type = _input_value(inputs, "Generation Type", "生成类型", "Auto Detect")
        if not online_source and (model_name == "未找到语言模型" or vision_name == "未找到视觉模型"):
            raise FileNotFoundError("请将 GGUF 语言模型和对应 mmproj 视觉模型放入 models/LLM。")
        images = _collect_images(inputs, allow_empty=generation_type in {"自动判别", "文生视频", "Auto Detect", "Text-to-Video"})
        videos = [inputs.get(f"Video {index}") for index in range(1, 4) if inputs.get(f"Video {index}") is not None]
        audios = [inputs.get(f"Audio {index}") for index in range(1, 4) if inputs.get(f"Audio {index}") is not None]
        media_tags = ""
        if videos:
            media_tags += "\n已连接视频输入：" + ", ".join(f"<Video {index}>" for index in range(1, len(videos) + 1)) + "。必须使用这些标签，不得写成 Picture。"
        if audios:
            media_tags += "\n已连接音频输入：" + ", ".join(f"<Audio {index}>" for index in range(1, len(audios) + 1)) + "。必须使用这些标签，不得写成 Picture N/A。音频用于用户指定的对白、语音、歌词和节奏。"
        # With no reference image, automatic mode is a true text-to-video task.
        if not images and generation_type in {"自动判别", "Auto Detect"}:
            generation_type = "Text-to-Video"
        # 正文模式和全参考模式的字段结构不同：正文用官方三段式，全参考才用六段式。
        sections = _sections_for(generation_type)
        system_prompt = BASE_SYSTEM_PROMPT if tuple(sections) == BASE_SECTION_NAMES else SYSTEM_PROMPT
        body_section = _body_section(sections)
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
        creative_instruction = _official_skill_instruction(creative_skill, generation_type)
        unload_after_generation = bool(_input_value(inputs, "Unload Model After Generation", "生成后卸载模型", True))
        output_chinese = bool(_input_value(inputs, "Output Chinese Prompt", "输出中文提示词", False))
        language_instruction = (
            "输出必须使用简体中文（官方字段名和格式标记保持不变）。"
            if output_chinese
            else "输出必须使用英文（官方字段名和格式标记保持不变）；不要翻译字段名。"
        )
        if output_chinese:
            # 官方规范原文里写着「用英文书写」，它会盖过系统提示的语言要求（规范在用户消息里、
            # 位置更靠后）。开着中文输出时必须把这条追在规范后面显式覆盖掉。
            creative_instruction += (
                "\n\n注意：本节点已开启“输出中文提示词”，上面规范中“用英文书写”的要求由本条覆盖——"
                "字段名与格式标记（integrated_multimodal_description、[Shot N]、<d>、<Picture N> 等）保持英文，"
                "正文描述一律用简体中文；仅对白、歌词和画面内文字保留其原本语言。"
            )
        description = _input_value(inputs, "Description", "简单描述", "").strip() or (
            "根据文字描述创作连贯、自然、有电影感的视频。" if not images
            else "根据参考图片创作连贯、自然、有电影感的视频。"
        )

        analysis_request = (
            f"目标视频总时长：{duration:g} 秒\n目标画面比例：{aspect_ratio}\n"
            f"参考图片数量：{len(images)}\n用户的简单描述：{description}\n"
            f"指定生成类型：{generation_type}\n生成类型要求：{generation_instruction}{media_tags}"
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
        if videos or audios:
            content.append({"type": "text", "text": "媒体引用规则：严格按连接顺序使用 <Picture N>、<Video N>、<Audio N>。不要把音频或视频写成 <Picture N/A>。用户提到的音频/视频必须在 subject_definitions、summary 或镜头描述中明确引用。"})

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
            llm = _VisionRuntime.load(
                model_name, vision_name, gpu_layers, context_length, enable_thinking
            )
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
                max_tokens=_completion_budget(context_length, 1024),
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
                + (
                    "现在严格按照上面给出的官方正文模式规范写出最终提示词，"
                    "只输出 integrated_multimodal_description、overall_soundscape、non_diegetic_music 三个字段。"
                    if tuple(sections) == BASE_SECTION_NAMES
                    else "现在严格按照六段 Ref2VA 格式写出最终提示词。"
                ) + language_instruction
            )
            messages = [
                {"role": "system", "content": system_prompt + "\n\n当前语言要求：" + language_instruction},
                {"role": "user", "content": final_request},
            ]
            prompt = _stream_completion(
                llm,
                messages=messages,
                stage="最终提示词",
                seed=(seed + 1) & 0xFFFFFFFFFFFFFFFF,
                max_tokens=_completion_budget(context_length, 3072),
                temperature=0.45,
                top_p=0.9,
                repeat_penalty=1.12,
                frequency_penalty=0.12,
            )
            section_count = _h3_section_count(prompt, sections)
            if section_count < 2 or _looks_like_internal_text(prompt):
                raise RuntimeError(
                    "所选语言模型没有遵循 H3 写作指令，返回了内部说明或无关文本。"
                    "这通常是 Uncensored 微调模型的指令遵循问题，请更换 Instruct 模型或更换种子。"
                )
            prompt = _normalize_sections(prompt, sections)
            prompt = _strip_type_heading(prompt, generation_type)
            prompt = _normalize_shot_times(prompt)
            if _detail_is_short(prompt, duration, sections) and len(_section_text(prompt, body_section, sections)) < 120:
                if tuple(sections) == BASE_SECTION_NAMES:
                    expansion_request = (
                        "只重写 integrated_multimodal_description 这一个字段。"
                        "不要输出其他字段、解释、Markdown 或内部分析。\n\n"
                        f"写到约 {max(180, min(700, round(duration * 45)))} 个中文汉字即可，"
                        f"为 {duration:g} 秒视频安排至少 {1 if duration <= 6 else 2 if duration <= 15 else 3} 个有明确分工的镜头。"
                        "按 [Shot 1] / [Shot N] At MM:SS.mmm, 的时间轴写，交代构图、主体动作、环境、运镜和画内声音，"
                        "不要为了长度重复、堆砌近义词或过度复杂化。\n\n"
                        f"目标画幅：{aspect_ratio}\n用户描述：{description}\n\n"
                        f"视觉资料：\n{visual_plan}\n\n"
                        f"当前过短正文：\n{_section_text(prompt, body_section, sections)}"
                    )
                else:
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
                        f"主体定义：\n{_section_text(prompt, 'subject_definitions', sections)}\n\n"
                        f"当前保留分析：\n{_section_text(prompt, 'retention_analysis', sections)}\n\n"
                        f"当前过短正文：\n{_section_text(prompt, 'detailed_description', sections)}"
                    )
                expansion = _stream_completion(
                    llm,
                    messages=[
                        # 补写这一步以前不带语言要求，中文开关会被它悄悄改成英文输出。
                        {"role": "system", "content": system_prompt + "\n\n当前语言要求：" + language_instruction},
                        {"role": "user", "content": expansion_request},
                    ],
                    stage="正文补写",
                    seed=(seed + 2) & 0xFFFFFFFFFFFFFFFF,
                    max_tokens=_completion_budget(context_length, 2048),
                    temperature=0.4,
                    top_p=0.9,
                    repeat_penalty=1.15,
                    frequency_penalty=0.15,
                )
                expansion = _normalize_sections(expansion, sections)
                if tuple(sections) == BASE_SECTION_NAMES:
                    new_body = _section_text(expansion, body_section, sections)
                    if new_body and new_body.upper() != "N/A":
                        prompt = _replace_section(prompt, body_section, new_body, sections)
                else:
                    new_retention = _section_text(expansion, "retention_analysis", sections)
                    new_details = _section_text(expansion, "detailed_description", sections)
                    if new_retention and new_retention.upper() != "N/A":
                        prompt = _replace_section(prompt, "retention_analysis", new_retention, sections)
                    if new_details and new_details.upper() != "N/A":
                        prompt = _replace_section(prompt, "detailed_description", new_details, sections)
            # 补写回来的正文同样要过一遍清洗：模型经常把 Shot 1 的时间戳写回来、
            # 或者又把时间写成 00:1.5 这种简写。
            prompt = _strip_type_heading(_normalize_shot_times(prompt), generation_type)
            if not images:
                prompt = _strip_t2va_alignment(prompt)
            errors = _quality_errors(prompt, duration, sections)
            format_error = _format_error(prompt, sections)
            if format_error:
                print("[H3 中文提示词] 格式检查提示：" + format_error)
            if errors:
                print("[H3 中文提示词] 质量检查提示：" + "；".join(errors))
            return (prompt,)
        finally:
            if not online_source and unload_after_generation:
                _VisionRuntime.close()


class H3ImagePromptGenerator:
    """Generate a list of creative image-generation prompts from text and up to two images."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _language_models() or ["No language models found"]
        vision_models = _vision_models() or ["No vision models found"]
        return {"required": {
            "Original Request": ("STRING", {"default": "", "multiline": True, "placeholder": "Describe what you want to create..."}),
            "Prompt Count": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
            "Seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF, "step": 1}),
            "Prompt Format": (["SDXL / Illustrious / NoobAI Tags", "Natural Language"], {"default": "Natural Language"}),
            "Aspect Ratio": (["Auto", "1:1 Square", "4:3 Landscape", "3:4 Portrait", "16:9 Widescreen", "9:16 Vertical", "2:3 Portrait", "21:9 Ultrawide"], {"default": "Auto"}),
            "Model Source": ("BOOLEAN", {"default": False, "label_on": "Online LLM", "label_off": "Local Model"}),
            "Language Model": (models,),
            "Vision Model": (vision_models,),
            "GPU Offload Layers": ("INT", {"default": -1, "min": -1, "max": 256, "step": 1}),
            "Online Request URL": ("STRING", {"default": "https://api.openai.com/v1", "multiline": False}),
            "Online API Key": ("STRING", {"default": "", "multiline": False, "password": True}),
            "Online Model": ("STRING", {"default": "", "multiline": False}),
            "Output Chinese": ("BOOLEAN", {"default": False}),
            "Context Length": (
                list(CONTEXT_LENGTH_OPTIONS),
                {
                    "default": str(DEFAULT_CONTEXT_LENGTH),
                    "tooltip": "上下文窗口，也决定 KV 显存占用。默认 12288，调大更占显存。",
                },
            ),
            "Enable Thinking": (
                "BOOLEAN",
                {
                    "default": False,
                    "label_on": "Thinking",
                    "label_off": "Direct",
                    "tooltip": "开启=先推理再输出（更慢、更细），推理内容不会写进提示词。",
                },
            ),
        }, "optional": {"Image 1": ("IMAGE",), "Image 2": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Image Prompts",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "Prompt Enhancer"

    @classmethod
    def IS_CHANGED(cls, **inputs):
        """Make the -1 seed mode bypass ComfyUI's execution cache."""
        try:
            seed = int(inputs.get("Seed", -1))
        except (TypeError, ValueError):
            seed = -1
        return time.time_ns() if seed == -1 else seed

    def generate(self, **inputs):
        online = _is_online_source(inputs.get("Model Source", False))
        request = inputs.get("Original Request", "").strip()
        if not request:
            raise ValueError("Original Request cannot be empty.")
        count = int(inputs.get("Prompt Count", 4))
        seed = int(inputs.get("Seed", -1))
        context_length = _context_length(inputs)
        enable_thinking = bool(_input_value(inputs, "Enable Thinking", "启用思考", False))
        actual_seed = random.SystemRandom().randint(0, 0xFFFFFFFF) if seed == -1 else seed
        prompt_format = inputs.get("Prompt Format", "Natural Language")
        tag_mode = prompt_format == "SDXL / Illustrious / NoobAI Tags"
        aspect_ratio = inputs.get("Aspect Ratio", "Auto")
        # Parenthesized user text is a literal constraint. Keep it byte-for-byte
        # (including the brackets) in every generated prompt.
        literal_segments = re.findall(r"\([^()\n]+\)|（[^（）\n]+）", request)
        images = [inputs[name] for name in ("Image 1", "Image 2") if inputs.get(name) is not None]
        framing_instruction = (
            "Choose a suitable aspect ratio from the concept and make composition adaptable to it."
            if aspect_ratio == "Auto" else
            f"Design every prompt for a {aspect_ratio} canvas. Treat this ratio as a composition constraint: plan framing, subject scale, visual balance, crop boundaries, and negative space so the main subject remains readable within the frame."
        )
        literal_instruction = (
            f"The following parenthesized text is mandatory literal content: {', '.join(literal_segments)}. Include each item exactly as written, including its brackets, in every prompt. Do not translate, paraphrase, or remove it. "
            if literal_segments else ""
        )
        content = [{"type": "text", "text": (
            f"Original user request:\n{request}\n\nGenerate exactly {count} distinct image-generation prompts. "
            "Improve the idea with tasteful creative direction using a structured subject-first workflow: identify the main subject and action, then composition, camera/viewpoint, environment, lighting, color palette, materials, mood, and rendering/style cues. "
            f"{framing_instruction} "
            "Each prompt must be self-contained and directly usable by an image model. "
            + literal_instruction
            + "Use only positive visual descriptions; never output negative prompts, negative tags, exclusions, or a separate negative-prompt field. "
            + "Do not add explanations, numbering, markdown fences, or commentary. Return one prompt per line."
        )}]
        if images:
            content[0]["text"] += "\nReference images are provided. Identify visible people, scene, objects, clothing, colors and style, and incorporate only supported traits into the prompts."
            for index, image in enumerate(images, 1):
                content.append({"type": "text", "text": f"Reference image {index}:"})
                content.append({"type": "image_url", "image_url": {"url": _tensor_to_data_url(image)}})
        language = "Simplified Chinese" if inputs.get("Output Chinese", False) else "English"
        if tag_mode:
            content[0]["text"] += (
                "\nFORMAT: SDXL / Illustrious / NoobAI-compatible positive tag prompts. "
                "Write in English only, as comma-separated weighted or unweighted tags. "
                "Use a practical tag order: quality/style, subject, appearance, clothing, pose/action, "
                "composition, environment, lighting, color, camera and medium/style. "
                "Do not write prose sentences, headings, numbering, or negative prompts."
            )
        else:
            content[0]["text"] += (
                f"\nFORMAT: Natural-language image prompts in {language}. "
                "Write fluent, vivid but reasonably concise sentences or a compact paragraph. "
                "Do not force tag syntax or add negative prompts."
            )
        if online:
            key = inputs.get("Online API Key", "").strip()
            if not key:
                raise ValueError("Online API Key is required in Online LLM mode.")
            llm = _OnlineRuntime(inputs.get("Online Request URL", ""), key, inputs.get("Online Model", ""))
        else:
            if inputs.get("Language Model") in {"No language models found", None}:
                raise FileNotFoundError("No local language model was found.")
            llm = _VisionRuntime.load(
                inputs["Language Model"], inputs["Vision Model"],
                inputs["GPU Offload Layers"], context_length, enable_thinking,
            )
        try:
            system = {"role": "system", "content": "You are an expert image prompt writer."}
            messages = [system, {"role": "user", "content": content}]
            completion_parameters = {
                "max_tokens": _completion_budget(context_length, 4096),
                "temperature": 0.8,
                "top_p": 0.95,
            }
            completion_parameters["seed"] = actual_seed

            def parse_prompts(text):
                lines = [line.strip().lstrip("-•* ").strip() for line in text.splitlines() if line.strip()]
                lines = [re.sub(r"^\d+[.)、:]\s*", "", line) for line in lines]
                return [line for line in lines if len(line) > 12 and not line.startswith("```")]

            def preserve_literals(prompt):
                missing = [item for item in literal_segments if item not in prompt]
                if missing:
                    prompt = prompt.rstrip(" ,，") + ", " + ", ".join(missing)
                return prompt

            raw = _stream_completion(llm, messages, "Image Prompt Generation", **completion_parameters)
            prompts = [preserve_literals(prompt) for prompt in parse_prompts(raw)]

            # Smaller local models sometimes answer with only one prompt despite
            # the requested count. Ask for the missing entries individually so
            # the node remains usable without failing the whole workflow.
            for index in range(len(prompts), count):
                remaining = count - index
                retry_content = list(content)
                retry_content.append({"type": "text", "text": (
                    f"The previous response contained too few entries. Generate exactly ONE additional, "
                    f"distinct prompt now (alternative {index + 1} of {count}). "
                    "Return only that single prompt on one line, with no numbering, explanation, or negative prompt."
                )})
                try:
                    retry_parameters = dict(completion_parameters)
                    retry_parameters["seed"] = (actual_seed + index) & 0xFFFFFFFF
                    retry_parameters["max_tokens"] = _completion_budget(context_length, 2048)
                    retry_parameters["temperature"] = 0.85
                    extra = _stream_completion(
                        llm, [system, {"role": "user", "content": retry_content}],
                        f"Image Prompt Generation ({index + 1}/{count})",
                        **retry_parameters,
                    )
                    candidates = parse_prompts(extra)
                    if candidates:
                        prompts.append(preserve_literals(candidates[0]))
                except Exception as retry_error:
                    print(f"[Image Prompt Generator] Unable to generate alternative {index + 1}: {retry_error}")

            # Do not turn a partially successful generation into a red node.
            # Returning the available prompts is preferable to aborting the
            # entire workflow; the output remains a valid STRING list.
            if not prompts:
                raise RuntimeError("The model did not return any usable image prompts.")
            return (prompts[:count],)
        finally:
            if not online:
                _VisionRuntime.close()


NODE_CLASS_MAPPINGS = {
    "H3Prompt": Qwen36MultiImageH3ChinesePrompt,
    "H3SaveImage": H3SaveImage,
    "H3ImagePromptGenerator": H3ImagePromptGenerator,
    "PromptEnhancerLoadImage": PromptEnhancerLoadImage,
    "RTXImageSuperResolution": RTXImageSuperResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Prompt": "H3 Prompt",
    "H3SaveImage": "H3 Save Image",
    "H3ImagePromptGenerator": "Image Prompt Generator",
    "PromptEnhancerLoadImage": "Load Image (recursive)",
    "RTXImageSuperResolution": "RTX Image Super Resolution (RGBA)",
}
