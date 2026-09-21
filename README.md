# H3 Prompt

## 中文说明

独立的 ComfyUI 节点，使用本地 GGUF 视觉语言模型或在线 OpenAI 兼容 API，
为 MiniMax H3 生成可直接使用的视频提示词。支持 1-9 张参考图、纯文生视频、
最多 3 个参考视频、最多 3 段参考音频、纯文生视频、生成类型选择、
创意技能选择，以及英文/简体中文输出切换。

### 功能

- 扫描 `models/LLM` 和 `extra_model_paths.yaml` 注册的全部 LLM 路径
- 本地 GGUF + `mmproj` 视觉模型推理，支持 GPU 卸载层数
- 在线 OpenAI 兼容 `chat/completions` API，模型可自动从 `/models` 获取
- 在线模型刷新按钮与模型下拉选择，无需手动输入模型名
- “模型来源”使用开关切换：关闭为本地模型，开启为线上 LLM
- 自动判别、文生视频、图生视频、首尾帧、尾帧和多参考生成
- 支持 1-9 张参考图、1-3 个参考视频、1-3 段参考音频，输入端口按需动态增加
- 视频和音频使用 `<Video N>`、`<Audio N>` 标签引用，可用于指定对白、语音和节奏
- 用户输入引号内的对白会逐字保留，不会改写或截断
- 使用随仓库附带的 MiniMax 官方 skills 本地副本，不运行时联网
- 可选择生成后卸载本地模型（默认开启）
- 3D 动画、品牌宣传、产品广告、音乐字幕、纸艺科普等创意技能
- Image Prompt Generator：根据原始需求和最多两张参考图生成多条图像提示词
- 图像提示词支持 `SDXL / Illustrious / NoobAI Tags` 或 `Natural Language` 格式
- Image Prompt Generator 同样支持在线模型刷新按钮与模型下拉选择
- 提供常用画面比例：1:1、4:3、3:4、16:9、9:16、2:3、21:9，也可自动判断
- 生成时会把画面比例作为构图约束，考虑主体尺度、取景、裁切边界和留白
- 提供随机种子：固定种子可复现生成结果，设为 `-1` 时每次执行都会绕过缓存并重新随机
- Qwen Image 2.1 Prompt Enhancer：封装官方 prompt_rewrite 工具链，把简短需求扩写成 2.1 用的长提示词
- 提供「上下文长度」参数：决定 KV 缓存的显存占用，显存不足时调小可让更大的模型放进显卡
- 上下文调小后会自动压缩单次输出上限，避免生成到一半被截断
- 提供「Enable Thinking」开关（默认关闭）：开启后让推理模型先内部思考再输出，思考内容不会写进提示词
- 按 GGUF 元数据识别模型架构，而不是文件名，所以 Qwen3.8 等新命名也能用上正确的对话模板
- 两种格式均只输出正面提示词，不生成 negative prompt、negative tags 或排除项

### 安装与使用

将仓库目录放入 `ComfyUI/custom_nodes/` 后重启 ComfyUI。依赖文件说明：

- `requirements.txt`：基础依赖，在线 API 模式必须安装
- `requirements-local-gguf.txt`：可选依赖，只有使用本地 GGUF 时安装

在 ComfyUI 使用的 Python 环境中执行：

```bash
pip install -r requirements.txt
# 本地 GGUF 模式额外安装（选择与你的 CUDA/GPU 匹配的构建）
pip install -r requirements-local-gguf.txt
```

在线 API 模式无需安装 `llama-cpp-python`。
在节点中选择模型、生成类型和创意技能；无图且选择“自动判别”时会自动采用文生视频方式。
GPU 卸载层数默认为 `-1`，表示全部放入显存；显存不足时可改为 16-24。

在线模式要求服务支持 OpenAI 多模态消息格式；API Key 只在节点运行时使用，不写入文件。

### Qwen Image 2.1 Prompt Enhancer

把官方 [prompt_rewrite](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite)
工具链封装成一个节点。它用的是 Qwen-Image-2.1 官方提示词增强模型
（PE-T2I / PE-I2I，基于 Qwen3.5-VL 9B 微调），把简短需求扩写成 2.1 真正吃的长提示词。

- 两个任务各有独立权重和独立系统提示词，节点已原样内置在 `pe_prompts/`，不会与权重脱节
- 输出四路：`Positive Prompt`、`WH Ratio`、`Ratio Follow`、`Parse OK`
- `t2i` 只接受文字；`edit` 需要 1-4 张参考图，模型会按顺序用 `<image1>`… 引用，顺序不能乱
- 思考块始终开启（官方要求），思考内容不会写进提示词
- 采样默认使用官方出厂值：`t2i` 的 `presence_penalty=1.5`，`edit` 为 `0`；切成 `Custom` 才能手改
- 本地 GGUF 与在线 LLM 都支持，和 H3 Prompt 共用同一套运行时
- 解析失败时 `Positive Prompt` 回退为原始回答文本，并输出 `Parse OK=false`，不会静默丢结果

建议的本地模型（适配 16GB 显存，来源见下方链接）：

| 用途 | 文件 | 大小 |
| --- | --- | --- |
| t2i | `Qwen-Image-2.1-PE-T2I.Q5_K_M.gguf` | 6.02 GB |
| t2i（更省显存） | `Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf` | 5.24 GB |
| edit | `Qwen-Image-2.1-PE-I2I.Q5_K_M.gguf` + `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` | 6.02 + 0.86 GB |
| edit（更省显存） | `Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf` + `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` | 5.24 + 0.86 GB |

- 量化版：[Qwen-Image-2.1-PE-T2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-T2I-GGUF)、[Qwen-Image-2.1-PE-I2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF)
- 官方权重：[Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)、[Qwen/Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)（bf16 约 20 GB，16GB 显卡放不下）

推荐设置：`Context Length` 用 `32768`（PE 输出很长，官方 t2i 允许 16256 个新 token），
`GPU Offload Layers` 用 `-1`。把基座模型（例如普通的 Qwen3.5-9B）填进去也能跑，
但它没按这套系统提示词训练过，`Parse OK` 基本会是 false。

参考：[官方 prompt_rewrite 文档](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite)

## English

Standalone ComfyUI node for generating production-ready MiniMax H3 video prompts with a local
GGUF vision-language model or an OpenAI-compatible hosted API. It supports 1-9 reference images,
up to three reference videos, up to three reference audio clips, text-to-video mode,
generation-type selection, creative skill selection, and English/Chinese output.

### Features

- Scans local `models/LLM` plus paths registered in `extra_model_paths.yaml`
- Local GGUF + `mmproj` vision inference with configurable GPU offload layers (default: `-1`, all layers on GPU)
- OpenAI-compatible `chat/completions` API with automatic `/models` discovery
- Refresh button and dropdown selection for hosted models; no manual model-name entry required
- A simple source switch selects local models (off) or hosted LLM (on)
- Auto, text-to-video, image-to-video, first/last-frame, last-frame, and multi-reference modes
- 1-9 reference images, 1-3 reference videos, and 1-3 reference audio clips, with sockets added on demand
- Videos and audio are referenced with `<Video N>` and `<Audio N>` tags for dialogue, speech, and rhythm
- Quoted dialogue in the request is reproduced verbatim and is never rewritten or truncated
- Vendored MiniMax official skills are used locally; no runtime network synchronization
- Optional model unload after generation (enabled by default)
- Creative skills for 3D shorts, brand ads, product ads, music subtitles, and paper-craft explainers
- Image Prompt Generator creates multiple image prompts from the request and up to two reference images
- Image prompts support `SDXL / Illustrious / NoobAI Tags` and `Natural Language` formats
- Image Prompt Generator shares the same online model refresh button and dropdown selection
- Common aspect ratios are available: 1:1, 4:3, 3:4, 16:9, 9:16, 2:3, and 21:9, plus Auto
- The selected ratio is treated as a composition constraint for framing, subject scale, crop boundaries, and negative space
- A Seed control is available: fixed seeds improve reproducibility; `-1` bypasses the execution cache and selects a new random seed on every run
- Qwen Image 2.1 Prompt Enhancer: wraps the official prompt_rewrite toolchain to expand a short request into a 2.1-ready long prompt
- A Context Length control sets the KV-cache footprint, so a smaller window lets a larger model stay on the GPU
- A smaller context length automatically trims the per-call reply budget so responses are not cut off halfway
- An Enable Thinking switch (off by default) lets reasoning models think internally while the reasoning is kept out of the prompt output
- The chat handler is chosen from the GGUF metadata rather than the file name, so newer names such as Qwen3.8 still get the correct chat template
- Both formats output positive prompts only; no negative prompts, negative tags, or exclusions are generated

### Qwen Image 2.1 Prompt Enhancer

Wraps the official
[prompt_rewrite](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite) toolchain.
It drives the official Qwen-Image-2.1 prompt-enhancer checkpoints (PE-T2I / PE-I2I, fine-tuned
Qwen3.5-VL 9B) and turns a short request into the long prompt 2.1 expects.

- Each task has its own checkpoint and its own system prompt; both prompts ship verbatim in `pe_prompts/`
- Four outputs: `Positive Prompt`, `WH Ratio`, `Ratio Follow`, `Parse OK`
- `t2i` takes text only; `edit` takes 1-4 reference images, referenced as `<image1>`... in connection order
- Thinking stays on (required by the official models) and never leaks into the prompt
- Official per-task sampling by default (`presence_penalty` 1.5 for t2i, 0 for edit); switch to `Custom` to override
- Local GGUF and hosted LLM sources, sharing the same runtime as H3 Prompt
- On a parse failure `Positive Prompt` falls back to the raw answer and `Parse OK` is false, so nothing is lost silently

Suggested local models for a 16 GB card ([Qwen-Image-2.1-PE-T2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-T2I-GGUF),
[Qwen-Image-2.1-PE-I2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF)):

| Task | Files | Size |
| --- | --- | --- |
| t2i | `Qwen-Image-2.1-PE-T2I.Q5_K_M.gguf` | 6.02 GB |
| t2i (leaner) | `Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf` | 5.24 GB |
| edit | `Qwen-Image-2.1-PE-I2I.Q5_K_M.gguf` + `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` | 6.02 + 0.86 GB |
| edit (leaner) | `Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf` + `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` | 5.24 + 0.86 GB |

Use `Context Length` 32768 (PE answers are long; official t2i allows 16256 new tokens) and
`GPU Offload Layers` -1. A stock base model will load too, but it was never trained against these
system prompts, so `Parse OK` will be false on most runs.

### Installation

Place this folder under `ComfyUI/custom_nodes/` and restart ComfyUI. Install the base dependencies
from `requirements.txt`. For local GGUF inference, additionally install the optional dependencies
from `requirements-local-gguf.txt` using a build compatible with your CUDA/GPU. Online API mode does
not require `llama-cpp-python`.
With no image connected and generation type set to Auto, the node automatically uses text-to-video.

For online mode, the endpoint must support OpenAI multimodal messages. API keys are used only at
runtime and are never written to disk.

## License

See the upstream MiniMax-H3 license for the bundled official skills and the repository license.
