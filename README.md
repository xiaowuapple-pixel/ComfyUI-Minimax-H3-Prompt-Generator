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
- 另外输出 `Width` / `Height` 两个整数：t2i 按模型选的画幅 + `Target Megapixels` 换算，edit 直接沿用参考图尺寸
- `t2i` 只接受文字；`edit` 最多 10 张参考图（模型上限），按顺序用 `<image1>`… 引用，顺序不能乱
- 图片端口是动态的：默认只显示 `Image 1`，连上以后才长出 `Image 2`，依次类推，最多 10 个
- 思考默认开启，长度由加载节点上的 `Plan Tokens` 控制（默认 800，`-1` 为不限）；思考内容始终不会写进提示词
- 采样默认使用官方出厂值：`t2i` 的 `presence_penalty=1.5`，`edit` 为 `0`；切成 `Custom` 才能手改
- 本地 GGUF 与在线 LLM 都支持，和 H3 Prompt 共用同一套运行时
- 解析失败时 `Positive Prompt` 回退为原始回答文本，并输出 `Parse OK=false`，不会静默丢结果

这套功能是四个节点，各管一件事：

| 节点 | 负责什么 |
| --- | --- |
| **Qwen Image 2.1 PE Loader (safetensors)** | 官方 PE 权重（推荐）：`T2I Encoder` / `I2I Encoder` 两个选择，从 `text_encoders` 里读 |
| **Qwen Image 2.1 PE Loader (GGUF)** | PE 的 GGUF 量化版：`T2I GGUF` + `I2I GGUF` + `Vision Model`(mmproj) + `GPU Offload Layers`。实测比原生 int8 快约 5 倍 |
| **Qwen Image 2.1 PE Settings**（可选） | 采样参数：预设、temperature、top_p、top_k、presence_penalty、max_new_tokens。接到主节点的 `pe_settings`；**不接就用官方出厂值** |
| **Qwen Image 2.1 Prompt Enhancer** | 只有每次运行才会变的东西：提示词、任务、种子、画幅、目标像素、缓存开关、输出条数，以及 1-10 张参考图 |

两个加载节点输出同一种 `PE Model`，**用哪个就接哪个**——这样每个模式只显示它需要的选择，
不会出现"选了 A 还要面对 B 的空白控件"：

| 你的情况 | 用哪个加载节点 | 它上面有几个选择 |
| --- | --- | --- |
| 官方 safetensors（质量基准） | `PE Loader (safetensors)` | 两个：T2I、I2I |
| 想要速度（GGUF 量化） | `PE Loader (GGUF)` | 三个：T2I、I2I、mmproj |

> 两个加载节点都自己负责载入与释放：一次只驻留一个模型，用完就还显存。
> 两个任务的选择只做一次（把 T2I 和 I2I 都选好），**由主节点按有没有连图自动决定用哪个**，
> 不需要自己保证"任务和模型"配对。

> 这个节点**只服务于 PE 权重**。在线 LLM 和普通 Qwen3.5 之类的通用模型已从节点里移除：
> 它们能读到系统提示词、甚至能照结构输出，但没在这套答案契约上训练过，吐不出下游要的 JSON，
> 跑一次要几分钟却只换来 `Parse OK=false`。GGUF 下拉里也只列出 PE 检查点。

`Task` 默认是 `Auto (by images)`：**连了图片就走 edit，没连图片就走 t2i**，节点按这个自动选权重和系统提示词。
两个 PE 编码器各 8.8 GB，不可能同时放进 16GB 显卡，所以节点全程只驻留一个：
只有在真正需要时才载入，切换任务时先释放另一个，`Unload Model After Generation` 打开时用完即放。
释放只丢掉编码器自己的引用（实测 14.6 GB → 1.4 GB），**不会牵连你的扩散模型**。

输出怎么接：

| 输出 | 接到哪 |
| --- | --- |
| `Positive Prompt` | 2.1 的文本编码节点（`CLIPTextEncode` / `TextEncodeQwenImage21`）的正向输入 |
| `Width` / `Height` | 直接接 `空Latent`（EmptyLatentImage）的 width/height。这是最省事的一路 |
| `WH Ratio` | 不用接，是模型选画幅的原始记录（形如 `16:9`）。注意官方 `分辨率选择器` 的选项带后缀（`16:9 (Widescreen)`），直接连过去校验不过，所以宽高已经帮你算好了 |
| `Ratio Follow` | 不用接，edit 专用信息（形如 `<image1>`），表示输出沿用哪张参考图的画幅 |
| `Parse OK` | 不用接。`false` 表示模型没吐出预期 JSON，此时 `Positive Prompt` 是原始回答文本，可以用来判断这次结果要不要用 |

画幅有专门的 `Aspect Ratio` 选择：默认 `Auto (model decides)` 是官方行为（模型自己定，结果从 `WH Ratio` 读）。
想固定就选一个比例，节点会做两件事：把这个画幅写进交给模型的请求（中英双语标注，
避免 edit 任务的输出语言被带偏），并让 `Width` / `Height` 按它计算。这样提示词描述的构图和实际画布是一致的。
如果模型的判断和你的设定不同，日志里会提示一句。

`Prompt Count` 决定一次输出几条提示词。六路输出都是**列表**（长度等于条数），
下游节点会按条数各跑一次——比如接一个 KSampler 就是一轮出 N 张。

- 同批各条用 `Seed`、`Seed+1`、`Seed+2`…，所以固定 Seed 能复现整批
- **耗时基本线性叠加**：官方契约一次回答只给一条，每条都得完整生成一遍。
  16GB 卡 + GGUF 实测：t2i 每条约 20-35 秒，edit 约 45-50 秒
- 唯一省下的是模型载入：整批只载入一次、只释放一次（省十几秒，不是每条都省）
- 配合缓存：同一批重复执行是毫秒级

官方 PE 编码器（放到 `models/text_encoders/`）：

- `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` — t2i 扩写
- `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors` — 带图改写真

来源：[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。每个 8.82 GB，16GB 显卡可以跑。
实测 4080 上约 14 token/s，一次 t2i 扩写（含思考块）约 2 分钟。

### 关于速度

4080 16GB + Q5_K_M GGUF，一次带图改写实测（请求：五张参考图 + 角色卡设定）：

| 模式 | 耗时 | 提示词 |
| --- | --- | --- |
| `Direct`（关思考） | 约 9 秒 | 455 字，没有权衡需求的过程，细节明显变少 |
| `Think` + `Plan Tokens` 800（默认） | 约 26 秒 | 688 字 |
| `Think` + `Plan Tokens` 400 | 约 15 秒 | 666 字 |
| `Think` + `Plan Tokens` -1（官方完整思考） | 约 92 秒 | 720 字 |
| 原生 int8 safetensors（开思考） | 十分钟以上 | — |

时间几乎全花在那轮思考上：同一条请求，模型写了 **7038 个 token 的规划、89 秒**，
而真正的答案只有 350 个 token。所以 `Plan Tokens` 是这里最值钱的开关：写到上限就收住，
把已经写好的规划交还给模型直接写答案（多付一次预填充，约 2-3 秒）。
规划越短越容易漏掉请求里的隐含要求——上面同一条请求里"附带详细文字说明"，
完整思考的版本写出了文字框和标题，400 的版本就没写。

这两个 checkpoint **没有 MTP 头**，投机解码用不上（节点里的"auto"会退化成普通采样）。

能用的加速手段，按效果排序：

1. **调 `Plan Tokens`**。默认 800（约 26 秒）；追求快就 400（约 15 秒），
   追求细节就 `-1`（约 92 秒）。它只在 `Thinking` 打开时生效。
2. **反复调提示词时关掉 `Unload Model After Generation`**。节点会留着已载入的模型，
   下一次运行连载入都省掉（实测 4.0 秒 → 0.0 秒）。跑完整出图工作流前记得再打开，
   否则那 6 GB 会一直占着显存。
3. **用 GGUF 路线**。同机实测：ComfyUI 原生 int8_convrot 约 14 token/s，
   llama.cpp 跑同级别 9B GGUF 约 75 token/s。见上面 GGUF 表格。
4. **开 `Use Cache`（默认开启）**。相同请求 + 相同种子会直接复用上次结果，第二次起是毫秒级。
   反复调图时这个开关能省掉绝大部分等待。缓存放在 `ComfyUI/user/qwen_image21_pe_cache/`，
   要强制重新生成就关掉它或清空这个目录。
5. **不用管输出上限**。节点在答案的 JSON 闭合的那一刻就停止生成，
   官方那两个巨大的上限（16256 / 24000 个新 token）不再意味着要等满。

建议的本地模型（适配 16GB 显存，来源见下方链接）：

| 用途 | 文件 | 大小 |
| --- | --- | --- |
| t2i | `Qwen-Image-2.1-PE-T2I.Q5_K_M.gguf` | 6.02 GB |
| t2i（更省显存） | `Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf` | 5.24 GB |
| edit | `Qwen-Image-2.1-PE-I2I.Q5_K_M.gguf` + `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` | 6.02 + 0.86 GB |
| edit（更省显存） | `Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf` + `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` | 5.24 + 0.86 GB |

- 量化版：[Qwen-Image-2.1-PE-T2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-T2I-GGUF)、[Qwen-Image-2.1-PE-I2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF)
- 官方权重：[Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)、[Qwen/Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)（bf16 约 20 GB，16GB 显卡放不下）

推荐设置：`Context Length` 用 `16384`，`GPU Offload Layers` 用 `-1`。
把基座模型（例如普通的 Qwen3.5-9B）填进去也能跑，
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
- Six outputs: `Positive Prompt`, `WH Ratio`, `Ratio Follow`, `Parse OK`, `Width`, `Height`
- `Width` / `Height` are pixels: t2i scales the model's ratio to `Target Megapixels`, edit reuses the source image's size
- `t2i` takes text only; `edit` takes up to 10 reference images (the model's limit), referenced as `<image1>`... in connection order
- Image sockets are dynamic: only `Image 1` shows at first, and connecting it reveals `Image 2`, up to ten
- Thinking is on by default and its length is capped by the loader's `Plan Tokens` (800 by default, -1 for no cap);
  the reasoning never leaks into the prompt
- Official per-task sampling by default (`presence_penalty` 1.5 for t2i, 0 for edit); switch to `Custom` to override
- Local GGUF and hosted LLM sources, sharing the same runtime as H3 Prompt
- On a parse failure `Positive Prompt` falls back to the raw answer and `Parse OK` is false, so nothing is lost silently

This is four nodes now, each owning one concern:

| Node | Owns |
| --- | --- |
| **Qwen Image 2.1 PE Loader (safetensors)** | The official PE weights (recommended): two pickers, `T2I Encoder` / `I2I Encoder`, read from `text_encoders` |
| **Qwen Image 2.1 PE Loader (GGUF)** | The PE GGUF quants: `T2I GGUF` + `I2I GGUF` + `Vision Model` (mmproj) + `GPU Offload Layers`. Measured about 5x faster than the native int8 path |
| **Qwen Image 2.1 PE Settings** (optional) | Sampling: preset, temperature, top_p, top_k, presence_penalty, max_new_tokens. Wire it into `pe_settings`; **leave it off to use the official settings** |
| **Qwen Image 2.1 Prompt Enhancer** | Only what changes per run: prompt, task, seed, aspect ratio, target megapixels, cache toggle, prompt count, and 1-10 reference images |

Both loaders output the same `PE Model` type, so you wire whichever one matches your case -- and each shows
only the pickers it needs:

| Your case | Loader | Pickers on it |
| --- | --- | --- |
| Official safetensors (quality reference) | `PE Loader (safetensors)` | two: T2I, I2I |
| Speed (GGUF quants) | `PE Loader (GGUF)` | three: T2I, I2I, mmproj |

> Both loaders own loading and unloading: one model resident at a time, released as soon as the run ends.
> You pick both tasks once (T2I and I2I); the enhancer decides which one a run needs from whether images
> are connected -- no need to keep "task and model" in sync by hand.

> This node serves PE checkpoints only. The online-LLM path and plain Qwen3.5-style models were removed:
> they read the system prompt and even mirror its structure, but they were never trained on its answer
> contract, so a run that costs minutes returns `Parse OK=false`. The GGUF dropdown lists PE checkpoints only.

`Task` defaults to `Auto (by images)`: images connected means an edit request, no images means text-to-image,
and the node picks the matching weights and system prompt by itself. The two PE encoders are 8.8 GB each and
cannot both fit a 16 GB card, so only one is ever resident: it is loaded on first need, the other is released
when the task switches, and `Unload Model After Generation` frees it after the run. That release only drops the
encoder's own reference (measured 14.6 GB -> 1.4 GB) and never touches your diffusion model.

Where the outputs go:

| Output | Destination |
| --- | --- |
| `Positive Prompt` | the positive side of the 2.1 text encoder (`CLIPTextEncode` / `TextEncodeQwenImage21`) |
| `Width` / `Height` | straight into an Empty Latent Image (or any width/height input) -- the simplest path |
| `WH Ratio` | leave unconnected; it is the model's own record of the canvas (`16:9`). The core Resolution Selector expects labels with suffixes (`16:9 (Widescreen)`), so a direct link will not validate -- that is why Width/Height are computed for you |
| `Ratio Follow` | leave unconnected; edit only, names the reference image whose framing the output keeps (`<image1>`) |
| `Parse OK` | leave unconnected; `false` means the answer was not the expected JSON and Positive Prompt holds the raw text |

`Aspect Ratio` controls the canvas: `Auto (model decides)` is the official behaviour (read the result from `WH Ratio`).
Pick a ratio to fix it and the node does two things: it tells the model about it (bilingual marker, so an edit run's
output language is not dragged along) and it makes `Width` / `Height` follow it. The description and the frame then
agree, and the log notes it when the model's own choice differed.

`Prompt Count` sets how many prompts a run returns. All six outputs are **lists** (length = count), so downstream
nodes run once per prompt -- one KSampler wired to it renders N images.

- Variants use `Seed`, `Seed+1`, `Seed+2`, ... so a fixed seed reproduces the whole batch
- **Cost is essentially linear**: the official contract allows one answer per response, so every prompt is a full
  generation. Measured on a 16 GB card with GGUF: about 20-35 s per t2i prompt, 45-50 s per edit prompt
- The only saving is the model load: one load and one release for the whole batch (tens of seconds, not per prompt)
- With the cache on, re-running the same batch costs milliseconds

Official PE encoders (put them in `models/text_encoders/`):

- `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` for t2i expansion
- `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors` for edit rewriting

From [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). Each is 8.82 GB and fits a 16 GB card.
Measured on a 4080: about 14 tokens/s, so one t2i expansion including its thinking block takes roughly two minutes.

### About speed

Measured on a 4080 (16 GB) with the Q5_K_M GGUF, one edit rewrite (five reference images plus a character-card
request):

| Mode | Time | Prompt |
| --- | --- | --- |
| `Direct` (thinking off) | ~9 s | 455 characters, written without weighing the request |
| `Think` + `Plan Tokens` 800 (default) | ~26 s | 688 characters |
| `Think` + `Plan Tokens` 400 | ~15 s | 666 characters |
| `Think` + `Plan Tokens` -1 (the official setting) | ~92 s | 720 characters |
| Native int8 safetensors (thinking on) | ten minutes or more | -- |

The time is almost entirely the plan: the same request emitted 7038 tokens of planning (89 s) against a
350-token answer. `Plan Tokens` is the lever -- the node stops reading once the budget is spent and hands the
plan back so the model writes the answer (one extra prefill, 2-3 s). Shorter plans miss implied requirements:
on that request "include a detailed text description", the full plan produced a text panel with a title, the
400-token one did not. Neither checkpoint ships an MTP head, so speculative decoding is not available (the
node's "auto" quietly falls back to plain sampling).

What actually helps, best first:

1. **Pick a `Plan Tokens` budget.** 800 is the default (~26 s); 400 is ~15 s; `-1` is the official full plan
   (~92 s). It only applies while `Thinking` is on.
2. **Turn `Unload Model After Generation` off while you iterate.** The node then keeps the runtime resident and
   a second run skips the load entirely (measured: 4.0 s -> 0.0 s). Turn it back on before queueing a full image
   workflow, or the 6 GB stays parked in VRAM.
3. **Use the GGUF path.** ComfyUI's native int8_convrot runs at about 14 tokens/s; llama.cpp runs the same 9B
   quant at about 75 tokens/s. Both numbers are from the same card.
4. **Keep `Use Cache` on (default).** Same request plus same seed reuses the previous result, so repeat runs are
   instant. The cache lives in `ComfyUI/user/qwen_image21_pe_cache/`; disable it or clear that folder to force a
   fresh generation.
5. **Leave the output budget alone.** The node stops the moment the answer's JSON closes, so the enormous official
   ceilings (16256 / 24000 new tokens) no longer cost anything.

Suggested local models for a 16 GB card ([Qwen-Image-2.1-PE-T2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-T2I-GGUF),
[Qwen-Image-2.1-PE-I2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF)):

| Task | Files | Size |
| --- | --- | --- |
| t2i | `Qwen-Image-2.1-PE-T2I.Q5_K_M.gguf` | 6.02 GB |
| t2i (leaner) | `Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf` | 5.24 GB |
| edit | `Qwen-Image-2.1-PE-I2I.Q5_K_M.gguf` + `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` | 6.02 + 0.86 GB |
| edit (leaner) | `Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf` + `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` | 5.24 + 0.86 GB |

Use `Context Length` 16384 and `GPU Offload Layers` -1. The default window used to be 32768; with the thinking
block off a reply is only a few hundred tokens, and the smaller window frees about 3 GB of VRAM without costing
any speed. Raise it only if you turn `Thinking` on or feed very long reference material.
A stock base model will load too, but it was never trained against these
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
