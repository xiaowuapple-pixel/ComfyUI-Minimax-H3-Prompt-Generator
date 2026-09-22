from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .qwen_image21_pe import (
    PromptEnhancerReleaseTextEncoder,
    QwenImage21PELoaderGGUF,
    QwenImage21PELoaderSafetensors,
    QwenImage21PESettings,
    QwenImage21PromptEnhancer,
    QwenImage21TextEncodeList,
)
from . import api  # noqa: F401

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    "QwenImage21PELoaderSafetensors": QwenImage21PELoaderSafetensors,
    "QwenImage21PELoaderGGUF": QwenImage21PELoaderGGUF,
    "QwenImage21PESettings": QwenImage21PESettings,
    "QwenImage21PromptEnhancer": QwenImage21PromptEnhancer,
    "QwenImage21TextEncodeList": QwenImage21TextEncodeList,
    "PromptEnhancerReleaseTextEncoder": PromptEnhancerReleaseTextEncoder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    "QwenImage21PELoaderSafetensors": "Qwen Image 2.1 PE Loader (safetensors)",
    "QwenImage21PELoaderGGUF": "Qwen Image 2.1 PE Loader (GGUF)",
    "QwenImage21PESettings": "Qwen Image 2.1 PE Settings",
    "QwenImage21PromptEnhancer": "Qwen Image 2.1 Prompt Enhancer",
    "QwenImage21TextEncodeList": "Text Encode Qwen Image 2.1 (List)",
    "PromptEnhancerReleaseTextEncoder": "Release Text Encoder (VRAM)",
}


WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
