from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .qwen_image21_pe import QwenImage21PromptEnhancer
from . import api  # noqa: F401

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    "QwenImage21PromptEnhancer": QwenImage21PromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    "QwenImage21PromptEnhancer": "Qwen Image 2.1 Prompt Enhancer",
}


def _patch_recursive_input_images():
    """Keep Load Image combo entries working for input subdirectories after updates."""
    try:
        import os
        import nodes
        import folder_paths

        def input_types(cls):
            input_dir = folder_paths.get_input_directory()
            files = []
            for root, _, names in os.walk(input_dir):
                for name in names:
                    full_path = os.path.join(root, name)
                    if os.path.isfile(full_path):
                        files.append(os.path.relpath(full_path, input_dir).replace(os.sep, "/"))
            files = folder_paths.filter_files_content_types(files, ["image"])
            return {"required": {"image": (sorted(files), {"image_upload": True})}}

        nodes.LoadImage.INPUT_TYPES = classmethod(input_types)
    except Exception:
        pass


_patch_recursive_input_images()

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
