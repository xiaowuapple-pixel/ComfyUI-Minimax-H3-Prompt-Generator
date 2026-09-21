import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Prompt nodes that expose the shared online-LLM widgets (request URL, API key
// and model name) and use the same backend route to list hosted models.
const PICKER_NODES = new Set(["H3Prompt", "H3ImagePromptGenerator", "QwenImage21PromptEnhancer"]);

// H3 Prompt places its refresh button directly under the model combo. Newer
// nodes append the button at the end of the widget list so the values already
// stored in saved workflows keep pointing at the same widgets.
const APPEND_BUTTON_NODES = new Set(["H3ImagePromptGenerator", "QwenImage21PromptEnhancer"]);

const LEGACY_WIDGET_VALUES = {
    "自动判别": "Auto Detect",
    "通用 H3 提示词": "General H3 Prompt",
    "3D 动画短片": "3D Animated Short",
    "品牌宣传片": "Brand Promo",
    "合作游戏片头": "Co-op Game Intro",
    "手绘实拍融合": "Hand-drawn Live Action",
    "极简产品广告": "Minimalist Product Ad",
    "音乐字幕视频": "Music Subtitle Video",
    "纸张拼贴科普": "Paper Collage Explainer",
    "纸艺定格科普": "Papercraft Stop-motion Explainer",
    "文生视频": "Text-to-Video",
    "图生视频": "Image-to-Video",
    "首尾帧生成": "First/Last Frame",
    "尾帧生成": "Last Frame",
    "多参考生成": "Multi-Reference",
};

const MIGRATED_WIDGETS = new Set(["Creative Skill", "Generation Type"]);

async function requestOnlineModels(addressWidget, keyWidget) {
    const response = await api.fetchApi("/qwen-h3-prompt/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            base_url: addressWidget.value || "",
            api_key: keyWidget.value || "",
        }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data.models;
}

function installOnlineModelPicker(nodeType, appendButton) {
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        // Older workflows may carry the legacy Chinese image widgets as
        // unknown fields. Remove those duplicates from the visible node;
        // the Python executor still accepts their values for compatibility.
        if (this.widgets) {
            this.widgets = this.widgets.filter(
                (widget) => !/^图片_[1-9]$/.test(widget.name) && !/^Reference Image [1-9]$/.test(widget.name)
            );
        }

        const addressWidget = this.widgets?.find((widget) => widget.name === "Online Request URL");
        const keyWidget = this.widgets?.find((widget) => widget.name === "Online API Key");
        const modelWidget = this.widgets?.find((widget) => widget.name === "Online Model");
        if (!addressWidget || !keyWidget || !modelWidget) return result;

        const savedModel = modelWidget.value || "";
        const modelIndex = this.widgets.indexOf(modelWidget);

        // Recreate the field as a real combo widget. Changing `type` on an
        // existing string widget does not change LiteGraph's renderer.
        this.widgets.splice(modelIndex, 1);
        const combo = this.addWidget("combo", "Online Model", savedModel, null, {
            values: savedModel ? [savedModel] : [],
            tooltip: "在线模型名称。先点击下方按钮刷新，再在下拉列表中选择。",
        });
        combo.tooltip = "在线模型名称。先点击下方按钮刷新，再在下拉列表中选择。";
        this.widgets.splice(this.widgets.indexOf(combo), 1);
        this.widgets.splice(modelIndex, 0, combo);

        const refresh = this.addWidget(
            "button",
            "Refresh Online Models",
            null,
            async () => {
                const previousLabel = refresh.name;
                refresh.name = "正在获取模型...";
                this.setDirtyCanvas(true, true);
                try {
                    const models = await requestOnlineModels(addressWidget, keyWidget);
                    combo.options.values = models;
                    if (!models.includes(combo.value)) combo.value = models[0];
                    refresh.name = `Refresh Online Models (${models.length})`;
                } catch (error) {
                    refresh.name = previousLabel;
                    alert(`获取在线模型失败：${error.message}`);
                }
                this.setDirtyCanvas(true, true);
            },
            { tooltip: "使用上面的请求地址和 API Key 拉取在线平台的可用模型列表。" }
        );
        refresh.tooltip = "使用上面的请求地址和 API Key 拉取在线平台的可用模型列表。";

        if (!appendButton) {
            const refreshIndex = this.widgets.indexOf(refresh);
            if (refreshIndex > modelIndex + 1) {
                this.widgets.splice(refreshIndex, 1);
                this.widgets.splice(modelIndex + 1, 0, refresh);
            }
        }

        this.setSize(this.computeSize());
        return result;
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = originalConfigure?.apply(this, arguments);
        for (const widget of this.widgets || []) {
            if (MIGRATED_WIDGETS.has(widget.name) && LEGACY_WIDGET_VALUES[widget.value]) {
                widget.value = LEGACY_WIDGET_VALUES[widget.value];
            }
        }
        return result;
    };
}

app.registerExtension({
    name: "QwenH3Prompt.OnlineModels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PICKER_NODES.has(nodeData.name)) return;
        installOnlineModelPicker(nodeType, APPEND_BUTTON_NODES.has(nodeData.name));
    },
});
