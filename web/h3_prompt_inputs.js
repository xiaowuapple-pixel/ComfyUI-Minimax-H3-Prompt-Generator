import { app } from "../../scripts/app.js";

// Dynamic sockets for the prompt nodes: one empty slot at a time, growing only
// when the previous one is linked, so a fresh node never shows a wall of ports.
//   H3 Prompt      -- 9 images, 3 videos, 3 audio clips
//   Qwen Image 2.1 Prompt Enhancer -- up to 10 reference images (the model's limit)
const NODE_LIMITS = {
  H3Prompt: { Image: 9, Video: 3, Audio: 3 },
  QwenImage21PromptEnhancer: { Image: 10 },
};

const TYPE_BY_GROUP = { Image: "IMAGE", Video: "VIDEO", Audio: "AUDIO" };

app.registerExtension({
  name: "minimax.h3.prompt.dynamic-media-inputs",
  nodeCreated(node) {
    const limits = NODE_LIMITS[node.comfyClass] || NODE_LIMITS[node.type];
    if (!limits) return;
    const typeFor = (group) => TYPE_BY_GROUP[group] || group.toUpperCase();
    const refresh = () => {
      for (const [group, limit] of Object.entries(limits)) {
        let inputs = (node.inputs || []).filter((item) => item.name.startsWith(`${group} `));
        // Keep linked sockets and one empty socket after the last linked one.
        let lastLinked = 0;
        inputs.forEach((item, index) => { if (item.link) lastLinked = index + 1; });
        const wanted = Math.min(limit, Math.max(1, lastLinked + 1));
        while (inputs.length < wanted) {
          const index = inputs.length + 1;
          node.addInput(`${group} ${index}`, typeFor(group));
          inputs = (node.inputs || []).filter((item) => item.name.startsWith(`${group} `));
        }
        while (inputs.length > wanted) {
          const extra = inputs[inputs.length - 1];
          if (extra.link) break;
          const absolute = node.inputs.indexOf(extra);
          if (absolute >= 0) node.removeInput(absolute);
          inputs = (node.inputs || []).filter((item) => item.name.startsWith(`${group} `));
        }
      }
      node.setSize(node.computeSize());
      node.setDirtyCanvas(true, true);
    };
    const original = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      const result = original?.apply(this, args);
      refresh();
      return result;
    };
    requestAnimationFrame(refresh);
  },
});
