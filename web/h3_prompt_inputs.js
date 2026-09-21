import { app } from "../../scripts/app.js";

// Dynamic sockets for the prompt nodes: one empty slot at a time, growing only
// when the previous one is linked, so a fresh node never shows a wall of ports.
// The backend always declares the full set; this only decides what is visible.
//
//   H3 Prompt                        -- 9 images, 3 videos, 3 audio clips  ("Image 1")
//   Qwen Image 2.1 Prompt Enhancer   -- 10 reference images              ("Image 1")
//   Text Encode Qwen Image 2.1 (List)-- 10 reference images              ("image_1")
//
// The separator differs because ComfyUI names the stock node's sockets
// `image_1`, and matching that keeps workflows wire-compatible with it.
const GROUPS = {
  Image: { type: "IMAGE", separator: " " },
  Video: { type: "VIDEO", separator: " " },
  Audio: { type: "AUDIO", separator: " " },
  image: { type: "IMAGE", separator: "_" },
};

const NODE_LIMITS = {
  H3Prompt: { Image: 9, Video: 3, Audio: 3 },
  QwenImage21PromptEnhancer: { Image: 10 },
  QwenImage21TextEncodeList: { image: 10 },
};

app.registerExtension({
  name: "prompt.enhancer.dynamic-media-inputs",
  nodeCreated(node) {
    const limits = NODE_LIMITS[node.comfyClass] || NODE_LIMITS[node.type];
    if (!limits) return;
    const socketName = (group, index) =>
      `${group}${(GROUPS[group] || GROUPS.Image).separator}${index}`;
    const socketType = (group) => (GROUPS[group] || GROUPS.Image).type;
    const refresh = () => {
      for (const [group, limit] of Object.entries(limits)) {
        let inputs = (node.inputs || []).filter((item) => item.name.startsWith(socketName(group, "")));
        // Keep linked sockets and one empty socket after the last linked one.
        let lastLinked = 0;
        inputs.forEach((item, index) => { if (item.link) lastLinked = index + 1; });
        const wanted = Math.min(limit, Math.max(1, lastLinked + 1));
        while (inputs.length < wanted) {
          const index = inputs.length + 1;
          node.addInput(socketName(group, index), socketType(group));
          inputs = (node.inputs || []).filter((item) => item.name.startsWith(socketName(group, "")));
        }
        while (inputs.length > wanted) {
          const extra = inputs[inputs.length - 1];
          if (extra.link) break;
          const absolute = node.inputs.indexOf(extra);
          if (absolute >= 0) node.removeInput(absolute);
          inputs = (node.inputs || []).filter((item) => item.name.startsWith(socketName(group, "")));
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
