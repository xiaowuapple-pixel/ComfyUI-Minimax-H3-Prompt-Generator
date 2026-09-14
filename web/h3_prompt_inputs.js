import { app } from "../../scripts/app.js";

// Keep the H3 Prompt node compact using real dynamic sockets (the same UX as
// H3 Reference). Optional sockets are created one at a time after the prior
// socket receives a link, so a newly-created node never shows a wall of ports.
app.registerExtension({
  name: "minimax.h3.prompt.dynamic-media-inputs",
  nodeCreated(node) {
    if (node.comfyClass !== "H3Prompt" && node.type !== "H3Prompt") return;
    const limits = { Image: 9, Video: 3, Audio: 3 };
    const typeFor = (group) => group === "Image" ? "IMAGE" : group.toUpperCase();
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
