const EXTERNAL_PROTOCOL = /^https?:\/\//i;

function visit(node, callback) {
  callback(node);

  if (Array.isArray(node.children)) {
    for (const child of node.children) {
      visit(child, callback);
    }
  }
}

export default function rehypeExternalLinks({ siteOrigin } = {}) {
  return (tree, file) => {
    const sourcePath = String(file.path ?? '').replaceAll('\\', '/');

    if (!sourcePath.includes('/src/content/blog/')) {
      return;
    }

    visit(tree, node => {
      if (node.type !== 'element' || node.tagName !== 'a') {
        return;
      }

      const href = node.properties?.href;
      if (typeof href !== 'string' || !EXTERNAL_PROTOCOL.test(href)) {
        return;
      }

      if (siteOrigin && new URL(href).origin === siteOrigin) {
        return;
      }

      const rel = Array.isArray(node.properties.rel)
        ? node.properties.rel
        : String(node.properties.rel ?? '').split(/\s+/).filter(Boolean);

      node.properties.target = '_blank';
      node.properties.rel = [...new Set([...rel, 'noopener', 'noreferrer'])];
    });
  };
}
