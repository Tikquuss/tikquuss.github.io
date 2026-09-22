const SKIPPED_TAGS = new Set(['code', 'kbd', 'math', 'pre', 'script', 'style']);
const MATH_CONTAINER_CLASSES = new Set([
  'explanation-popover',
  'explanation-trigger',
]);

function isEscaped(value, index) {
  let backslashes = 0;

  for (let cursor = index - 1; cursor >= 0 && value[cursor] === '\\'; cursor -= 1) {
    backslashes += 1;
  }

  return backslashes % 2 === 1;
}

function findClosingDelimiter(value, start) {
  for (let cursor = start; cursor < value.length; cursor += 1) {
    if (
      value[cursor] === '$'
      && value[cursor - 1] !== '$'
      && value[cursor + 1] !== '$'
      && !isEscaped(value, cursor)
    ) {
      return cursor;
    }
  }

  return -1;
}

function splitInlineMath(value) {
  const children = [];
  let textStart = 0;
  let cursor = 0;

  while (cursor < value.length) {
    if (
      value[cursor] !== '$'
      || value[cursor - 1] === '$'
      || value[cursor + 1] === '$'
      || isEscaped(value, cursor)
    ) {
      cursor += 1;
      continue;
    }

    const closing = findClosingDelimiter(value, cursor + 1);
    if (closing === -1 || closing === cursor + 1) {
      cursor += 1;
      continue;
    }

    if (cursor > textStart) {
      children.push({ type: 'text', value: value.slice(textStart, cursor) });
    }

    children.push({
      type: 'element',
      tagName: 'span',
      properties: { className: ['math', 'math-inline'] },
      children: [{ type: 'text', value: value.slice(cursor + 1, closing) }],
    });

    cursor = closing + 1;
    textStart = cursor;
  }

  if (textStart === 0) {
    return null;
  }

  if (textStart < value.length) {
    children.push({ type: 'text', value: value.slice(textStart) });
  }

  return children;
}

function hasMathContainerClass(node) {
  const classNames = node.properties?.className;
  const names = Array.isArray(classNames) ? classNames : [classNames];
  return names.some(name => MATH_CONTAINER_CLASSES.has(name));
}

function rendersEmbeddedMath(node) {
  return node.type === 'element'
    && (node.tagName === 'figcaption' || hasMathContainerClass(node));
}

function renderEmbeddedMath(node, insideMathContainer = false) {
  if (!Array.isArray(node.children)) {
    return;
  }

  const mathContext = insideMathContainer || rendersEmbeddedMath(node);

  for (let index = 0; index < node.children.length; index += 1) {
    const child = node.children[index];

    if (mathContext && child.type === 'text') {
      const replacement = splitInlineMath(child.value);
      if (replacement) {
        node.children.splice(index, 1, ...replacement);
        index += replacement.length - 1;
      }
      continue;
    }

    if (
      child.type === 'element'
      && !SKIPPED_TAGS.has(child.tagName)
      && !child.properties?.className?.includes?.('math-inline')
      && !child.properties?.className?.includes?.('math-display')
    ) {
      renderEmbeddedMath(child, mathContext);
    }
  }
}

export default function rehypeMathInHtml() {
  return tree => {
    renderEmbeddedMath(tree);
  };
}
