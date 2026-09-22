import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import rehypeRaw from 'rehype-raw';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeExternalLinks from './src/lib/rehypeExternalLinks.mjs';
import rehypeMathInHtml from './src/lib/rehypeMathInHtml.mjs';

export default defineConfig({
  site: 'https://tikquuss.github.io',
  integrations: [mdx()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeRaw,
      rehypeMathInHtml,
      rehypeKatex,
      [rehypeExternalLinks, { siteOrigin: 'https://tikquuss.github.io' }],
    ],
  },
});
