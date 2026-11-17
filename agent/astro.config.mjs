import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import rehypeRaw from 'rehype-raw';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeExternalLinks from './src/lib/rehypeExternalLinks.mjs';
import rehypeMathInHtml from './src/lib/rehypeMathInHtml.mjs';

export default defineConfig({
  site: 'https://tikquuss.github.io',
  // Astro 7 defaults to 'jsx', which deletes line breaks between text and tags
  // ("at⏎<a>Mila</a>" renders as "atMila"); true keeps a space, as in Astro 4.
  compressHTML: true,
  integrations: [mdx()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeRaw,
        rehypeMathInHtml,
        rehypeKatex,
        [rehypeExternalLinks, { siteOrigin: 'https://tikquuss.github.io' }],
      ],
    }),
  },
});
