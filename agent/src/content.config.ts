import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const publications = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/publications' }),
  schema: z.object({
    title: z.string(),
    authors: z.string(),
    authorContributions: z.object({
      lead: z.array(z.string()).default([]),
      core: z.array(z.string()).default([]),
      highlighted: z.array(z.string()).default([]),
    }).optional(),
    venues: z.array(z.string()).min(1),
    date: z.string(),
    paperUrl: z.string().optional(),
    codeUrl: z.string().optional(),
    projectUrl: z.string().optional(),
    image: z.string().optional(),
    tags: z.array(z.string()).default([]),
    abstract: z.string().optional(),
  }),
});

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    date: z.string(),
    category: z.string().optional(),
    image: z.string().optional(),
    tags: z.array(z.string()).default([]),
    excerpt: z.string().optional(),
  }),
});

const tutorials = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/tutorials' }),
  schema: z.object({
    title: z.string(),
    date: z.string(),
    category: z.string().optional(),
    image: z.string().optional(),
    tags: z.array(z.string()).default([]),
    excerpt: z.string().optional(),
  }),
});

const talks = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/talks' }),
  schema: z.object({
    title: z.string(),
    type: z.enum(['Talk', 'Poster', 'Tutorial']),
    venue: z.string(),
    date: z.string(),
    location: z.string().optional(),
    slidesUrl: z.string().optional(),
    videoUrl: z.string().optional(),
    image: z.string().optional(),
  }),
});

const teaching = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/teaching' }),
  schema: z.object({
    title: z.string(),
    type: z.string(),
    venue: z.string(),
    date: z.string(),
    period: z.string().optional(),
    location: z.string().optional(),
    summary: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
});

export const collections = { publications, blog, tutorials, talks, teaching };
