import { defineCollection, z } from 'astro:content';

const publications = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    authors: z.string(),
    venue: z.string(),
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
  type: 'content',
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
  type: 'content',
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
  type: 'content',
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
  type: 'content',
  schema: z.object({
    title: z.string(),
    type: z.string(),
    venue: z.string(),
    date: z.string(),
    location: z.string().optional(),
    summary: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
});

export const collections = { publications, blog, tutorials, talks, teaching };
