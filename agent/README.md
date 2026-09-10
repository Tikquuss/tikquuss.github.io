# Pascal Research Website

New Astro personal academic website for Pascal Jr. Tikeng Notsawo.

## Launch Locally

```bash
npm ci
npm run dev
```

Astro prints the local URL, usually:

```text
http://localhost:4321
```

Use another port if needed:

```bash
npm run dev -- --port 4322
```

On Windows, do not run `npm ci` while `npm run dev` is still running. Vite/Astro keeps
`node_modules/@esbuild/.../esbuild.exe` open, and `npm ci` deletes `node_modules` before
reinstalling it. Stop the dev server first with `Ctrl+C` in the same terminal that started it.

If a hidden or old process is still locking the port or `esbuild.exe`, run this from the
`agent/` folder:

```powershell
$project = (Get-Location).Path
$targets = @()
$targets += Get-NetTCPConnection -LocalPort 4321 -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess
$targets += Get-CimInstance Win32_Process |
  Where-Object {
    ($_.Name -eq 'node.exe' -or $_.Name -eq 'esbuild.exe') -and
    $_.CommandLine -like "*$project*"
  } |
  Select-Object -ExpandProperty ProcessId
$targets | Sort-Object -Unique | ForEach-Object {
  Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
}
if (Test-Path node_modules\.vite) {
  Remove-Item -LiteralPath node_modules\.vite -Recurse -Force
}
npm ci
npm run dev
```

## Build And Preview

```bash
npm run build
npm run preview
```

The static build is written to:

```text
dist/
```

## Project Structure

```text
src/
  components/        Reusable cards, navigation, comments, news
  content/           Markdown content collections
    blog/
    publications/
    talks/
    teaching/
    tutorials/
  layouts/           Base article and publication layouts
  pages/             Astro routes, including demos and teaching
  styles/            Global site CSS
public/
  images/
  gifs/
  videos/
  pdf/
  presentations/
```

## Add Content

Add a publication:

```text
src/content/publications/my-paper.md
```

Publication example:

```markdown
---
title: "Paper title"
authors: "Author One, Author Two"
venues:
  - "Conference or journal, year"
date: "2026-01-01"
image: "/images/publications/my-paper.png"
paperUrl: "https://arxiv.org/abs/..."
tags: ["Grokking", "Generalization"]
abstract: "Short summary for page metadata."
---

## Abstract

The full abstract goes here. Markdown emphasis and math, such as $x^2$, are supported.
```

The `abstract` frontmatter field is a short metadata summary. The Markdown body
contains the full abstract displayed on the publication page. Add further sections
for figures, results, or project details after the abstract as needed. Paper, code,
and project links belong in `paperUrl`, `codeUrl`, and `projectUrl` frontmatter;
avoid repeating them in the body. Distinct additional links can go in a `## Resources`
section.

The optional `image` should be a figure relevant to the paper. Omit it when no
figure is available; the card will show a styled topic and year instead.

Add blog posts and tutorials in:

```text
src/content/blog/
src/content/tutorials/
```

Use a quoted month-only date such as `date: "2025-01"` when only the month is known.
It displays as "January 2025" in the post and blog listing.

Math works directly in Markdown:

```latex
Inline: $x^2 + y^2 = 1$

Display:
$$
\nabla_\theta L(\theta)
$$
```

## Interactive Demos

Demos live in:

```text
src/pages/demos/
```

The current working example is:

```text
src/pages/demos/modular-addition.astro
```

It is a self-contained vanilla JavaScript demo. Future demos can follow the same pattern, or use a framework island only when the interaction becomes large enough to justify it.

## Giscus Comments

Comments are implemented through `src/components/Giscus.astro`.

Set these environment variables when GitHub Discussions is enabled:

```bash
PUBLIC_GISCUS_ENABLED=true
PUBLIC_GISCUS_REPO=Tikquuss/tikquuss.github.io
PUBLIC_GISCUS_REPO_ID=R_kgDOG2kYVg
PUBLIC_GISCUS_CATEGORY=General
PUBLIC_GISCUS_CATEGORY_ID=your_category_id
```

Use https://giscus.app to generate the category ID after enabling Discussions on the repository. Reactions are enabled and serve as the like system.

## GitHub Pages

The Astro config targets:

```text
https://tikquuss.github.io
```

Build output is fully static and can be deployed to GitHub Pages.

Deployment is handled by the repository-root workflow:

```text
../.github/workflows/deploy.yml
```

That workflow installs dependencies in `agent/`, builds this Astro project, uploads `agent/dist`, and deploys it with GitHub Pages. In the GitHub repository settings, set Pages source to `GitHub Actions`.
