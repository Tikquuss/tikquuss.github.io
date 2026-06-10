## Project Goal

Build a **new personal academic website** inside the folder:

```text
agent/
```

**Do not modify any other part of the repository.**

The existing website is based on Jekyll and is considered legacy. It should be used only as a source of content and inspiration, not as an architectural reference.

The objective is to create a modern, maintainable, highly customizable personal research website.

---

# Context

The website belongs to a PhD student in Machine Learning Theory.

Research interests include:

* Machine Learning Theory
* Generalization
* Grokking
* Optimization
* Representation Learning
* AI Safety / Alignment
* LLM Unlearning
* Mathematics
* Physics

The website is both:

1. A personal website
2. A research communication platform

It should serve as:

* Academic homepage
* Publication archive
* Research blog
* Teaching page
* Tutorial repository
* Talk repository
* Interactive research demo platform

---

# Technology Choice

The preferred stack is:

```text
Astro
```

with:

```text
Markdown / MDX
KaTeX (or MathJax)
Giscus
GitHub Pages
```

### Reasons

The previous Jekyll website caused recurring issues:

* fragile local development
* Math rendering failures
* difficult customization
* outdated architecture
* poor support for interactive demos

Astro provides:

* static-site generation
* GitHub Pages deployment
* modern developer experience
* custom design freedom
* excellent support for interactive JavaScript demos
* Markdown-based content authoring

---

# High-Level Requirements

The website must support:

## 1. GitHub Pages Deployment

The final website must be deployable on:

```text
https://<username>.github.io
```

No backend.

No database.

Static hosting only.

---

## 2. LaTeX Support

Blog posts, tutorials, and project pages must support:

Inline math:

```latex
$x^2+y^2=1$
```

Display math:

```latex
$$
\nabla_\theta L(\theta)
$$
```

The user should not need to implement any custom LaTeX rendering system.

---

## 3. Comments and Likes

Use:

```text
Giscus
```

Requirements:

* comments on blog posts
* comments on tutorial pages
* GitHub-based authentication
* reactions enabled

GitHub reactions can be used as the "like" system.

No custom backend should be created.

---

## 4. Multimedia Support

Must support:

* images
* GIFs
* videos
* embedded YouTube videos
* embedded presentations

The content author should be able to insert them easily into Markdown/MDX.

---

## 5. Interactive Research Demos

A major requirement.

The website must support interactive demos for papers.

Examples:

* grokking visualization
* modular arithmetic demos
* neural network visualizations
* embedding visualizations
* optimization dynamics
* mathematical simulations

Demos should be implemented as small self-contained frontend applications.

No backend required.

Prefer:

* vanilla JavaScript
* TypeScript
* React only when necessary

---

## 6. Fully Custom Design

The website must **not** look like a standard academic template.

Avoid:

* AcademicPages
* Minimal Mistakes look
* Generic Bootstrap academic themes

The design should feel:

* personal
* research-oriented
* modern
* mathematical
* minimal

The owner has previous experience with:

* HTML
* CSS
* Bootstrap
* Angular

Therefore customization is expected.

---

# Website Structure

---

## Home Page

Primary landing page.

Must include:

* photo
* name
* affiliation
* short biography
* research interests
* links
* recent news

Example content:

```text
Photo

Pascal Junior Tikeng Notsawo

PhD Student
Université de Montréal & Mila

Research interests:
- Machine Learning Theory
- Grokking
- Generalization
- AI Alignment
- LLM Unlearning
```

Links:

* CV
* Google Scholar
* GitHub
* Publications
* Blog
* Teaching
* Talks

Recent news:

```text
[2026] New paper accepted at ICML.
[2026] Attending Deep Learning Indaba.
```

---

## About Page

Contains:

* longer biography
* academic background
* research interests
* news updates

This page may evolve over time.

---

## Publications Page

A core feature.

For each publication:

Required:

```text
Title
Authors
Venue
Year
Thumbnail image
```

Clicking a publication must open a dedicated project page.

---

## Individual Publication Page

Must support:

* abstract
* images
* videos
* presentations
* links
* blog posts
* demos
* supplementary material

Think of this page as:

```text
mini-project website
```

for a research paper.

---

## Blog

Markdown/MDX based.

Requirements:

* categories
* tags
* math
* images
* videos
* comments

The author should be able to write long mathematical posts easily.

---

## Tutorials

Similar to blog posts.

May contain:

* math
* code
* figures
* videos

Examples:

* Machine Learning Theory
* Probability
* Linear Algebra
* Deep Learning

---

## Talks

Repository of talks.

Each talk page may contain:

* title
* event
* date
* slides
* video recording
* resources

---

## Demos

Dedicated page listing all interactive demos.

Example:

```text
Demos
 ├── Grokking on Modular Addition
 ├── Representation Learning Visualizer
 ├── Optimization Dynamics Playground
 └── Neural Network Geometry
```

---

# Proposed Astro Structure

```text
agent/
├── src/
│   ├── layouts/
│   │   ├── BaseLayout.astro
│   │   └── PostLayout.astro
│   │
│   ├── components/
│   │   ├── Navbar.astro
│   │   ├── Footer.astro
│   │   ├── PublicationCard.astro
│   │   └── NewsSection.astro
│   │
│   ├── content/
│   │   ├── blog/
│   │   ├── tutorials/
│   │   ├── publications/
│   │   └── talks/
│   │
│   └── pages/
│       ├── index.astro
│       ├── about.astro
│       ├── publications.astro
│       ├── teaching.astro
│       ├── talks.astro
│       ├── blog/
│       ├── tutorials/
│       └── demos/
│
├── public/
│   ├── images/
│   ├── gifs/
│   ├── videos/
│   ├── pdf/
│   └── presentations/
│
├── astro.config.mjs
├── package.json
└── README.md
```

---

# Design Principles

Prioritize:

* simplicity
* readability
* maintainability
* long-term extensibility

Avoid:

* over-engineering
* backend services
* custom CMS
* databases
* unnecessary frameworks

The owner should be able to:

1. Write a Markdown file
2. Push to GitHub
3. Have the website update automatically

---

# Migration Philosophy

Do **not** attempt to reproduce the existing website exactly.

The current website is considered messy and difficult to maintain.

Instead:

1. Extract useful content.
2. Reorganize it cleanly.
3. Build a modern architecture from scratch.

---

# Success Criteria

The project is successful if:

* GitHub Pages deployment works
* LaTeX works reliably
* Blog writing is easy
* Publication pages are attractive
* Interactive demos are easy to add
* Comments work
* The design feels personal and not template-based
* Future content additions require minimal effort