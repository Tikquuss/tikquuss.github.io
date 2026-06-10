# tikquuss.github.io

Personal academic website for Pascal Jr. Tikeng Notsawo.

The active website is the Astro project in:

```text
agent/
```

The previous Jekyll/AcademicPages website has been archived in:

```text
legacy-jekyll/
```

## Run The New Site Locally

```powershell
cd agent
npm ci
npm run dev
```

Astro usually serves the site at:

```text
http://localhost:4321/
```

Stop the dev server with `Ctrl+C` before running `npm ci` again.

## Build Locally

```powershell
cd agent
npm run build
```

The generated static site is written to:

```text
agent/dist/
```

## GitHub Pages Deployment

Deployment is handled by the root workflow:

```text
.github/workflows/deploy.yml
```

On every push to `master` or `main`, GitHub Actions:

1. installs dependencies in `agent/`,
2. builds the Astro site,
3. uploads `agent/dist/`,
4. deploys it to GitHub Pages.

In the GitHub repository settings, use:

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

The site is configured for:

```text
https://tikquuss.github.io/
```

## Legacy Jekyll Archive

The old site source is preserved in `legacy-jekyll/` for reference. It is not used by GitHub Pages anymore.

To inspect or revive it later:

```powershell
cd legacy-jekyll
bundle install
bundle exec jekyll serve --livereload
```

The old generated `_site/` and `.jekyll-cache/` folders are archived locally but ignored by Git.

## Content Editing

For the new site, edit Markdown content in:

```text
agent/src/content/
```

Main collections:

```text
blog/
publications/
talks/
teaching/
tutorials/
```

Static assets for the new site live in:

```text
agent/public/
```
