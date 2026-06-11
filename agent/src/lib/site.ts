export const SITE = {
  name: 'Pascal Jr. Tikeng Notsawo',
  shortName: 'Pascal Tikeng',
  title: 'Pascal Jr. Tikeng Notsawo',
  description:
    'Personal research website for Pascal Jr. Tikeng Notsawo, a PhD student working on machine learning theory, grokking, generalization, optimization, and alignment.',
  url: 'https://tikquuss.github.io',
  repo: 'Tikquuss/tikquuss.github.io',
  repoId: 'R_kgDOG2kYVg',
  authorEmail: 'pascal.tikeng.notsawo@mila.quebec',
  cvUrl: 'https://drive.google.com/file/d/1nyCLwQo6L4_XYxhiFiNOIEhPNYZtPAbO/view?usp=sharing',
  scholarUrl: 'https://scholar.google.com/citations?user=vUerGI8AAAAJ&hl=en',
  githubUrl: 'https://github.com/Tikquuss',
  twitterUrl: 'https://twitter.com/tikquuss',
};

const rawBase = import.meta.env.BASE_URL ?? '/';
export const basePath = rawBase === '/' ? '' : rawBase.replace(/\/$/, '');

export function withBase(path: string) {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${basePath}${normalized}` || '/';
}

export function resolveUrl(url: string) {
  if (/^(https?:|mailto:|#)/.test(url)) return url;
  return withBase(url);
}

export function formatDate(date: string, options: Intl.DateTimeFormatOptions = {}) {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    ...options,
  });
}

export function formatMonthYear(date: string) {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
  });
}
