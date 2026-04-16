// Prerender every route at build time — required for static-adapter +
// GitHub Pages deployment. Individual routes can override (e.g. to false)
// for client-only behaviour, but our entire site is static.
export const prerender = true;
