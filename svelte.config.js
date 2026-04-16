import adapter from '@sveltejs/adapter-static';
import { mdsvex } from 'mdsvex';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const _projectRoot = dirname(fileURLToPath(import.meta.url));

/** @type {import('@sveltejs/kit').Config} */
const config = {
	extensions: ['.svelte', '.svx', '.md'],
	preprocess: [
		mdsvex({
			extensions: ['.svx', '.md'],
			// Posts get wrapped in our blog post layout (typography, code blocks).
			// The layout path must be an absolute fs path or a node-resolvable
			// import — relative paths like 'src/lib/...' get treated as packages.
			layout: {
				_: resolve(_projectRoot, 'src/lib/components/blog/PostLayout.svelte')
			},
			// Disable smartypants — it converts straight quotes (and other ASCII
			// punctuation) which then breaks code samples and `$math$` clashes
			// with Svelte 5 runes. We render math via Unicode + code blocks
			// instead of KaTeX for now, which avoids the whole class of issues.
			smartypants: false
		})
	],
	compilerOptions: {
		// Runes mode for our own .svelte files, but not for mdsvex output
		// (.md / .svx) which still uses legacy $$props in its generated wrapper.
		runes: ({ filename }) => {
			const parts = filename.split(/[/\\]/);
			if (parts.includes('node_modules')) return undefined;
			if (filename.endsWith('.md') || filename.endsWith('.svx')) return false;
			return true;
		}
	},
	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: '404.html',
			precompress: false,
			strict: true
		})
	}
};

export default config;
