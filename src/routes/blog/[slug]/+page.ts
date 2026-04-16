import { error } from '@sveltejs/kit';
import { posts, getPost } from '$lib/data/posts';
import type { Component } from 'svelte';
import type { PageLoad } from './$types';

export const prerender = true;

export const entries = () =>
	posts.map((post) => ({ slug: post.slug }));

export const load: PageLoad = async ({ params }) => {
	const post = getPost(params.slug);
	if (!post) throw error(404, 'Post not found');

	// Dynamic import of the matching markdown component (mdsvex compiles each
	// .md to a Svelte component at build time).
	let Component: Component;
	try {
		const mod = await import(`../../../lib/data/posts/${params.slug}.md`);
		Component = mod.default;
	} catch (_err) {
		throw error(404, 'Post body missing');
	}

	return { post, Component };
};
