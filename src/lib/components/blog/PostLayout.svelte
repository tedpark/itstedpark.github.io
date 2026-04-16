<!--
PostLayout — wraps every ``.md`` post body via mdsvex's ``layout`` config.

The post route (``routes/blog/[slug]/+page.svelte``) renders the markdown
component inside this layout, which provides the typography wrapper and
nothing else. Frontmatter (title, date, tags) is rendered by the route, not
here, so the route can put it above and the post body below.
-->
<script lang="ts">
	let { children } = $props();
</script>

<article class="prose-blog">
	{@render children()}
</article>
