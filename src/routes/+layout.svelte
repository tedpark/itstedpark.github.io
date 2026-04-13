<script lang="ts">
	import '../app.css';
	let { children } = $props();
</script>

<svelte:head>
	<title>Portfolio</title>
	<meta name="description" content="AI Engineer & Quant Developer — building production systems end-to-end" />
	<meta name="theme-color" content="#0a0a0a" />
</svelte:head>

<div class="min-h-screen bg-background text-foreground antialiased font-sans">
	{@render children()}
</div>

