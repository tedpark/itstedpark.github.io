<script lang="ts">
	let { children } = $props();
</script>

<svelte:head>
	<title>Ted Park — Portfolio</title>
	<meta name="description" content="Quant Developer & AI Engineer Portfolio" />
</svelte:head>

<div class="min-h-screen bg-background text-foreground antialiased">
	{@render children()}
</div>

