<script lang="ts">
	import type { Project } from '$lib/data/projects';

	let { project, index }: { project: Project; index: number } = $props();

	let activeIndex = $state(0);
	let lightboxOpen = $state(false);
	let lightboxIndex = $state(0);

	function openLightbox(idx: number) {
		lightboxIndex = idx;
		lightboxOpen = true;
	}

	function closeLightbox() {
		lightboxOpen = false;
	}

	function prev() {
		lightboxIndex = (lightboxIndex - 1 + project.screenshots.length) % project.screenshots.length;
	}

	function next() {
		lightboxIndex = (lightboxIndex + 1) % project.screenshots.length;
	}

	function handleKey(e: KeyboardEvent) {
		if (!lightboxOpen) return;
		if (e.key === 'Escape') closeLightbox();
		if (e.key === 'ArrowLeft') prev();
		if (e.key === 'ArrowRight') next();
	}

	const numberLabel = ['01', '02', '03'][index] ?? String(index + 1).padStart(2, '0');
</script>

<svelte:window onkeydown={handleKey} />

<div class="flex flex-col gap-10">
	<!-- Header row -->
	<div class="flex items-start justify-between">
		<div class="flex flex-col gap-1">
			<span class="text-xs font-mono text-muted-foreground tracking-widest uppercase">
				{project.period}
			</span>
			<h2 class="text-3xl md:text-4xl font-semibold tracking-tight">{project.title}</h2>
			<p class="text-muted-foreground text-base">{project.subtitle}</p>
		</div>
		<span class="text-4xl font-semibold text-muted-foreground/20 font-mono tabular-nums select-none">
			{numberLabel}
		</span>
	</div>

	<!-- Main screenshot + thumbnails -->
	<div class="flex flex-col gap-3">
		<button
			type="button"
			class="group w-full overflow-hidden rounded-xl ring-1 ring-border/60 hover:ring-border transition-all cursor-zoom-in"
			onclick={() => openLightbox(activeIndex)}
		>
			<img
				src={project.screenshots[activeIndex].src}
				alt={project.screenshots[activeIndex].alt}
				class="w-full object-cover block group-hover:scale-[1.01] transition-transform duration-500"
				loading={index === 0 ? 'eager' : 'lazy'}
			/>
		</button>

		{#if project.screenshots.length > 1}
			<div class="flex gap-2 overflow-x-auto pb-0.5 scroll-smooth">
				{#each project.screenshots as shot, idx}
					<button
						type="button"
						class="flex-none w-14 h-9 rounded-md overflow-hidden ring-1 transition-all cursor-pointer {activeIndex === idx
							? 'ring-foreground/60 opacity-100'
							: 'ring-border/40 opacity-40 hover:opacity-70'}"
						onclick={() => (activeIndex = idx)}
					>
						<img src={shot.src} alt={shot.alt} class="w-full h-full object-cover" />
					</button>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Metrics + content -->
	<div class="grid md:grid-cols-5 gap-8 md:gap-12">
		<!-- Left: highlights -->
		<div class="md:col-span-3 flex flex-col gap-4">
			{#if project.id === 'stock-trading-ai'}
				<div class="grid grid-cols-3 gap-3">
					<div class="rounded-lg border border-border/60 bg-card p-3">
						<p class="text-xs font-mono text-muted-foreground mb-1">OOS Sharpe</p>
						<p class="text-xl font-semibold tabular-nums">3.716</p>
					</div>
					<div class="rounded-lg border border-border/60 bg-card p-3">
						<p class="text-xs font-mono text-muted-foreground mb-1">Ann. Return</p>
						<p class="text-xl font-semibold tabular-nums">+71.5%</p>
					</div>
					<div class="rounded-lg border border-border/60 bg-card p-3">
						<p class="text-xs font-mono text-muted-foreground mb-1">SPY vs</p>
						<p class="text-xl font-semibold tabular-nums">+11.7%</p>
					</div>
				</div>
			{/if}

			<ul class="flex flex-col gap-2.5">
				{#each project.highlights as h}
					<li class="flex gap-3 text-sm leading-relaxed">
						<span class="text-border mt-1 flex-none">—</span>
						<span class="text-muted-foreground">{h}</span>
					</li>
				{/each}
			</ul>
		</div>

		<!-- Right: description + tags -->
		<div class="md:col-span-2 flex flex-col gap-5">
			<p class="text-sm text-muted-foreground leading-relaxed">{project.description}</p>
			<div class="flex flex-wrap gap-1.5">
				{#each project.tags as tag}
					<span
						class="text-xs font-mono px-2 py-0.5 rounded-md border border-border/50 text-muted-foreground"
					>
						{tag}
					</span>
				{/each}
			</div>
		</div>
	</div>
</div>

<!-- Lightbox -->
{#if lightboxOpen}
	<div
		class="fixed inset-0 z-[200] bg-black/96 flex items-center justify-center"
		role="dialog"
		aria-modal="true"
		onclick={closeLightbox}
	>
		<!-- Top bar -->
		<div class="absolute top-0 inset-x-0 h-12 flex items-center justify-between px-5 pointer-events-none">
			<span class="text-xs font-mono text-white/30 pointer-events-auto">
				{lightboxIndex + 1} / {project.screenshots.length}
			</span>
			<button
				type="button"
				class="text-xs font-mono text-white/40 hover:text-white/80 transition-colors pointer-events-auto"
				onclick={closeLightbox}
			>
				ESC
			</button>
		</div>

		<!-- Image (stop propagation so clicking image doesn't close) -->
		<div
			class="max-w-[88vw] max-h-[82vh]"
			onclick={(e) => e.stopPropagation()}
			role="presentation"
		>
			<img
				src={project.screenshots[lightboxIndex].src}
				alt={project.screenshots[lightboxIndex].alt}
				class="max-w-[88vw] max-h-[82vh] object-contain rounded-lg shadow-2xl"
			/>
		</div>

		<!-- Prev / Next -->
		{#if project.screenshots.length > 1}
			<button
				type="button"
				class="absolute left-4 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-all"
				onclick={(e) => { e.stopPropagation(); prev(); }}
			>
				‹
			</button>
			<button
				type="button"
				class="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-all"
				onclick={(e) => { e.stopPropagation(); next(); }}
			>
				›
			</button>
		{/if}

		<!-- Thumbnail strip -->
		<div class="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-1.5">
			{#each project.screenshots as _, idx}
				<button
					type="button"
					class="w-1.5 h-1.5 rounded-full transition-all {lightboxIndex === idx
						? 'bg-white/80'
						: 'bg-white/20 hover:bg-white/40'}"
					onclick={(e) => { e.stopPropagation(); lightboxIndex = idx; }}
				></button>
			{/each}
		</div>
	</div>
{/if}
