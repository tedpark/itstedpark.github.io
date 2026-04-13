<script lang="ts">
	import type { Project } from '$lib/data/projects';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { Card, CardContent, CardHeader } from '$lib/components/ui/card/index.js';

	let { project }: { project: Project } = $props();

	let currentIndex = $state(0);
	let lightboxOpen = $state(false);
	let lightboxIndex = $state(0);

	function prev() {
		currentIndex = (currentIndex - 1 + project.screenshots.length) % project.screenshots.length;
	}

	function next() {
		currentIndex = (currentIndex + 1) % project.screenshots.length;
	}

	function openLightbox(i: number) {
		lightboxIndex = i;
		lightboxOpen = true;
	}

	function closeLightbox() {
		lightboxOpen = false;
	}

	function lightboxPrev() {
		lightboxIndex = (lightboxIndex - 1 + project.screenshots.length) % project.screenshots.length;
	}

	function lightboxNext() {
		lightboxIndex = (lightboxIndex + 1) % project.screenshots.length;
	}

	function onKeydown(e: KeyboardEvent) {
		if (!lightboxOpen) return;
		if (e.key === 'Escape') closeLightbox();
		if (e.key === 'ArrowLeft') lightboxPrev();
		if (e.key === 'ArrowRight') lightboxNext();
	}
</script>

<svelte:window onkeydown={onKeydown} />

<Card class="overflow-hidden">
	<div class="grid gap-0 lg:grid-cols-[1fr_400px]">
		<!-- Screenshot carousel -->
		<div class="relative bg-zinc-950 aspect-[16/10] lg:aspect-auto min-h-[240px]">
			{#each project.screenshots as ss, i}
				<button
					class="absolute inset-0 h-full w-full cursor-zoom-in transition-opacity duration-300"
					class:opacity-100={i === currentIndex}
					class:opacity-0={i !== currentIndex}
					onclick={() => openLightbox(i)}
					tabindex={i === currentIndex ? 0 : -1}
					aria-label="확대해서 보기"
				>
					<img
						src={ss.src}
						alt={ss.alt}
						class="h-full w-full object-cover object-top"
						loading="lazy"
					/>
				</button>
			{/each}

			<!-- Nav arrows -->
			{#if project.screenshots.length > 1}
				<button
					onclick={(e) => { e.stopPropagation(); prev(); }}
					class="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-black/50 p-1.5 text-white backdrop-blur-sm hover:bg-black/70 transition-colors"
					aria-label="이전"
				>
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
						<path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7" />
					</svg>
				</button>
				<button
					onclick={(e) => { e.stopPropagation(); next(); }}
					class="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-black/50 p-1.5 text-white backdrop-blur-sm hover:bg-black/70 transition-colors"
					aria-label="다음"
				>
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
						<path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
					</svg>
				</button>

				<!-- Dots -->
				<div class="absolute bottom-2 left-1/2 flex -translate-x-1/2 gap-1">
					{#each project.screenshots as _, i}
						<button
							onclick={(e) => { e.stopPropagation(); currentIndex = i; }}
							class="h-1.5 rounded-full transition-all"
							class:w-4={i === currentIndex}
							class:bg-white={i === currentIndex}
							class:w-1.5={i !== currentIndex}
							class:bg-white/40={i !== currentIndex}
							aria-label={`스크린샷 ${i + 1}`}
						/>
					{/each}
				</div>
			{/if}

			<!-- Counter -->
			<div class="absolute right-2 top-2 rounded bg-black/50 px-2 py-0.5 text-xs text-white/70 backdrop-blur-sm tabular-nums">
				{currentIndex + 1} / {project.screenshots.length}
			</div>
		</div>

		<!-- Content -->
		<div class="flex flex-col p-6">
			<CardHeader class="p-0 mb-4">
				<div class="flex items-start justify-between gap-2">
					<div>
						<p class="text-muted-foreground mb-0.5 text-xs">{project.period}</p>
						<h3 class="text-xl font-bold leading-tight">{project.title}</h3>
						<p class="text-muted-foreground text-sm">{project.subtitle}</p>
					</div>
				</div>
			</CardHeader>

			<CardContent class="p-0 flex-1 flex flex-col gap-4">
				<p class="text-sm leading-relaxed">{project.description}</p>

				<ul class="space-y-1.5">
					{#each project.highlights as h}
						<li class="flex gap-2 text-sm">
							<span class="text-muted-foreground mt-0.5 shrink-0">·</span>
							<span>{h}</span>
						</li>
					{/each}
				</ul>

				<div class="mt-auto flex flex-wrap gap-1.5 pt-2">
					{#each project.tags as tag}
						<Badge variant="outline" class="text-xs">{tag}</Badge>
					{/each}
				</div>
			</CardContent>
		</div>
	</div>
</Card>

<!-- Lightbox -->
{#if lightboxOpen}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
		onclick={closeLightbox}
		role="dialog"
		aria-modal="true"
		aria-label="스크린샷 확대 보기"
	>
		<!-- Close -->
		<button
			onclick={closeLightbox}
			class="absolute right-4 top-4 rounded-full bg-white/10 p-2 text-white hover:bg-white/20 transition-colors"
			aria-label="닫기"
		>
			<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
			</svg>
		</button>

		<!-- Image -->
		<div
			class="relative mx-4 max-h-[90vh] max-w-6xl w-full"
			onclick={(e) => e.stopPropagation()}
			role="presentation"
		>
			<img
				src={project.screenshots[lightboxIndex].src}
				alt={project.screenshots[lightboxIndex].alt}
				class="mx-auto max-h-[85vh] w-auto rounded-lg object-contain shadow-2xl"
			/>
			<p class="mt-2 text-center text-sm text-white/50 tabular-nums">
				{lightboxIndex + 1} / {project.screenshots.length}
			</p>
		</div>

		<!-- Prev/Next -->
		{#if project.screenshots.length > 1}
			<button
				onclick={(e) => { e.stopPropagation(); lightboxPrev(); }}
				class="absolute left-4 top-1/2 -translate-y-1/2 rounded-full bg-white/10 p-3 text-white hover:bg-white/20 transition-colors"
				aria-label="이전"
			>
				<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
					<path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7" />
				</svg>
			</button>
			<button
				onclick={(e) => { e.stopPropagation(); lightboxNext(); }}
				class="absolute right-4 top-1/2 -translate-y-1/2 rounded-full bg-white/10 p-3 text-white hover:bg-white/20 transition-colors"
				aria-label="다음"
			>
				<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
					<path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
				</svg>
			</button>
		{/if}
	</div>
{/if}
