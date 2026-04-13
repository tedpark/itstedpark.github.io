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

<div class="flex flex-col gap-14">

	<!-- ① Identity -->
	<div class="flex items-start gap-6">
		<span class="text-[5.5rem] font-bold text-foreground/[0.05] font-mono tabular-nums leading-none select-none flex-none -mt-2">
			{numberLabel}
		</span>
		<div class="flex flex-col gap-2 pt-1">
			<span class="text-[11px] font-mono text-muted-foreground tracking-[0.25em] uppercase">
				{project.period}
			</span>
			<h2 class="text-3xl md:text-[2.5rem] font-bold tracking-tight text-foreground leading-tight">
				{project.title}
			</h2>
			<p class="text-base text-foreground/55">{project.subtitle}</p>
		</div>
	</div>

	<!-- ② Screenshot -->
	<div class="flex flex-col gap-3">
		<button
			type="button"
			class="group w-full overflow-hidden rounded-2xl ring-1 ring-border/60 hover:ring-border/90 transition-all duration-300 cursor-zoom-in"
			onclick={() => openLightbox(activeIndex)}
		>
			<img
				src={project.screenshots[activeIndex].src}
				alt={project.screenshots[activeIndex].alt}
				class="w-full object-cover block group-hover:scale-[1.015] transition-transform duration-700 ease-out"
				loading={index === 0 ? 'eager' : 'lazy'}
			/>
		</button>

		{#if project.screenshots.length > 1}
			<div class="flex gap-2 overflow-x-auto py-0.5">
				{#each project.screenshots as shot, idx}
					<button
						type="button"
						class="flex-none w-16 h-10 rounded-lg overflow-hidden ring-1 transition-all duration-200 cursor-pointer {activeIndex === idx
							? 'ring-foreground/50 opacity-100 scale-105'
							: 'ring-border/30 opacity-35 hover:opacity-65 hover:ring-border/50'}"
						onclick={() => (activeIndex = idx)}
					>
						<img src={shot.src} alt={shot.alt} class="w-full h-full object-cover" />
					</button>
				{/each}
			</div>
		{/if}
	</div>

	<!-- ③ Metrics -->
	<div class="grid grid-cols-3 gap-px bg-border/40 rounded-xl overflow-hidden ring-1 ring-border/40">
		{#each project.metrics as metric}
			<div class="bg-card px-5 py-4 flex flex-col gap-1.5">
				<p class="text-[11px] font-mono text-muted-foreground uppercase tracking-widest">{metric.label}</p>
				<p class="text-2xl font-bold tabular-nums tracking-tight text-foreground">{metric.value}</p>
			</div>
		{/each}
	</div>

	<!-- ④ Description -->
	<p class="text-foreground/70 text-[17px] leading-[1.8] max-w-3xl">
		{project.description}
	</p>

	<!-- ⑤ Highlights -->
	<div class="grid sm:grid-cols-2 gap-x-10 gap-y-3.5">
		{#each project.highlights as h}
			<div class="flex gap-3.5 items-start">
				<span class="flex-none mt-[5px] w-1 h-1 rounded-full bg-foreground/30 ring-2 ring-foreground/10"></span>
				<span class="text-sm text-foreground/65 leading-relaxed">{h}</span>
			</div>
		{/each}
	</div>

	<!-- ⑥ Stack -->
	<div class="flex flex-wrap gap-1.5 pt-1 border-t border-border/40">
		{#each project.tags as tag}
			<span class="text-[11px] font-mono px-2.5 py-1 rounded-md bg-foreground/[0.05] text-foreground/50 hover:text-foreground/75 hover:bg-foreground/[0.08] transition-colors cursor-default">
				{tag}
			</span>
		{/each}
	</div>

</div>


<!-- Lightbox -->
{#if lightboxOpen}
	<div
		class="fixed inset-0 z-[200] bg-black/95 flex items-center justify-center"
		role="dialog"
		aria-modal="true"
		onclick={closeLightbox}
	>
		<!-- Top bar -->
		<div class="absolute top-0 inset-x-0 h-14 flex items-center justify-between px-6">
			<div class="flex items-center gap-3">
				<span class="text-xs font-mono text-white/25">{project.title}</span>
				<span class="text-white/15">·</span>
				<span class="text-xs font-mono text-white/25 tabular-nums">
					{lightboxIndex + 1} / {project.screenshots.length}
				</span>
			</div>
			<button
				type="button"
				class="text-[11px] font-mono text-white/30 hover:text-white/70 transition-colors tracking-widest uppercase"
				onclick={closeLightbox}
			>
				Close
			</button>
		</div>

		<!-- Image -->
		<div
			class="max-w-[90vw] max-h-[80vh]"
			onclick={(e) => e.stopPropagation()}
			role="presentation"
		>
			<img
				src={project.screenshots[lightboxIndex].src}
				alt={project.screenshots[lightboxIndex].alt}
				class="max-w-[90vw] max-h-[80vh] object-contain rounded-xl shadow-2xl"
			/>
		</div>

		<!-- Prev / Next -->
		{#if project.screenshots.length > 1}
			<button
				type="button"
				class="absolute left-5 top-1/2 -translate-y-1/2 w-11 h-11 flex items-center justify-center rounded-full bg-white/5 hover:bg-white/10 text-white/50 hover:text-white text-xl transition-all"
				onclick={(e) => { e.stopPropagation(); prev(); }}
				aria-label="Previous"
			>‹</button>
			<button
				type="button"
				class="absolute right-5 top-1/2 -translate-y-1/2 w-11 h-11 flex items-center justify-center rounded-full bg-white/5 hover:bg-white/10 text-white/50 hover:text-white text-xl transition-all"
				onclick={(e) => { e.stopPropagation(); next(); }}
				aria-label="Next"
			>›</button>
		{/if}

		<!-- Dot strip -->
		<div class="absolute bottom-6 left-1/2 -translate-x-1/2 flex gap-2">
			{#each project.screenshots as _, idx}
				<button
					type="button"
					class="transition-all duration-200 rounded-full {lightboxIndex === idx
						? 'w-4 h-1.5 bg-white/80'
						: 'w-1.5 h-1.5 bg-white/20 hover:bg-white/40'}"
					onclick={(e) => { e.stopPropagation(); lightboxIndex = idx; }}
					aria-label={`Screenshot ${idx + 1}`}
				></button>
			{/each}
		</div>
	</div>
{/if}
