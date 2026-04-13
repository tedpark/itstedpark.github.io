<script lang="ts">
	import { projects } from '$lib/data/projects';
	import ProjectCard from '$lib/components/ProjectCard.svelte';
</script>

<div class="min-h-screen">
	<!-- Nav -->
	<nav class="fixed top-0 inset-x-0 z-50 border-b border-border/40 bg-background/80 backdrop-blur-md">
		<div class="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
			<span class="text-xs text-muted-foreground font-mono tracking-wider">portfolio</span>
			<a href="https://github.com/tedpark" target="_blank" rel="noopener noreferrer"
				class="text-xs text-muted-foreground hover:text-foreground transition-colors font-mono"
			>github ↗</a>
		</div>
	</nav>

	<!-- Hero -->
	<section class="max-w-5xl mx-auto px-6 pt-36 pb-24">
		<div class="flex flex-col gap-5">
			<p class="text-xs text-muted-foreground font-mono tracking-[0.25em] uppercase">2022 – Present</p>
			<h1 class="text-5xl md:text-[4.5rem] font-semibold tracking-tight leading-[1.05]">
				AI systems,<br />
				<span class="text-muted-foreground">end-to-end.</span>
			</h1>
			<p class="text-muted-foreground text-base md:text-lg max-w-lg leading-relaxed">
				전략 설계 → 모델 학습 → 백테스트 → 실매매 → 모니터링.<br />
				아이디어에서 프로덕션까지 전 과정을 혼자 구축·운용합니다.
			</p>
			<div class="flex flex-wrap gap-2 pt-2">
				{#each ['Python', 'PyTorch', 'SAC RL', 'FastAPI', 'Tauri 2', 'Rust', 'SvelteKit', 'Docker', 'IBKR'] as tech}
					<span class="text-xs font-mono px-2.5 py-1 rounded-full border border-border/60 text-muted-foreground">
						{tech}
					</span>
				{/each}
			</div>
		</div>
	</section>

	<!-- Projects -->
	{#each projects as project, i}
		<section class="border-t border-border/60">
			<div class="max-w-5xl mx-auto px-6 py-16 md:py-24">
				<ProjectCard {project} index={i} />
			</div>
		</section>
	{/each}

	<!-- Footer -->
	<footer class="border-t border-border/60">
		<div class="max-w-5xl mx-auto px-6 py-8 flex items-center justify-between">
			<span class="text-xs text-muted-foreground font-mono">© 2024</span>
			<a href="https://github.com/tedpark" target="_blank" rel="noopener noreferrer"
				class="text-xs text-muted-foreground hover:text-foreground transition-colors font-mono"
			>github.com/tedpark</a>
		</div>
	</footer>
</div>
