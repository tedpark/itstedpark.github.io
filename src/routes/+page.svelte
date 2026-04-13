<script lang="ts">
	import { projects } from '$lib/data/projects';
	import ProjectCard from '$lib/components/ProjectCard.svelte';
</script>

<div class="min-h-screen">

	<!-- Nav -->
	<nav class="fixed top-0 inset-x-0 z-50 border-b border-border/50 bg-background/85 backdrop-blur-md">
		<div class="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
			<span class="text-[11px] text-muted-foreground font-mono uppercase tracking-[0.2em]">Portfolio</span>
			<a
				href="https://github.com/tedpark"
				target="_blank"
				rel="noopener noreferrer"
				class="text-[11px] text-muted-foreground hover:text-foreground transition-colors font-mono"
			>github.com/tedpark ↗</a>
		</div>
	</nav>

	<!-- Hero -->
	<section class="max-w-5xl mx-auto px-6 pt-40 pb-20">

		<p class="text-[11px] font-mono text-muted-foreground tracking-[0.3em] uppercase mb-6">
			AI Engineer · Quant Developer
		</p>

		<h1 class="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.05] mb-6">
			Production-grade AI,<br />
			<span class="text-foreground/35">end-to-end.</span>
		</h1>

		<p class="text-foreground/60 text-lg leading-relaxed max-w-xl mb-10">
			전략 설계 → 모델 학습 → 백테스트 → 실매매 → 모니터링.<br />
			아이디어에서 프로덕션까지, 전 과정을 혼자 구축·운용합니다.
		</p>

		<!-- Stats strip -->
		<div class="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm font-mono text-muted-foreground">
			<span>3 Projects</span>
			<span class="text-border/80">·</span>
			<span>Live on IBKR</span>
			<span class="text-border/80">·</span>
			<span>OOS Sharpe 3.716</span>
			<span class="text-border/80">·</span>
			<span>Python · Rust · SvelteKit</span>
		</div>
	</section>

	<!-- Projects -->
	{#each projects as project, i}
		<section class="border-t border-border/50">
			<div class="max-w-5xl mx-auto px-6 py-20 md:py-28">
				<ProjectCard {project} index={i} />
			</div>
		</section>
	{/each}

	<!-- Footer -->
	<footer class="border-t border-border/50">
		<div class="max-w-5xl mx-auto px-6 py-8 flex items-center justify-between">
			<span class="text-[11px] text-muted-foreground font-mono">© 2026</span>
			<a
				href="https://github.com/tedpark"
				target="_blank"
				rel="noopener noreferrer"
				class="text-[11px] text-muted-foreground hover:text-foreground transition-colors font-mono"
			>github.com/tedpark</a>
		</div>
	</footer>

</div>
