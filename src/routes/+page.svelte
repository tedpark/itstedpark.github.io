<script lang="ts">
	import { projects } from '$lib/data/projects';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { Separator } from '$lib/components/ui/separator/index.js';
	import ProjectCard from '$lib/components/ProjectCard.svelte';
</script>

<!-- Hero -->
<section class="border-b">
	<div class="mx-auto max-w-5xl px-6 py-20">
		<p class="text-muted-foreground mb-3 text-sm font-medium tracking-widest uppercase">Portfolio</p>
		<h1 class="mb-4 text-4xl font-bold tracking-tight sm:text-5xl">박승환 · Ted Park</h1>
		<p class="text-muted-foreground mb-6 max-w-2xl text-lg leading-relaxed">
			StatArb 전략 설계부터 AI 모델 학습, 실매매 실행, 모니터링까지 —<br />
			전체 사이클을 혼자 구축하는 Quant Developer / AI Engineer.
		</p>
		<div class="flex flex-wrap gap-2">
			{#each ['Python', 'PyTorch', 'SAC RL', 'FastAPI', 'Tauri 2 (Rust)', 'SvelteKit', 'IBKR'] as tag}
				<Badge variant="secondary">{tag}</Badge>
			{/each}
		</div>
		<div class="text-muted-foreground mt-8 flex gap-6 text-sm">
			<a href="https://github.com/itstedpark" target="_blank" rel="noopener noreferrer" class="hover:text-foreground transition-colors">GitHub →</a>
			<a href="https://medium.com/@itstedpark" target="_blank" rel="noopener noreferrer" class="hover:text-foreground transition-colors">Medium →</a>
			<a href="mailto:itstedpark@gmail.com" class="hover:text-foreground transition-colors">itstedpark@gmail.com</a>
		</div>
	</div>
</section>

<!-- Performance highlight -->
<section class="border-b bg-muted/30">
	<div class="mx-auto max-w-5xl px-6 py-10">
		<p class="text-muted-foreground mb-6 text-xs font-medium tracking-widest uppercase">Live Trading Performance · IBKR (OOS 기준)</p>
		<div class="grid grid-cols-2 gap-6 sm:grid-cols-4">
			{#each [
				{ label: 'OOS Sharpe', value: '3.716', sub: 'HMM MTF 포트 v9' },
				{ label: 'Ann. Return', value: '+71.5%', sub: 'vs SPY +11.7%' },
				{ label: 'Max Drawdown', value: '-6.8%', sub: 'vs SPY -9.8%' },
				{ label: 'Cum. PnL', value: '+$34,192', sub: 'StatArb · WR 68.6%' }
			] as stat}
				<div>
					<div class="text-2xl font-bold tabular-nums">{stat.value}</div>
					<div class="text-sm font-medium">{stat.label}</div>
					<div class="text-muted-foreground text-xs">{stat.sub}</div>
				</div>
			{/each}
		</div>
	</div>
</section>

<!-- Projects -->
<section class="mx-auto max-w-5xl px-6 py-16">
	<h2 class="mb-10 text-2xl font-bold">Projects</h2>
	<div class="space-y-6">
		{#each projects as project}
			<ProjectCard {project} />
		{/each}
	</div>
</section>

<!-- Footer -->
<footer class="border-t">
	<div class="text-muted-foreground mx-auto max-w-5xl px-6 py-8 text-sm">
		<Separator class="mb-6" />
		<div class="flex flex-col gap-1 sm:flex-row sm:justify-between">
			<span>박승환 · itstedpark@gmail.com · 010-2969-3764</span>
			<span>독학학위제 컴퓨터과학 학사 · 정보처리기사</span>
		</div>
	</div>
</footer>

