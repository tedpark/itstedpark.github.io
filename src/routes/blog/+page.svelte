<script lang="ts">
	import { posts } from '$lib/data/posts';
	import SiteNav from '$lib/components/SiteNav.svelte';
</script>

<svelte:head>
	<title>Blog · Ted Park</title>
	<meta
		name="description"
		content="Notes on reinforcement learning, MLOps, and quantitative finance — from a 4-year solo build of a stat-arb trading system."
	/>
</svelte:head>

<div class="min-h-screen">
	<SiteNav label="Blog" />

	<!-- Header -->
	<section class="max-w-3xl mx-auto px-6 pt-40 pb-16">
		<p class="text-[11px] font-mono text-muted-foreground tracking-[0.3em] uppercase mb-6">
			Notes · Build Logs
		</p>
		<h1 class="text-4xl sm:text-5xl font-bold tracking-tight leading-[1.05] mb-6">
			Writing about<br />
			<span class="text-foreground/35">what I built.</span>
		</h1>
		<p class="text-foreground/60 text-lg leading-relaxed max-w-xl">
			Reinforcement learning, MLOps, and quantitative finance —
			grounded in a four-year solo build of a SAC pair-trading system.
			Every post links to runnable code and real benchmark numbers.
		</p>
	</section>

	<!-- Post list -->
	<section class="max-w-3xl mx-auto px-6 pb-24">
		{#if posts.length === 0}
			<p class="text-muted-foreground text-sm font-mono">
				No posts yet — the first one is on the way.
			</p>
		{:else}
			<ul class="divide-y divide-border/40">
				{#each posts as post}
					<li>
						<a
							href={`/blog/${post.slug}`}
							class="block py-7 group"
						>
							<div class="flex items-center gap-3 mb-2">
								<time
									class="text-[10px] font-mono text-muted-foreground/60 tracking-[0.15em] uppercase"
									datetime={post.date}
								>
									{post.date}
								</time>
								<span class="text-border/80">·</span>
								<span class="text-[10px] font-mono text-muted-foreground/60">
									{post.readingTime} min
								</span>
							</div>

							<h2
								class="text-2xl sm:text-3xl font-semibold tracking-tight leading-tight mb-2 group-hover:text-foreground/80 transition-colors"
							>
								{post.title}
							</h2>

							{#if post.subtitle}
								<p class="text-muted-foreground text-base mb-3">{post.subtitle}</p>
							{/if}

							<p class="text-foreground/55 text-sm leading-relaxed mb-4 max-w-2xl">
								{post.summary}
							</p>

							<div class="flex flex-wrap gap-2">
								{#each post.tags as tag}
									<span
										class="text-[10px] font-mono text-muted-foreground/70 px-2 py-0.5 border border-border/60 rounded"
									>{tag}</span>
								{/each}
							</div>
						</a>
					</li>
				{/each}
			</ul>
		{/if}
	</section>

	<!-- Footer -->
	<footer class="border-t border-border/50">
		<div class="max-w-5xl mx-auto px-6 py-8 flex items-center justify-between">
			<span class="text-[11px] text-muted-foreground font-mono">© 2026</span>
			<div class="flex items-center gap-6">
				<a
					href="mailto:itstedpark@gmail.com"
					class="text-[11px] text-muted-foreground hover:text-foreground transition-colors font-mono"
				>itstedpark@gmail.com</a>
				<a
					href="https://github.com/tedpark"
					target="_blank"
					rel="noopener noreferrer"
					class="text-[11px] text-muted-foreground hover:text-foreground transition-colors font-mono"
				>github.com/tedpark</a>
			</div>
		</div>
	</footer>
</div>
