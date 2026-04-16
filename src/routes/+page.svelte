<script lang="ts">
	import { projects } from '$lib/data/projects';
	import { posts } from '$lib/data/posts';
	import ProjectCard from '$lib/components/ProjectCard.svelte';
	import SiteNav from '$lib/components/SiteNav.svelte';

	// Featured post = latest. Rest = every other recent post (up to 5).
	const featured = posts[0];
	const recent = posts.slice(1, 6);
</script>

<div class="min-h-screen">

	<!-- Nav (shared with /blog) -->
	<SiteNav label="Portfolio" />

	<!-- Hero -->
	<section class="max-w-5xl mx-auto px-6 pt-40 pb-20">

		<p class="text-[11px] font-mono text-muted-foreground tracking-[0.3em] uppercase mb-6">
			ML / Quant Engineer · Writing &amp; Code
		</p>

		<h1 class="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.05] mb-6">
			Deep dives, then<br />
			<span class="text-foreground/35">the code behind them.</span>
		</h1>

		<p class="text-foreground/60 text-lg leading-relaxed max-w-xl mb-10">
			Writing on reinforcement learning, MLOps, and quantitative systems —
			each post backed by runnable code and real benchmark numbers from a
			four-year solo build of a SAC pair-trading system.
		</p>

		<!-- Stats strip -->
		<div class="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm font-mono text-muted-foreground">
			<span>{posts.length} Posts</span>
			<span class="text-border/80">·</span>
			<span>3 Projects</span>
			<span class="text-border/80">·</span>
			<span>Live on IBKR</span>
			<span class="text-border/80">·</span>
			<span>OOS Sharpe 3.716</span>
		</div>
	</section>

	<!-- ═══ 01 — Featured Writing ═══ -->
	{#if featured}
		<section>
			<div class="max-w-5xl mx-auto px-6">
				<div class="flex items-center gap-4 py-7 border-t border-white/[0.1]">
					<span class="text-[10px] font-mono text-muted-foreground/50 tracking-[0.35em] uppercase whitespace-nowrap">
						01 — Featured Writing
					</span>
					<div class="h-px flex-1 bg-white/[0.06]"></div>
					<a
						href="/blog"
						class="text-[10px] font-mono text-muted-foreground/70 hover:text-foreground transition-colors tracking-[0.2em] uppercase whitespace-nowrap"
					>All posts ↗</a>
				</div>
			</div>

			<!-- Featured (latest) — hero card treatment -->
			<div class="max-w-5xl mx-auto px-6 pb-12">
				<a
					href={`/blog/${featured.slug}`}
					class="block group rounded-xl border border-white/[0.08] hover:border-white/[0.18] bg-gradient-to-br from-white/[0.02] to-transparent transition-all p-8 md:p-12"
				>
					<div class="flex items-center gap-3 mb-5">
						<span class="text-[10px] font-mono text-foreground/80 px-2 py-0.5 border border-foreground/30 rounded tracking-[0.15em] uppercase">
							Latest
						</span>
						<time
							class="text-[10px] font-mono text-muted-foreground/70 tracking-[0.15em] uppercase"
							datetime={featured.date}
						>
							{featured.date}
						</time>
						<span class="text-border/80">·</span>
						<span class="text-[10px] font-mono text-muted-foreground/70">
							{featured.readingTime} min read
						</span>
					</div>

					<h2
						class="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight leading-[1.1] mb-4 group-hover:text-foreground/85 transition-colors"
					>
						{featured.title}
					</h2>

					{#if featured.subtitle}
						<p class="text-foreground/55 text-lg leading-relaxed mb-5 max-w-3xl">
							{featured.subtitle}
						</p>
					{/if}

					<p class="text-foreground/45 text-sm leading-relaxed mb-6 max-w-3xl">
						{featured.summary}
					</p>

					<div class="flex flex-wrap gap-2 mb-6">
						{#each featured.tags as tag}
							<span
								class="text-[10px] font-mono text-muted-foreground/80 px-2 py-0.5 border border-border/60 rounded"
							>{tag}</span>
						{/each}
					</div>

					<span class="text-[11px] font-mono text-foreground/70 group-hover:text-foreground transition-colors tracking-[0.15em] uppercase">
						Read post →
					</span>
				</a>
			</div>

			<!-- Recent posts list -->
			{#if recent.length > 0}
				<div class="max-w-5xl mx-auto px-6 pb-24 md:pb-32">
					<p class="text-[10px] font-mono text-muted-foreground/50 tracking-[0.35em] uppercase mb-6">
						More posts
					</p>
					<ul class="divide-y divide-border/40">
						{#each recent as post}
							<li>
								<a
									href={`/blog/${post.slug}`}
									class="block py-6 group"
								>
									<div class="flex items-center gap-3 mb-2">
										<time
											class="text-[10px] font-mono text-muted-foreground/60 tracking-[0.15em] uppercase"
											datetime={post.date}
										>{post.date}</time>
										<span class="text-border/80">·</span>
										<span class="text-[10px] font-mono text-muted-foreground/60">
											{post.readingTime} min
										</span>
									</div>
									<h3
										class="text-xl sm:text-2xl font-semibold tracking-tight leading-snug mb-2 group-hover:text-foreground/80 transition-colors"
									>{post.title}</h3>
									{#if post.subtitle}
										<p class="text-foreground/55 text-sm leading-relaxed max-w-3xl">
											{post.subtitle}
										</p>
									{/if}
								</a>
							</li>
						{/each}
					</ul>
				</div>
			{/if}
		</section>
	{/if}

	<!-- ═══ 02 — Projects ═══ -->
	{#each projects as project, i}
		<section>
			<!-- Labeled divider -->
			<div class="max-w-5xl mx-auto px-6">
				<div class="flex items-center gap-4 py-7 border-t border-white/[0.1]">
					<span class="text-[10px] font-mono text-muted-foreground/50 tracking-[0.35em] uppercase whitespace-nowrap">
						{['02 — Live System', '03 — Side Project', '04 — Side Project'][i]}
					</span>
					<div class="h-px flex-1 bg-white/[0.06]"></div>
				</div>
			</div>
			<div class="max-w-5xl mx-auto px-6 pb-24 md:pb-32">
				<ProjectCard {project} index={i} />
			</div>
		</section>
	{/each}

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
