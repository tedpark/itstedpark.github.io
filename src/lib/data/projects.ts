export type Screenshot = {
	src: string;
	alt: string;
};

export type Metric = {
	label: string;
	value: string;
};

export type Project = {
	id: string;
	title: string;
	subtitle: string;
	description: string;
	highlights: string[];
	tags: string[];
	screenshots: Screenshot[];
	github?: string;
	period: string;
	metrics: Metric[];
};

export const projects: Project[] = [
	{
		id: 'stock-trading-ai',
		title: 'Stock Trading AI',
		subtitle: 'SAC RL-based Statistical Arbitrage Trading System',
		period: '2022 – Present',
		description:
			'Built and operated the full cycle solo — from StatArb strategy design and AI model training, through backtesting, live execution, and real-time monitoring. Screened 124,750 pair combinations with an 8-stage filter pipeline and deployed the top 32 pairs live on IBKR.',
		highlights: [
			'OOS Sharpe 3.716 · Ann. Return +71.5% vs SPY +11.7%',
			'HMM regime classification → strategy branching for stable performance across regime transitions',
			'SAC RL position sizing — entropy-maximizing objective for exploration-exploitation balance',
			'XGBoost + LightGBM + CatBoost ensemble, Optuna hyperparameter optimization',
			'FastAPI + DuckDB + MongoDB + Redis multi-DB architecture',
			'IBKR ib-async integration, Gateway in Docker with autoheal auto-recovery',
			'Tauri 2 desktop dashboard + Rust TUI monitoring tool — both built in-house'
		],
		tags: [
			'Python', 'PyTorch', 'SAC RL', 'HMM', 'XGBoost', 'LightGBM', 'CatBoost',
			'TFT', 'Optuna', 'FastAPI', 'DuckDB', 'MongoDB', 'Redis',
			'Docker', 'IBKR', 'Tauri 2', 'Rust', 'Ratatui', 'SvelteKit'
		],
		metrics: [
			{ label: 'OOS Sharpe', value: '3.716' },
			{ label: 'Ann. Return', value: '+71.5%' },
			{ label: 'Live Pairs', value: '32' }
		],
		screenshots: Array.from({ length: 11 }, (_, i) => ({
			src: `/screenshots/trading/trading-${String(i + 1).padStart(2, '0')}.png`,
			alt: `Stock Trading AI screenshot ${i + 1}`
		}))
	},
	{
		id: 'readbooks-ai',
		title: 'ReadBooks.ai',
		subtitle: 'LLM-powered PDF Translation Desktop App',
		period: '2023',
		description:
			'A Tauri 2 native app that translates technical books paragraph by paragraph. Attempted direct fine-tuning with T5 and fairseq — abandoned after Korean data shortage caused token noise — then rebuilt on Claude API. Ships with 30+ language support and an SSE-streamed Ask AI panel.',
		highlights: [
			'Paragraph-level PDF text extraction via pdfjs-dist',
			'Rust backend calls Claude Haiku API → 30+ language output',
			'SSE streaming for page-context-aware Ask AI',
			'Honest failure: T5/fairseq fine-tune attempt failed → insufficient Korean data → pivoted to Claude API'
		],
		tags: ['Tauri 2', 'Rust', 'SvelteKit', 'TailwindCSS', 'Claude API', 'reqwest', 'tokio', 'pdfjs-dist'],
		metrics: [
			{ label: 'Languages', value: '30+' },
			{ label: 'LLM Backend', value: 'Claude' },
			{ label: 'Platform', value: 'Native' }
		],
		screenshots: Array.from({ length: 5 }, (_, i) => ({
			src: `/screenshots/readbooks/readbooks-${String(i + 1).padStart(2, '0')}.png`,
			alt: `ReadBooks.ai screenshot ${i + 1}`
		}))
	},
	{
		id: 'mandai',
		title: 'Mandai',
		subtitle: 'Goal Management Desktop App with Built-in AI Coach',
		period: '2024',
		description:
			'A Tauri 2 desktop app that unifies Mandala Chart, GTD, and Pomodoro into a single workflow. Visualizes goal hierarchy with a 3D depth UI and delivers execution guidance through an Ask AI feature that injects the current goal context into the prompt.',
		highlights: [
			'Tauri 2 Rust backend — DuckDB persistence, Pomodoro state management',
			'Multi-provider: OpenAI · Anthropic · Gemini · Groq — user-selectable at runtime',
			'SSE-streamed Ask AI with a GTD expert system prompt',
			'Multi Mandala Chart support, GTD state machine, drill-down navigation'
		],
		tags: ['Tauri 2', 'Rust', 'SvelteKit', 'TailwindCSS', 'DuckDB', 'OpenAI', 'Claude', 'Gemini', 'Groq'],
		metrics: [
			{ label: 'AI Providers', value: '4' },
			{ label: 'Methodology', value: '3-in-1' },
			{ label: 'Storage', value: 'Local-first' }
		],
		screenshots: Array.from({ length: 7 }, (_, i) => ({
			src: `/screenshots/mandai/mandai-${String(i + 1).padStart(2, '0')}.png`,
			alt: `Mandai screenshot ${i + 1}`
		}))
	}
];
