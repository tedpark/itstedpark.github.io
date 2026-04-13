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
		subtitle: 'Statistical Arbitrage System · SAC RL · Live on IBKR',
		period: '2022 – Present',
		description:
			'A full-stack algorithmic trading system designed, built, and operated solo — from strategy research through live execution on IBKR. The central problem with most pair trading systems is regime dependency: strategies calibrated in trending markets collapse during mean-reverting periods. This system addresses it by combining HMM-based regime classification with SAC reinforcement learning for adaptive position sizing — the HMM detects market state in real time and routes signals to the appropriate strategy branch, while the SAC agent learns an entropy-maximizing policy that naturally shrinks exposure when signal confidence is low. The data pipeline ingests from FMP, FRED, yfinance, and Alpha Vantage into a DuckDB-backed feature store, normalizes and engineers features across multiple timeframes, then drives an XGBoost / LightGBM / CatBoost ensemble plus a PyTorch TFT model for signal generation. Live orders are submitted to IBKR via ib-async. A Rust TUI and a Tauri 2 desktop dashboard provide real-time visibility into signals, P&L, and position state.',
		highlights: [
			'OOS Sharpe 3.716 · Ann. Return +71.5% vs SPY benchmark +11.7%',
			'HMM regime classifier feeds a strategy router — different signal logic per regime state',
			'SAC RL position sizing: entropy-maximizing objective balances exploration vs exploitation',
			'XGBoost + LightGBM + CatBoost ensemble with Optuna sweep; PyTorch TFT for sequence prediction',
			'Data pipeline: FMP + FRED + yfinance + Alpha Vantage → DuckDB feature store → multi-timeframe feature engineering → model training & inference',
			'FastAPI service layer: MongoDB (Beanie ODM) for trade records, Redis for cache, DuckDB for analytical queries — each layer purpose-fit',
			'Live execution on IBKR — 32 active stat-arb pairs, real-time order management and P&L tracking via ib_async',
			'Rust TUI (Ratatui + Tokio) for terminal monitoring; Tauri 2 + SvelteKit desktop dashboard'
		],
		tags: [
			'Python', 'PyTorch', 'SAC RL', 'HMM', 'XGBoost', 'LightGBM', 'CatBoost',
			'TFT', 'Optuna', 'FastAPI', 'DuckDB', 'MongoDB', 'Redis',
			'FMP API', 'Docker', 'IBKR', 'Tauri 2', 'Rust', 'Ratatui', 'SvelteKit'
		],
		metrics: [
			{ label: 'OOS Sharpe', value: '3.716' },
			{ label: 'Ann. Return', value: '+71.5%' },
			{ label: 'IBKR Live Pairs', value: '32' }
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
			'A Tauri 2 native desktop app for reading English technical books in your native language, paragraph by paragraph. The original approach attempted to fine-tune T5 and fairseq models directly on Korean–English pairs — this was abandoned when the Korean training corpus proved too thin, producing token-level noise instead of coherent output. The architecture was rebuilt around Claude Haiku via the Anthropic API, with the Rust backend handling all network I/O through reqwest, while pdfjs-dist on the SvelteKit frontend extracts and segments paragraph-level text blocks from PDF files. An SSE-streamed Ask AI panel lets users ask questions mid-read, injecting the current page text as context so answers are always relevant to what\'s on screen.',
		highlights: [
			'pdfjs-dist parses PDF structure and extracts text at paragraph granularity',
			'Rust backend (reqwest + tokio) calls Claude Haiku API with async concurrency',
			'30+ language support — translation target is user-configurable at runtime',
			'SSE streaming delivers Ask AI responses token-by-token for low-latency feel',
			'Failure-driven pivot: T5/fairseq fine-tune failed → insufficient Korean data → Claude API',
			'Fully offline-capable except for API calls; no server, no account required beyond API key'
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
			'A Tauri 2 desktop app that replaces three separate productivity tools — Mandala Chart, GTD, and Pomodoro — with one coherent workflow. The Mandala Chart gives goals a spatial structure: each outer cell expands into its own 3×3 action plan, with a drill-down navigator that moves through hierarchy levels. GTD state management runs as an explicit state machine in Rust, tracking items across Inbox → Next Actions → Waiting → Done with enforced transitions. Pomodoro sessions drive the focus cycle and write session data to DuckDB through the Rust backend, keeping everything local-first with no cloud dependency. An Ask AI feature injects the current goal and its context into a configurable LLM prompt — supporting OpenAI, Anthropic, Gemini, and Groq — and streams the coaching response back via SSE.',
		highlights: [
			'3-in-1 workflow: Mandala Chart spatial hierarchy + GTD state machine + Pomodoro timer',
			'Rust state machine enforces GTD transitions — no invalid state changes possible',
			'Drill-down navigation: click any cell to expand its own 3×3 Mandala sub-plan',
			'DuckDB via Rust backend — all data stays local, zero cloud dependency',
			'Multi-provider AI: OpenAI · Anthropic · Gemini · Groq — user-selectable at runtime',
			'SSE-streamed Ask AI with goal context injection and GTD expert system prompt'
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
