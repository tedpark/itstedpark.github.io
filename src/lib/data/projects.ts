export type Screenshot = {
	src: string;
	alt: string;
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
};

export const projects: Project[] = [
	{
		id: 'stock-trading-ai',
		title: 'Stock Trading AI',
		subtitle: 'SAC RL 기반 StatArb 트레이딩 시스템',
		period: '2022 ~ 현재',
		description:
			'StatArb 전략 설계부터 AI 모델 학습, 백테스트 검증, 실매매 실행, 모니터링까지 전체 사이클을 혼자 구축·운용. 124,750개 조합에서 8단계 필터링으로 선별한 32개 페어를 IBKR로 실제 운용 중.',
		highlights: [
			'OOS Sharpe 3.716, Ann +71.5% (vs SPY +11.7%)',
			'HMM 레짐 분류 → 전략 분기로 레짐 전환 구간 성능 안정화',
			'SAC RL 포지션 사이징 — 엔트로피 최대화 목적함수로 탐색-착취 균형',
			'XGBoost + LightGBM + CatBoost 앙상블, Optuna 하이퍼파라미터 최적화',
			'FastAPI + DuckDB + MongoDB + Redis 멀티 DB 아키텍처',
			'IBKR ib-async 비동기 연동, Gateway Docker + autoheal 자동 복구',
			'Tauri 2 데스크톱 대시보드 + Rust TUI 모니터링 툴 직접 구축'
		],
		tags: [
			'Python', 'PyTorch', 'SAC RL', 'HMM', 'XGBoost', 'LightGBM', 'CatBoost',
			'TFT', 'Optuna', 'FastAPI', 'DuckDB', 'MongoDB', 'Redis',
			'Docker', 'IBKR', 'Tauri 2', 'Rust', 'Ratatui', 'SvelteKit'
		],
		screenshots: Array.from({ length: 11 }, (_, i) => ({
			src: `/screenshots/trading/trading-${String(i + 1).padStart(2, '0')}.png`,
			alt: `Stock Trading AI screenshot ${i + 1}`
		}))
	},
	{
		id: 'readbooks-ai',
		title: 'ReadBooks.ai',
		subtitle: 'LLM 기반 PDF 번역 데스크톱 앱',
		period: '2023',
		description:
			'영문 기술 서적을 문단별로 번역하는 Tauri 2 네이티브 앱. T5·fairseq로 직접 학습을 시도했다가 실패하고, Claude API 기반으로 재구현. 30개+ 언어 지원, SSE 스트리밍 Ask AI 기능 내장.',
		highlights: [
			'pdfjs-dist로 문단 단위 PDF 텍스트 추출',
			'Rust 백엔드에서 Claude Haiku API 호출 → 30개+ 언어 번역',
			'SSE 스트리밍으로 현재 페이지 컨텍스트 기반 Ask AI 구현',
			'실패 경험: T5·fairseq 직접 학습 → 한국어 데이터 부족으로 토큰 나열 → Claude API로 전환'
		],
		tags: ['Tauri 2', 'Rust', 'SvelteKit', 'TailwindCSS', 'Claude API', 'reqwest', 'tokio', 'pdfjs-dist'],
		screenshots: Array.from({ length: 5 }, (_, i) => ({
			src: `/screenshots/readbooks/readbooks-${String(i + 1).padStart(2, '0')}.png`,
			alt: `ReadBooks.ai screenshot ${i + 1}`
		}))
	},
	{
		id: 'mandai',
		title: 'Mandai',
		subtitle: 'AI 코치 내장 목표 관리 데스크톱 앱',
		period: '2024',
		description:
			'만다라트 + GTD + 뽀모도로를 결합한 목표 관리 데스크톱 앱. 3D 깊이감 있는 UI로 목표의 계층 구조를 시각화하고, 현재 목표 컨텍스트를 AI에 전달해 실행 가이드를 받을 수 있는 Ask AI 기능 내장.',
		highlights: [
			'Tauri 2 Rust 백엔드 — DuckDB 데이터 저장, 뽀모도로 상태 관리',
			'멀티 프로바이더: OpenAI · Anthropic · Gemini · Groq — 사용자가 직접 선택',
			'SSE 스트리밍으로 GTD 전문가 역할 LLM Ask AI 구현',
			'멀티 만다라트, GTD 상태 머신, 드릴다운 내비게이션'
		],
		tags: ['Tauri 2', 'Rust', 'SvelteKit', 'TailwindCSS', 'DuckDB', 'OpenAI', 'Claude', 'Gemini', 'Groq'],
		screenshots: Array.from({ length: 7 }, (_, i) => ({
			src: `/screenshots/mandai/mandai-${String(i + 1).padStart(2, '0')}.png`,
			alt: `Mandai screenshot ${i + 1}`
		}))
	}
];
