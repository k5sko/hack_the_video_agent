# frontend/ — P1

**Owner:** Person 1

**Responsibility:** the single-page Beeline UI — query box, YouTube player with
hard clip boundaries and auto-advance, force-directed concept graph, "I already
know…" checklist, the `2h 20m → 12m` counter, and the Graph-path / Search-only
kill-shot toggle.

**Expected input:** a `PathResult` matching `shared/types.ts`. During development
this comes from a local JSON file (`src/data/*.json`, seeded from
`shared/sample_path.json`) — no backend required. P4 later swaps the source for
`POST /api/path`.

**Expected output:** a running Vite dev server on `http://localhost:5173` where
the full demo flow works: type a concept → graph lights up → press play → tick
"I already know" → path shrinks → flip to Search only → answer degrades.

**How to run:**

```bash
cd frontend
npm install
npm run dev
```

**Constraints:** React + TypeScript + Vite, `react-force-graph`, YouTube IFrame
Player API. Desktop-first, one page. Blue/green/gray/red are reserved
exclusively for node states. No backend, no auth, no extra screens.
