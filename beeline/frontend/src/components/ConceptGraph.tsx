import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type LinkObject, type NodeObject } from "react-force-graph-2d";
import type { NodeState } from "@shared/types";
import { buildGraphData, type ConceptNode } from "../lib/graph";

/**
 * Blue / green / gray / red are reserved for node states and are used for
 * nothing else anywhere in this app.
 */
export const NODE_COLORS: Record<NodeState, string> = {
  on_path: "#2563EB",
  known: "#8FC4A8",
  not_needed: "#BEBEC4",
  gap: "#D93025",
};

const LABEL_COLORS: Record<NodeState, string> = {
  on_path: "#12305F",
  known: "#4C7A61",
  not_needed: "#94949A",
  gap: "#8F1D14",
};

interface Props {
  states: Map<string, NodeState>;
  /** Concepts covered by the clip playing right now — these pulse. */
  currentConcepts: string[];
  /** Concepts already watched in this path — these get a check. */
  completedConcepts: string[];
  selected: string | null;
  onSelect: (conceptId: string) => void;
  /**
   * Show only the concepts that bear on this query, rather than all 100. The
   * full corpus is the honest picture but it reads as homework; what a learner
   * needs to see is their own route and the handful of things around it.
   */
  focus: boolean;
}

export default function ConceptGraph({
  states,
  currentConcepts,
  completedConcepts,
  selected,
  onSelect,
  focus,
}: Props) {
  // Everything relevant to this query: the path itself, what was pruned as
  // known, the gaps, and one hop of context so the path has something to sit in.
  const relevantKey = [...states.entries()]
    .filter(([, s]) => s !== "not_needed")
    .map(([id]) => id)
    .sort()
    .join("|");

  // react-force-graph mutates these nodes with x/y, so the object identity has
  // to stay stable or the layout is thrown away. It only changes when the set of
  // shown nodes actually changes.
  const graphData = useMemo(() => {
    const full = buildGraphData();
    if (!focus) return full;

    const core = new Set(relevantKey ? relevantKey.split("|") : []);
    if (core.size === 0) return full;

    const keep = new Set(core);
    for (const node of full.nodes) {
      if (!core.has(node.id)) continue;
      for (const neighbour of [...node.assumes, ...node.requiredBy]) {
        keep.add(neighbour);
      }
    }
    const nodes = full.nodes.filter((n) => keep.has(n.id));
    const ids = new Set(nodes.map((n) => n.id));
    const links = full.links.filter((l) => {
      const source = typeof l.source === "string" ? l.source : l.source.id;
      const target = typeof l.target === "string" ? l.target : l.target.id;
      return ids.has(source) && ids.has(target);
    });
    return { nodes, links };
  }, [focus, relevantKey]);

  const wrapRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- imperative kapsule handle
  const fgRef = useRef<any>(undefined);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [, setFrame] = useState(0);

  const currentKey = currentConcepts.join("|");
  const current = useMemo(
    () => new Set(currentKey ? currentKey.split("|") : []),
    [currentKey],
  );
  const completed = useMemo(
    () => new Set(completedConcepts),
    [completedConcepts],
  );

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Spread 38 concepts out enough that every label is legible on a projector.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || size.width === 0) return;
    fg.d3Force("charge")?.strength(-300).distanceMax(600);
    fg.d3Force("link")?.distance(46).strength(0.45);
    fg.d3ReheatSimulation?.();
    (window as unknown as Record<string, unknown>).__fg = fg;
    const t = window.setTimeout(() => fgRef.current?.zoomToFit(400, 46), 4200);
    return () => window.clearTimeout(t);
  }, [size.width, size.height, graphData]);

  // Drives the pulse. Only runs while a clip is actually playing.
  useEffect(() => {
    if (current.size === 0) return;
    const id = window.setInterval(() => setFrame((n) => (n + 1) % 1000), 80);
    return () => window.clearInterval(id);
  }, [current]);

  const stateOf = (id: string): NodeState => states.get(id) ?? "not_needed";

  const paintNode = (
    raw: NodeObject,
    ctx: CanvasRenderingContext2D,
    scale: number,
  ) => {
    const node = raw as unknown as ConceptNode;
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    const state = stateOf(node.id);
    const emphasised = state === "on_path" || state === "gap";
    const r = emphasised ? 5.5 : 3.8;

    if (current.has(node.id)) {
      const phase = (performance.now() % 1500) / 1500;
      ctx.beginPath();
      ctx.arc(x, y, r + 2 + phase * 13, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(37, 99, 235, ${(1 - phase) * 0.55})`;
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }

    if (selected === node.id) {
      ctx.beginPath();
      ctx.arc(x, y, r + 4.5, 0, Math.PI * 2);
      ctx.strokeStyle = "#1A1A1C";
      ctx.lineWidth = 1.1;
      ctx.stroke();
    }

    if (state === "gap") {
      // Hollow ring: required, but nothing in the corpus fills it.
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = "#FFFFFF";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = NODE_COLORS.gap;
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = NODE_COLORS[state];
      ctx.globalAlpha = state === "known" ? 0.75 : 1;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    if (state === "on_path" && completed.has(node.id)) {
      ctx.beginPath();
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 1.3;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.moveTo(x - 2.4, y - 0.1);
      ctx.lineTo(x - 0.8, y + 1.8);
      ctx.lineTo(x + 2.6, y - 2.2);
      ctx.stroke();
    }

    // Label everything that carries meaning right now. The rest of the corpus
    // stays as texture until you zoom in or pick it — 38 labels in a 400px
    // column is unreadable, and the point is that the path is legible.
    const labelled =
      state !== "not_needed" || selected === node.id || scale > 1.9;
    if (!labelled) return;

    const fontSize = Math.min(12, 11 / scale);
    ctx.font = `${emphasised ? 600 : 400} ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = LABEL_COLORS[state];
    ctx.fillText(node.id, x, y + r + 2.5);
  };

  const paintPointerArea = (
    raw: NodeObject,
    color: string,
    ctx: CanvasRenderingContext2D,
  ) => {
    const node = raw as unknown as ConceptNode;
    ctx.beginPath();
    ctx.arc(node.x ?? 0, node.y ?? 0, 9, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  };

  const endpointId = (end: LinkObject["source"]): string =>
    typeof end === "object" && end !== null ? String(end.id) : String(end);

  const linkStyle = (link: LinkObject) => {
    const a = stateOf(endpointId(link.source));
    const b = stateOf(endpointId(link.target));
    if (a === "gap" || b === "gap") return { color: "rgba(217,48,37,0.45)", width: 1.1 };
    if (a === "on_path" && b === "on_path")
      return { color: "rgba(37,99,235,0.45)", width: 1.3 };
    if (a === "on_path" && b === "known")
      return { color: "rgba(143,196,168,0.6)", width: 1 };
    return { color: "rgba(26,26,28,0.09)", width: 0.5 };
  };

  return (
    <div className="graph-canvas" ref={wrapRef}>
      {size.width > 0 && size.height > 0 && (
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          width={size.width}
          height={size.height}
          backgroundColor="#FFFFFF"
          nodeCanvasObject={paintNode}
          nodePointerAreaPaint={paintPointerArea}
          nodeLabel={(n: NodeObject) => String(n.id)}
          linkColor={(l: LinkObject) => linkStyle(l).color}
          linkWidth={(l: LinkObject) => linkStyle(l).width}
          linkDirectionalArrowLength={2.6}
          linkDirectionalArrowRelPos={1}
          onNodeClick={(n: NodeObject) => onSelect(String(n.id))}
          cooldownTime={4000}
          d3VelocityDecay={0.32}
          onEngineStop={() => fgRef.current?.zoomToFit(500, 36)}
        />
      )}
    </div>
  );
}
