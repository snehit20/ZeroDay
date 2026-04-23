"use client";

import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { ComparisonRow } from "@/lib/types";

interface TradeoffChartProps {
  data: ComparisonRow[];
  title?: string;
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-[var(--color-surface-elevated)] border border-[var(--color-border)] rounded-lg px-3 py-2 shadow-xl">
      <p className="text-xs font-semibold text-[var(--color-text-primary)] mb-1">{d.pipeline}</p>
      <p className="text-xs text-[var(--color-text-secondary)]">
        Accuracy: {(d.accuracy * 100).toFixed(1)}%
      </p>
      <p className="text-xs text-[var(--color-text-secondary)]">
        DP Diff: {d.demographic_parity_diff.toFixed(4)}
      </p>
      <p className="text-xs text-[var(--color-accent)]">
        Score: {d.score.toFixed(4)}
      </p>
    </div>
  );
};

export function TradeoffChart({ data, title }: TradeoffChartProps) {
  const chartData = data.map((row) => ({
    ...row,
    x: row.accuracy,
    y: row.demographic_parity_diff,
  }));

  return (
    <div>
      {title && (
        <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider mb-3">
          {title}
        </p>
      )}
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(39,39,42,0.6)" />
          <XAxis
            dataKey="x"
            name="Accuracy"
            tick={{ fontSize: 10, fill: "#71717a" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            label={{ value: "Accuracy", position: "bottom", fontSize: 11, fill: "#71717a", offset: -5 }}
          />
          <YAxis
            dataKey="y"
            name="DP Diff"
            tick={{ fontSize: 10, fill: "#71717a" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => v.toFixed(2)}
            label={{ value: "Bias (DP Diff)", angle: -90, position: "insideLeft", fontSize: 11, fill: "#71717a", offset: 15 }}
          />
          <ReferenceLine y={0.05} stroke="#22c55e" strokeDasharray="4 4" strokeOpacity={0.5} />
          <Tooltip content={<CustomTooltip />} />
          <Scatter data={chartData} fill="#2563eb">
            {chartData.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.rank === 1 ? "#10b981" : entry.strategy === "baseline" ? "#ef4444" : "#2563eb"}
                fillOpacity={entry.rank === 1 ? 1 : 0.6}
                r={entry.rank === 1 ? 8 : 5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-6 mt-2">
        <div className="flex items-center gap-1.5 text-[10px] text-[var(--color-text-muted)]">
          <div className="w-2.5 h-2.5 rounded-full bg-[#10b981]" /> Best Strategy
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-[var(--color-text-muted)]">
          <div className="w-2.5 h-2.5 rounded-full bg-[#ef4444]" /> Baseline
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-[var(--color-text-muted)]">
          <div className="w-2.5 h-2.5 rounded-full bg-[#2563eb]" /> Other
        </div>
      </div>
    </div>
  );
}
