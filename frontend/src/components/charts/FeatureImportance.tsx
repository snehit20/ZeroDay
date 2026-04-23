"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

interface FeatureImportanceProps {
  baseline: Record<string, number>;
  best: Record<string, number>;
  title?: string;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[var(--color-surface-elevated)] border border-[var(--color-border)] rounded-lg px-3 py-2 shadow-xl">
      <p className="text-xs font-semibold text-[var(--color-text-primary)] mb-1">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-xs" style={{ color: p.color }}>
          {p.name}: {(p.value * 100).toFixed(1)}%
        </p>
      ))}
    </div>
  );
};

export function FeatureImportance({ baseline, best, title }: FeatureImportanceProps) {
  const allFeatures = new Set([...Object.keys(baseline), ...Object.keys(best)]);
  const chartData = Array.from(allFeatures)
    .map((name) => ({
      name,
      baseline: baseline[name] || 0,
      mitigated: best[name] || 0,
    }))
    .sort((a, b) => b.mitigated - a.mitigated)
    .slice(0, 10);

  return (
    <div>
      {title && (
        <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider mb-3">
          {title}
        </p>
      )}
      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 4, right: 20, bottom: 4, left: 10 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(39,39,42,0.6)" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 10, fill: "#71717a" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 11, fill: "#a1a1aa" }}
            axisLine={false}
            tickLine={false}
            width={110}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(99,102,241,0.03)" }} />
          <Bar
            dataKey="baseline"
            name="Baseline"
            fill="#2563eb"
            fillOpacity={0.3}
            radius={[0, 3, 3, 0]}
            barSize={12}
          />
          <Bar
            dataKey="mitigated"
            name="Mitigated"
            fill="#10b981"
            fillOpacity={0.8}
            radius={[0, 3, 3, 0]}
            barSize={12}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
