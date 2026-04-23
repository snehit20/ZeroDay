"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

interface ComparisonBarProps {
  data: Record<string, { before: number; after: number; delta: number }>;
  title?: string;
  labelBefore?: string;
  labelAfter?: string;
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

export function ComparisonBar({ data, title, labelBefore = "Before", labelAfter = "After" }: ComparisonBarProps) {
  const chartData = Object.entries(data).map(([name, vals]) => ({
    name,
    before: vals.before,
    after: vals.after,
  }));

  return (
    <div>
      {title && (
        <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider mb-3">
          {title}
        </p>
      )}
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(39,39,42,0.6)" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 10, fill: "#71717a" }}
            axisLine={false}
            tickLine={false}
            angle={-25}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#71717a" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(99,102,241,0.03)" }} />
          <Legend
            wrapperStyle={{ fontSize: "11px", color: "#a1a1aa" }}
            iconType="circle"
            iconSize={8}
          />
          <Bar dataKey="before" name={labelBefore} fill="#2563eb" radius={[3, 3, 0, 0]} maxBarSize={28} fillOpacity={0.5} />
          <Bar dataKey="after" name={labelAfter} fill="#10b981" radius={[3, 3, 0, 0]} maxBarSize={28} fillOpacity={0.8} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
