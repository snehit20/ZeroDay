"use client";

import { SectionCard, InsightCard } from "@/components/cards/Cards";
import { DistributionChart } from "@/components/charts/DistributionChart";
import { BiasReport } from "@/lib/types";
import { cn, formatPercent } from "@/lib/utils";
import { Users, BarChart3, Scale, AlertTriangle, Info } from "lucide-react";

export function DetectionTab({ biasReport }: { biasReport: BiasReport }) {
  const sensitiveFeatures = Object.keys(biasReport.distribution_bias);

  return (
    <div className="space-y-6">
      {/* Distribution charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        {sensitiveFeatures.map((feat) => (
          <SectionCard key={feat} title={`Group Distribution: ${feat}`} icon={Users} badge={`Ratio: ${biasReport.distribution_bias[feat].imbalance_ratio.toFixed(1)}`} badgeVariant={biasReport.distribution_bias[feat].imbalance_ratio > 5 ? "danger" : "warning"}>
            <DistributionChart data={biasReport.distribution_bias[feat].group_proportions} />
          </SectionCard>
        ))}
      </div>

      {/* Outcome disparity */}
      <div className="grid lg:grid-cols-2 gap-6">
        {sensitiveFeatures.map((feat) => (
          <SectionCard key={feat} title={`Outcome Disparity: ${feat}`} icon={BarChart3} badge={`Gap: ${formatPercent(biasReport.outcome_bias[feat]?.disparity || 0)}`} badgeVariant={(biasReport.outcome_bias[feat]?.disparity || 0) > 0.15 ? "danger" : "warning"}>
            <DistributionChart data={biasReport.outcome_bias[feat]?.outcome_rates || {}} color="#f59e0b" />
          </SectionCard>
        ))}
      </div>

      {/* Fairness table */}
      <SectionCard title="Fairness Compliance Metrics" icon={Scale} description="Demographic parity and disparate impact assessment per protected attribute">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)]">
                {["Protected Attribute", "DP Difference", "Disparate Impact", "Status"].map((h, i) => (
                  <th key={h} className={cn("py-3 px-3 text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)]", i === 0 ? "text-left" : "text-right")}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(biasReport.fairness_metrics).map(([feat, m]) => (
                <tr key={feat} className="border-b border-[var(--color-border-subtle)] hover:bg-[var(--color-surface-hover)] transition-colors">
                  <td className="py-3 px-3 font-medium capitalize">{feat.replace("_", " ")}</td>
                  <td className="py-3 px-3 text-right font-mono text-xs"><span className={m.demographic_parity_difference > 0.1 ? "text-[var(--color-danger)]" : "text-[var(--color-success)]"}>{m.demographic_parity_difference.toFixed(4)}</span></td>
                  <td className="py-3 px-3 text-right font-mono text-xs text-[var(--color-text-secondary)]">{m.disparate_impact_ratio.toFixed(4)}</td>
                  <td className="py-3 px-3 text-right"><span className={cn("px-2.5 py-0.5 rounded-full text-[10px] font-bold", m.demographic_parity_difference > 0.15 ? "bg-[var(--color-danger-muted)] text-[var(--color-danger)]" : m.demographic_parity_difference > 0.05 ? "bg-[var(--color-warning-muted)] text-[var(--color-warning)]" : "bg-[var(--color-success-muted)] text-[var(--color-success)]")}>{m.demographic_parity_difference > 0.15 ? "UNFAIR" : m.demographic_parity_difference > 0.05 ? "AT RISK" : "COMPLIANT"}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Advanced bias */}
      <div className="grid lg:grid-cols-2 gap-6">
        <SectionCard title="Proxy Variable Detection" icon={AlertTriangle} badge="Detected" badgeVariant="danger">
          {Object.entries(biasReport.advanced_bias.proxy_bias).map(([feature, info]) => (
            <div key={feature} className="flex items-center justify-between p-4 rounded-lg bg-[var(--color-danger-muted)] border border-[var(--color-danger)]/20">
              <div>
                <p className="text-sm font-semibold">&apos;{feature}&apos; identified as proxy variable</p>
                <p className="text-xs text-[var(--color-text-muted)] mt-0.5">Statistically correlates with &apos;{info.correlated_with}&apos;</p>
              </div>
              <span className="font-mono text-sm text-[var(--color-danger)] font-bold">r = {info.correlation.toFixed(3)}</span>
            </div>
          ))}
        </SectionCard>
        <SectionCard title="Intersectional Bias Analysis" icon={Users} badge={`Peak: ${formatPercent(biasReport.advanced_bias.intersectional_bias.max_disparity || 0)}`} badgeVariant="danger">
          <div className="space-y-2">
            {Object.entries(biasReport.advanced_bias.intersectional_bias.group_rates || {}).map(([group, rate]) => (
              <div key={group} className="flex items-center justify-between py-1.5 text-xs">
                <span className="text-[var(--color-text-secondary)]">{group}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 rounded-full bg-[var(--color-surface-elevated)] overflow-hidden">
                    <div className="h-full rounded-full bg-[var(--color-accent)]" style={{ width: `${rate * 100}%` }} />
                  </div>
                  <span className="font-mono text-[var(--color-text-muted)] w-12 text-right">{formatPercent(rate)}</span>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      </div>

      <SectionCard title="Complete Insight Log" icon={Info}>
        <div className="space-y-2">
          {biasReport.insights.map((insight, i) => <InsightCard key={i} insight={insight} index={i} />)}
        </div>
      </SectionCard>
    </div>
  );
}
