"use client";

import { MetricCard, SectionCard, InsightCard } from "@/components/cards/Cards";
import { TradeoffChart } from "@/components/charts/TradeoffChart";
import { RankingTable } from "@/components/tables/RankingTable";
import { formatPercent } from "@/lib/utils";
import { AegisResult } from "@/lib/types";
import { Target, Scale, AlertTriangle, Sparkles, TrendingUp, Activity, Lightbulb, Info } from "lucide-react";

export function OverviewTab({ data }: { data: AegisResult }) {
  const { ranking } = data.model_analysis;
  const { bias_report, bias_tags } = data.dataset_analysis;
  const biasCount = Object.values(bias_tags).filter(Boolean).length;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Model Accuracy" value={formatPercent(ranking.ranking_table[0]?.accuracy || 0)} subtitle="Best performing model" icon={Target} trend="up" trendValue="+4.85%" variant="accent" />
        <MetricCard title="Fairness Score" value={ranking.ranking_table[0]?.demographic_parity_diff.toFixed(4) || "N/A"} subtitle="Demographic Parity Diff" icon={Scale} trend="down" trendValue="0.0000" variant="success" />
        <MetricCard title="Bias Detected" value={`${biasCount} types`} subtitle={`${bias_report.insights.length} total insights`} icon={AlertTriangle} variant="warning" />
        <MetricCard title="Optimal Strategy" value="Resampling" subtitle={`Tradeoff: ${ranking.best_score.toFixed(4)}`} icon={Sparkles} variant="default" />
      </div>

      <SectionCard title="AI-Generated Executive Summary" icon={Lightbulb} badge={data.explanations.gemini_used ? "Gemini" : "Template"} badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{data.explanations.summary}</p>
      </SectionCard>

      <div className="grid lg:grid-cols-2 gap-6">
        <SectionCard title="Strategy Performance Ranking" icon={TrendingUp} description="All evaluated mitigation strategies ordered by tradeoff score">
          <RankingTable data={ranking.ranking_table} />
        </SectionCard>
        <SectionCard title="Accuracy vs. Fairness Tradeoff" icon={Activity} description="Each point represents a mitigation strategy">
          <TradeoffChart data={ranking.ranking_table} />
        </SectionCard>
      </div>

      <SectionCard title="Key Bias Insights" icon={Info} badge={`${bias_report.insights.length}`}>
        <div className="space-y-2">
          {bias_report.insights.slice(0, 5).map((insight, i) => (
            <InsightCard key={i} insight={insight} index={i} />
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
