"use client";

import { SectionCard } from "@/components/cards/Cards";
import { TradeoffChart } from "@/components/charts/TradeoffChart";
import { RankingTable } from "@/components/tables/RankingTable";
import { Sparkles, Lightbulb, TrendingUp, Activity, Scale } from "lucide-react";

interface MitigationTabProps {
  ranking: { best_strategy: string; best_score: number; ranking_table: any[] };
  explanations: { strategy_justification: string; tradeoff_analysis: string };
}

export function MitigationTab({ ranking, explanations }: MitigationTabProps) {
  return (
    <div className="space-y-6">
      {/* Winner banner */}
      <div className="p-6 rounded-xl border border-[var(--color-success)]/30 bg-[var(--color-success-muted)] glow-success">
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-[var(--color-success)] flex items-center justify-center shrink-0">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1">
            <p className="text-[10px] text-[var(--color-success)] font-bold uppercase tracking-widest">Optimal Mitigation Strategy</p>
            <h3 className="text-lg font-bold capitalize mt-0.5">{ranking.best_strategy.replace("_", " ")}</h3>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-[var(--color-success)]">{ranking.best_score.toFixed(4)}</p>
            <p className="text-[10px] text-[var(--color-text-muted)] font-medium">Composite Score</p>
          </div>
        </div>
      </div>

      <SectionCard title="Strategy Selection Rationale" icon={Lightbulb} badge="Gemini" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{explanations.strategy_justification}</p>
      </SectionCard>

      <div className="grid lg:grid-cols-2 gap-6">
        <SectionCard title="Complete Strategy Ranking" icon={TrendingUp} description="All strategies ranked by accuracy-fairness composite">
          <RankingTable data={ranking.ranking_table} />
        </SectionCard>
        <SectionCard title="Tradeoff Scatter Plot" icon={Activity} description="Accuracy (x) vs. bias (y) — lower-right is optimal">
          <TradeoffChart data={ranking.ranking_table} />
        </SectionCard>
      </div>

      <SectionCard title="Accuracy-Fairness Tradeoff Assessment" icon={Scale} badge="AI Analysis" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{explanations.tradeoff_analysis}</p>
      </SectionCard>
    </div>
  );
}
