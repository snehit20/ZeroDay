"use client";

import { MetricCard, SectionCard } from "@/components/cards/Cards";
import { FeatureImportance } from "@/components/charts/FeatureImportance";
import { formatPercent } from "@/lib/utils";
import { Target, Activity, BarChart3, Brain, Wrench, Lightbulb, Sparkles } from "lucide-react";

interface ExplainabilityTabProps {
  explainability: any;
  explanations: any;
}

export function ExplainabilityTab({ explainability, explanations }: ExplainabilityTabProps) {
  const preds = explainability.predictions_analysis;

  return (
    <div className="space-y-6">
      {/* Prediction metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Prediction Agreement" value={formatPercent(preds.prediction_agreement || 0)} subtitle="Alignment with baseline" icon={Target} variant="accent" />
        <MetricCard title="Predictions Changed" value={preds.predictions_changed?.toLocaleString() || "0"} subtitle={`of ${preds.baseline?.total.toLocaleString() || 0} total`} icon={Activity} variant="warning" />
        <MetricCard title="Baseline Positive Rate" value={formatPercent(preds.baseline?.positive_rate || 0)} subtitle={`${preds.baseline?.positive_count.toLocaleString()} positive`} icon={BarChart3} />
        <MetricCard title="Mitigated Positive Rate" value={formatPercent(preds.best?.positive_rate || 0)} subtitle={`${preds.best?.positive_count.toLocaleString()} positive`} icon={BarChart3} variant="success" />
      </div>

      {/* Feature importance */}
      <SectionCard title="Feature Importance Comparison" icon={BarChart3} description="Side-by-side feature weight analysis: baseline model vs. mitigated model">
        <FeatureImportance baseline={explainability.feature_importance.baseline} best={explainability.feature_importance.best} />
      </SectionCard>

      {/* AI explanations */}
      <SectionCard title="Root Cause Analysis" icon={Brain} badge="Gemini" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{explanations.bias_explanation}</p>
      </SectionCard>

      <SectionCard title="Mitigation Methodology" icon={Wrench} badge="Gemini" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{explanations.strategy_justification}</p>
      </SectionCard>

      <SectionCard title="Model Behavioral Analysis" icon={Lightbulb} badge="AI Generated" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{explainability.explanation}</p>
      </SectionCard>

      {/* Recommendation */}
      <div className="p-5 rounded-xl border border-[var(--color-accent)]/30 bg-[var(--color-accent-muted)] glow-sm">
        <div className="flex items-start gap-3">
          <Sparkles className="w-5 h-5 text-[var(--color-accent)] mt-0.5 shrink-0" />
          <div>
            <p className="text-[10px] font-bold text-[var(--color-accent)] uppercase tracking-widest mb-1">Governance Recommendation</p>
            <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{explanations.recommendation}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
