"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import {
  ArrowRight, Search, Wrench, GitCompare, Brain,
  ChevronRight, Zap, Lock, BarChart3, Shield, MoveRight,
} from "lucide-react";
import { WebGLShader } from "@/components/ui/web-gl-shader";
import { LiquidButton } from "@/components/ui/liquid-glass-button";
import { Button } from "@/components/ui/button";

const FEATURES = [
  {
    icon: Search,
    phase: "Phase 1",
    title: "Bias Detection",
    desc: "Automated scanning for distribution imbalance, outcome disparity, proxy correlation, and intersectional biases across all protected attributes.",
  },
  {
    icon: Wrench,
    phase: "Phase 2",
    title: "Bias Mitigation",
    desc: "Evaluate 10+ debiasing strategies — resampling, reweighting, fairlearn reduction, and threshold optimization — ranked by accuracy-fairness tradeoff.",
  },
  {
    icon: GitCompare,
    phase: "Phase 3",
    title: "Dataset Comparison",
    desc: "Statistical comparison of baseline vs. debiased datasets with KS-tests, representation shift tracking, and fairness delta analysis.",
  },
  {
    icon: Brain,
    phase: "Phase 4",
    title: "Model Explainability",
    desc: "Gemini-powered natural language explanations of model behavior, feature importance shifts, and fairness-accuracy tradeoff reasoning.",
  },
];

const STATS = [
  { value: "10+", label: "Mitigation Strategies" },
  { value: "5", label: "Bias Dimensions" },
  { value: "4", label: "Pipeline Phases" },
  { value: "< 30s", label: "End-to-End Analysis" },
];

/* ─── Animated Hero Section ───────────────────────────────────── */
function AnimatedHero() {
  const [titleNumber, setTitleNumber] = useState(0);
  const titles = useMemo(
    () => ["Detect", "Mitigate", "Compare", "Explain", "Govern"],
    []
  );

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setTitleNumber(titleNumber === titles.length - 1 ? 0 : titleNumber + 1);
    }, 2000);
    return () => clearTimeout(timeoutId);
  }, [titleNumber, titles]);

  return (
    <div className="flex gap-8 py-20 lg:py-32 items-center justify-center flex-col">
      {/* Badge */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <Button variant="secondary" size="sm" className="gap-3 bg-white/10 border border-white/10 text-white/80 hover:bg-white/15 backdrop-blur-sm rounded-full px-5">
          <Zap className="w-3.5 h-3.5" />
          Powered by Gemini AI
          <MoveRight className="w-3.5 h-3.5" />
        </Button>
      </motion.div>


      {/* Animated Title */}
      <div className="flex gap-4 flex-col">
        <h1 className="text-5xl md:text-7xl max-w-3xl tracking-tighter text-center font-regular">
          <span className="text-white/90">AI Bias? We</span>
          <span className="relative flex w-full justify-center overflow-hidden text-center md:pb-4 md:pt-1">
            &nbsp;
            {titles.map((title, index) => (
              <motion.span
                key={index}
                className="absolute font-bold bg-gradient-to-r from-blue-400 via-cyan-300 to-blue-500 bg-clip-text text-transparent"
                initial={{ opacity: 0, y: -100 }}
                transition={{ type: "spring", stiffness: 50 }}
                animate={
                  titleNumber === index
                    ? { y: 0, opacity: 1 }
                    : { y: titleNumber > index ? -150 : 150, opacity: 0 }
                }
              >
                {title}
              </motion.span>
            ))}
          </span>
        </h1>

        <p className="text-lg md:text-xl leading-relaxed tracking-tight text-white/50 max-w-2xl text-center mx-auto">
          An enterprise-grade AI governance platform that systematically audits
          machine learning models for fairness, applies corrective strategies,
          and generates transparent, explainable reports.
        </p>
      </div>

      {/* CTA Buttons */}
      <div className="flex flex-row gap-4">
        <Link href="/results">
          <Button size="lg" className="gap-3 bg-white/5 border border-white/10 text-white/70 hover:bg-white/10 hover:text-white backdrop-blur-sm" variant="outline">
            Explore Demo <ChevronRight className="w-4 h-4" />
          </Button>
        </Link>
        <Link href="/upload">
          <LiquidButton className="text-white border border-white/20 rounded-full" size="xl">
            Begin Bias Analysis <ArrowRight className="w-4 h-4" />
          </LiquidButton>
        </Link>
      </div>
    </div>
  );
}

/* ─── Landing Page ────────────────────────────────────────────── */
export default function LandingPage() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* WebGL Shader Background */}
      <WebGLShader />

      {/* Content overlay */}
      <div className="relative z-10">
        {/* ──── Header ──── */}
        <header className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl font-bold tracking-tighter text-white">AEGIS</span>
          </div>
          <Link
            href="/upload"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10 hover:bg-white/15 text-white text-sm font-semibold transition-colors"
          >
            Launch Analysis <ArrowRight className="w-4 h-4" />
          </Link>
        </header>

        {/* ──── Animated Hero ──── */}
        <section className="max-w-7xl mx-auto px-6">
          <AnimatedHero />

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-2xl mx-auto mt-8 mb-20"
          >
            {STATS.map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">{stat.value}</div>
                <div className="text-[11px] text-white/40 mt-1 font-medium uppercase tracking-wider">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </section>

        {/* ──── Pipeline Phases ──── */}
        <section className="max-w-7xl mx-auto px-6 pb-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="text-center mb-14"
          >
            <h2 className="text-2xl font-bold mb-3 text-white">Four-Phase Governance Pipeline</h2>
            <p className="text-sm text-white/40 max-w-md mx-auto">
              End-to-end bias governance — from automated detection through AI-powered explainability
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
            {FEATURES.map((feat, i) => {
              const Icon = feat.icon;
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 + i * 0.1 }}
                  className="group rounded-xl p-6 transition-all duration-300 bg-white/[0.03] backdrop-blur-sm border border-white/[0.06] hover:bg-white/[0.06] hover:border-white/10"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                      <Icon className="w-5 h-5 text-blue-400 group-hover:text-blue-300 transition-colors" />
                    </div>
                    <span className="text-[10px] font-bold uppercase tracking-widest text-white/30 group-hover:text-blue-400/60 transition-colors">{feat.phase}</span>
                  </div>
                  <h3 className="text-sm font-bold mb-2 text-white">{feat.title}</h3>
                  <p className="text-xs text-white/40 leading-relaxed">{feat.desc}</p>
                </motion.div>
              );
            })}
          </div>
        </section>

        {/* ──── Trust Bar ──── */}
        <section className="border-t border-white/[0.06] py-8">
          <div className="max-w-7xl mx-auto px-6 flex items-center justify-center gap-8 text-[11px] text-white/30 font-medium">
            <div className="flex items-center gap-2">
              <Lock className="w-3.5 h-3.5" />
              Enterprise-Grade Architecture
            </div>
            <div className="w-px h-4 bg-white/10" />
            <div className="flex items-center gap-2">
              <BarChart3 className="w-3.5 h-3.5" />
              UCI Adult Dataset Validated
            </div>
            <div className="w-px h-4 bg-white/10" />
            <div className="flex items-center gap-2">
              <Shield className="w-3.5 h-3.5" />
              Deterministic Mitigation Decisions
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
