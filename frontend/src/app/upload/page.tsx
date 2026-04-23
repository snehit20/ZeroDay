"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import Image from "next/image";
import {
  Upload, FileSpreadsheet, X, Play, Settings2,
  ChevronDown, Database, Cpu, Box, Check,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { cn } from "@/lib/utils";
import { PipelineMode } from "@/lib/types";

const MODES: { value: PipelineMode; label: string; desc: string }[] = [
  { value: "full_pipeline", label: "Full Pipeline", desc: "Detection + Mitigation + Comparison + Explainability" },
  { value: "analysis", label: "Analysis Only", desc: "Bias detection and dataset statistical comparison" },
  { value: "train", label: "Train & Evaluate", desc: "Detection + Mitigation with model training" },
];

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [mode, setMode] = useState<PipelineMode>("full_pipeline");
  const [isDragging, setIsDragging] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedModelType, setSelectedModelType] = useState("logistic_regression");

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && (dropped.name.endsWith(".csv") || dropped.name.endsWith(".xlsx"))) {
      setFile(dropped);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0]);
  };

  const handleModelSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f && f.name.endsWith(".pkl")) setModelFile(f);
  };

  const handleRun = () => {
    router.push("/processing");
  };

  const activeModelSource = modelFile ? "uploaded" : "selected";

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      <Navbar />

      <main className="max-w-4xl mx-auto px-6 pt-28 pb-20">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-10"
        >
          <h1 className="text-3xl font-bold mb-2">Configure Pipeline</h1>
          <p className="text-sm text-[var(--color-text-muted)] max-w-md mx-auto">
            Upload your training dataset, configure the pipeline mode, and optionally provide a pre-trained model
          </p>
        </motion.div>

        {/* ──── Dataset Upload Zone ──── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-3">
            Training Dataset
          </label>
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={cn(
              "relative rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden",
              isDragging
                ? "border-[var(--color-accent)] bg-[var(--color-accent-muted)] scale-[1.005]"
                : file
                ? "border-[var(--color-success)]/30 bg-[var(--color-success-muted)]"
                : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/30"
            )}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-white/[0.015] to-transparent pointer-events-none" />
            <div className="relative px-8 py-14 text-center">
              <AnimatePresence mode="wait">
                {file ? (
                  <motion.div key="file" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} className="flex flex-col items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-[var(--color-success-muted)] border border-[var(--color-success)]/20 flex items-center justify-center">
                      <FileSpreadsheet className="w-6 h-6 text-[var(--color-success)]" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold">{file.name}</p>
                      <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                        {(file.size / 1024).toFixed(1)} KB &middot; Ready for analysis
                      </p>
                    </div>
                    <button onClick={() => setFile(null)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--color-text-muted)] hover:text-[var(--color-danger)] hover:bg-[var(--color-danger-muted)] transition-colors">
                      <X className="w-3.5 h-3.5" /> Remove File
                    </button>
                  </motion.div>
                ) : (
                  <motion.div key="empty" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} className="flex flex-col items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-[var(--color-accent-muted)] border border-[var(--color-accent)]/20 flex items-center justify-center">
                      <Upload className="w-6 h-6 text-[var(--color-accent)]" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold">Drop your CSV or XLSX file here</p>
                      <p className="text-xs text-[var(--color-text-muted)] mt-0.5">or click below to browse your filesystem</p>
                    </div>
                    <label className="px-5 py-2 rounded-xl bg-[var(--color-surface-elevated)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)] text-xs font-semibold cursor-pointer transition-colors">
                      Browse Files
                      <input type="file" accept=".csv,.xlsx" onChange={handleFileSelect} className="hidden" />
                    </label>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>

        {/* ──── Model Upload ──── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mt-8">
          <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-3">
            Pre-trained Model <span className="text-[var(--color-text-muted)] font-normal normal-case tracking-normal">(optional)</span>
          </label>
          <div className="flex items-center gap-4">
            {modelFile ? (
              <div className="flex-1 flex items-center gap-3 p-4 rounded-xl border border-[var(--color-success)]/30 bg-[var(--color-success-muted)]">
                <Box className="w-5 h-5 text-[var(--color-success)] shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold truncate">{modelFile.name}</p>
                  <p className="text-[10px] text-[var(--color-success)]">Using Uploaded Model</p>
                </div>
                <button onClick={() => setModelFile(null)} className="p-1.5 rounded-lg hover:bg-[var(--color-danger-muted)] transition-colors">
                  <X className="w-3.5 h-3.5 text-[var(--color-text-muted)] hover:text-[var(--color-danger)]" />
                </button>
              </div>
            ) : (
              <label className="flex-1 flex items-center gap-3 p-4 rounded-xl border border-dashed border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/30 cursor-pointer transition-colors">
                <Box className="w-5 h-5 text-[var(--color-text-muted)] shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-[var(--color-text-secondary)]">Upload Model (.pkl)</p>
                  <p className="text-[10px] text-[var(--color-text-muted)]">Using Selected Model: {selectedModelType.replace("_", " ")}</p>
                </div>
                <span className="px-3 py-1 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-[10px] font-semibold">Browse</span>
                <input type="file" accept=".pkl" onChange={handleModelSelect} className="hidden" />
              </label>
            )}
          </div>
          {/* Active model indicator */}
          <div className="flex items-center gap-2 mt-2 text-[10px] text-[var(--color-text-muted)]">
            <Check className="w-3 h-3 text-[var(--color-success)]" />
            Active: {activeModelSource === "uploaded" ? `Uploaded model (${modelFile?.name})` : `${selectedModelType.replace("_", " ")} (built-in)`}
          </div>
        </motion.div>

        {/* ──── Pipeline Mode ──── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="mt-8">
          <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-3">
            Pipeline Mode
          </label>
          <div className="grid grid-cols-3 gap-3">
            {MODES.map((m) => (
              <button
                key={m.value}
                onClick={() => setMode(m.value)}
                className={cn(
                  "p-4 rounded-xl border text-left transition-all duration-200",
                  mode === m.value
                    ? "border-[var(--color-accent)]/40 bg-[var(--color-accent-muted)] glow-sm"
                    : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/20"
                )}
              >
                <p className="text-sm font-semibold mb-1">{m.label}</p>
                <p className="text-[10px] text-[var(--color-text-muted)] leading-relaxed">{m.desc}</p>
              </button>
            ))}
          </div>
        </motion.div>

        {/* ──── Advanced Settings ──── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="mt-6">
          <button onClick={() => setShowSettings(!showSettings)} className="flex items-center gap-2 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors font-medium">
            <Settings2 className="w-3.5 h-3.5" />
            Advanced Configuration
            <ChevronDown className={cn("w-3.5 h-3.5 transition-transform duration-200", showSettings && "rotate-180")} />
          </button>

          <AnimatePresence>
            {showSettings && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                <div className="grid grid-cols-2 gap-4 mt-4 p-5 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
                  <div>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-2">Accuracy Weight (α)</label>
                    <input type="number" defaultValue={0.6} step={0.1} min={0} max={1} className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors" />
                  </div>
                  <div>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-2">Fairness Weight (β)</label>
                    <input type="number" defaultValue={0.4} step={0.1} min={0} max={1} className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors" />
                  </div>
                  <div>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-2">Model Architecture</label>
                    <select value={selectedModelType} onChange={(e) => setSelectedModelType(e.target.value)} className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors">
                      <option value="logistic_regression">Logistic Regression</option>
                      <option value="random_forest">Random Forest</option>
                      <option value="xgboost">XGBoost</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-2">Gemini AI Explanations</label>
                    <select className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors">
                      <option value="enabled">Enabled</option>
                      <option value="disabled">Disabled</option>
                    </select>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* ──── Execute Button ──── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }} className="mt-10 flex justify-center">
          <button onClick={handleRun} className="group flex items-center gap-3 px-10 py-4 rounded-2xl bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white font-bold text-base transition-all glow">
            <Play className="w-5 h-5" />
            Execute Bias Analysis Pipeline
          </button>
        </motion.div>

        {/* ──── Info Footer ──── */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="grid grid-cols-2 gap-4 mt-12">
          <div className="flex items-start gap-3 p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
            <Database className="w-4 h-4 text-[var(--color-accent)] mt-0.5 shrink-0" />
            <div>
              <p className="text-xs font-semibold mb-0.5">No dataset available?</p>
              <p className="text-[10px] text-[var(--color-text-muted)] leading-relaxed">
                Execute without uploading to use the built-in UCI Adult dataset for demonstration purposes.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
            <Cpu className="w-4 h-4 text-[var(--color-accent)] mt-0.5 shrink-0" />
            <div>
              <p className="text-xs font-semibold mb-0.5">Estimated Processing Time</p>
              <p className="text-[10px] text-[var(--color-text-muted)] leading-relaxed">
                Full pipeline completes in ~30 seconds. Analysis-only mode completes in under 1 second.
              </p>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
