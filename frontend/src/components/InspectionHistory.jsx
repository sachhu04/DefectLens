import React, { useState } from 'react';
import { 
  History, 
  Download, 
  Copy, 
  Check, 
  Trash2, 
  Filter, 
  Eye, 
  Zap, 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  Layers,
  Sparkles
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell
} from 'recharts';

export default function InspectionHistory({
  history,
  onSelectRecord,
  onRemoveRecord,
  onClearHistory,
  deduplicate,
  onToggleDeduplicate
}) {
  const [copied, setCopied] = useState(false);
  const [filterType, setFilterType] = useState('all'); // 'all', 'anomalous', 'normal'

  if (!history || history.length === 0) {
    return (
      <div className="bg-dark-card/40 backdrop-blur-sm border border-dark-border/80 rounded-2xl p-8 text-center animate-fade-in">
        <div className="w-14 h-14 rounded-2xl bg-dark-bg border border-dark-border flex items-center justify-center mx-auto mb-3 text-dark-muted">
          <History className="w-6 h-6 opacity-60" />
        </div>
        <h3 className="text-sm font-semibold text-white mb-1">Live Inspection Session Log</h3>
        <p className="text-xs text-dark-muted max-w-md mx-auto">
          Every image you inspect will be automatically logged here in real time. 
          Track anomaly scores, detect patterns, deduplicate repeat tests, and export the session to CSV.
        </p>
      </div>
    );
  }

  // Filtered history for display
  const filteredHistory = history.filter(item => {
    if (filterType === 'anomalous') return item.prediction === 'anomalous';
    if (filterType === 'normal') return item.prediction === 'normal';
    return true;
  });

  // KPI calculations
  const totalCount = history.length;
  const anomalousCount = history.filter(h => h.prediction === 'anomalous').length;
  const normalCount = totalCount - anomalousCount;
  const defectRate = totalCount > 0 ? ((anomalousCount / totalCount) * 100).toFixed(1) : 0;
  const avgLatency = (history.reduce((acc, h) => acc + h.inferenceMs, 0) / totalCount).toFixed(1);
  const avgScore = (history.reduce((acc, h) => acc + h.anomalyScore, 0) / totalCount).toFixed(3);
  const lastThreshold = history[0]?.threshold || 0.69;

  // Chart data: chronological order (oldest to newest for intuitive left-to-right reading)
  const chartData = [...history].reverse().map((item, idx) => ({
    name: item.fileName.length > 12 ? `${item.fileName.slice(0, 10)}..` : item.fileName,
    fullName: item.fileName,
    score: parseFloat(item.anomalyScore.toFixed(3)),
    threshold: item.threshold,
    prediction: item.prediction,
    category: item.category,
    model: item.model,
    latency: item.inferenceMs.toFixed(1),
    index: idx + 1
  }));

  // Generate CSV text dynamically
  const generateCsv = () => {
    const headers = ['Timestamp', 'Filename', 'Category', 'Model', 'Prediction', 'Anomaly_Score', 'Threshold', 'Latency_ms', 'Token_Reduction_Pct'];
    const rows = history.map(item => [
      `"${item.timestamp}"`,
      `"${item.fileName}"`,
      `"${item.category}"`,
      `"${item.model}"`,
      `"${item.prediction}"`,
      item.anomalyScore.toFixed(4),
      item.threshold.toFixed(4),
      item.inferenceMs.toFixed(2),
      item.tokenReductionPercent.toFixed(1)
    ]);
    return [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  };

  const handleDownloadCsv = () => {
    const csvContent = generateCsv();
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `defectlens_inspection_session_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCopyCsv = () => {
    navigator.clipboard.writeText(generateCsv());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const CustomChartTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const isDefect = data.prediction === 'anomalous';
      return (
        <div className="bg-dark-card border border-dark-border p-3 rounded-xl shadow-2xl text-xs space-y-1.5 backdrop-blur-md">
          <div className="flex items-center justify-between gap-4 border-b border-dark-border/60 pb-1.5">
            <span className="font-semibold text-white truncate max-w-[150px]">{data.fullName}</span>
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${isDefect ? 'bg-status-anomaly/15 text-status-anomaly' : 'bg-status-normal/15 text-status-normal'}`}>
              {data.prediction}
            </span>
          </div>
          <div className="flex justify-between gap-4 text-dark-muted">
            <span>Score:</span>
            <span className="font-bold text-white tabular-nums">{data.score}</span>
          </div>
          <div className="flex justify-between gap-4 text-dark-muted">
            <span>Category Threshold:</span>
            <span className="text-amber-400 tabular-nums">{data.threshold}</span>
          </div>
          <div className="flex justify-between gap-4 text-dark-muted">
            <span>Latency:</span>
            <span className="text-white tabular-nums">{data.latency} ms</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-dark-card/50 backdrop-blur-sm border border-dark-border rounded-2xl p-6 shadow-sm space-y-6 animate-slide-up">
      
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-dark-border">
        <div>
          <div className="flex items-center gap-2">
            <History className="w-5 h-5 text-primary" />
            <h2 className="text-base font-bold text-white tracking-tight">Live Session Inspection Log</h2>
            <span className="px-2 py-0.5 text-[11px] font-medium bg-primary/10 text-primary border border-primary/20 rounded-full">
              {totalCount} {totalCount === 1 ? 'Sample' : 'Samples'}
            </span>
          </div>
          <p className="text-xs text-dark-muted mt-0.5">
            Real-time feed of tested images, score trajectory against threshold, and session analytics.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Deduplication Toggle */}
          <button
            onClick={() => onToggleDeduplicate(!deduplicate)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all ${
              deduplicate 
                ? 'bg-primary/10 border-primary/30 text-primary hover:bg-primary/20' 
                : 'bg-dark-bg border-dark-border text-dark-muted hover:text-white'
            }`}
            title={deduplicate ? "Smart Deduplication: Re-testing an image updates its entry in place" : "Deduplication disabled: Every test creates a new row"}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>{deduplicate ? 'Deduplication: Active' : 'Deduplication: Off'}</span>
          </button>

          {/* Copy CSV */}
          <button
            onClick={handleCopyCsv}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-dark-bg hover:bg-dark-border border border-dark-border text-dark-muted hover:text-white transition-all shadow-sm"
            title="Copy session data to clipboard as CSV"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-status-normal" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy CSV'}</span>
          </button>

          {/* Download CSV */}
          <button
            onClick={handleDownloadCsv}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-primary hover:bg-primary-hover text-white transition-all shadow-md shadow-primary/20"
            title="Download inspection session log as CSV"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Session CSV</span>
          </button>

          {/* Clear Session */}
          <button
            onClick={onClearHistory}
            className="p-1.5 rounded-xl text-dark-muted hover:text-status-anomaly hover:bg-status-anomaly/10 border border-transparent hover:border-status-anomaly/20 transition-all"
            title="Clear inspection history"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-dark-bg/60 border border-dark-border/80 p-3.5 rounded-xl">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-dark-muted block mb-0.5">Tested Images</span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-white">{totalCount}</span>
            <span className="text-[11px] text-dark-muted font-normal">({deduplicate ? 'unique' : 'total'})</span>
          </div>
        </div>

        <div className="bg-dark-bg/60 border border-dark-border/80 p-3.5 rounded-xl">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-dark-muted block mb-0.5">Defect Rate</span>
          <div className="flex items-baseline gap-1.5">
            <span className={`text-xl font-bold ${anomalousCount > 0 ? 'text-status-anomaly' : 'text-status-normal'}`}>
              {defectRate}%
            </span>
            <span className="text-[11px] text-dark-muted font-normal">({anomalousCount} defective)</span>
          </div>
        </div>

        <div className="bg-dark-bg/60 border border-dark-border/80 p-3.5 rounded-xl">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-dark-muted block mb-0.5">Avg Score</span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-white">{avgScore}</span>
            <span className="text-[11px] text-dark-muted font-normal">(threshold {lastThreshold})</span>
          </div>
        </div>

        <div className="bg-dark-bg/60 border border-dark-border/80 p-3.5 rounded-xl">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-dark-muted block mb-0.5">Avg Latency</span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-primary">{avgLatency}</span>
            <span className="text-[11px] text-dark-muted font-normal">ms / image</span>
          </div>
        </div>
      </div>

      {/* Interactive Anomaly Score Real-Time Chart */}
      <div className="bg-dark-bg/40 border border-dark-border/60 rounded-xl p-4 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-white flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-primary" />
            Anomaly Score Distribution vs. Decision Threshold
          </span>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="flex items-center gap-1 text-dark-muted">
              <span className="w-2.5 h-2.5 rounded-sm bg-status-normal" /> Normal
            </span>
            <span className="flex items-center gap-1 text-dark-muted">
              <span className="w-2.5 h-2.5 rounded-sm bg-status-anomaly" /> Defective
            </span>
            <span className="flex items-center gap-1 text-amber-400">
              <span className="w-3 h-0.5 border-t border-dashed border-amber-400" /> Threshold ({lastThreshold})
            </span>
          </div>
        </div>

        <div className="h-44 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" opacity={0.6} />
              <XAxis dataKey="name" stroke="#71717a" fontSize={10} tickLine={false} />
              <YAxis stroke="#71717a" fontSize={10} tickLine={false} domain={[0, 'auto']} />
              <Tooltip content={<CustomChartTooltip />} />
              <ReferenceLine 
                y={lastThreshold} 
                stroke="#f59e0b" 
                strokeDasharray="4 4" 
                strokeWidth={1.5}
                label={{ value: `Threshold`, fill: '#f59e0b', fontSize: 10, position: 'insideTopRight' }}
              />
              <Bar dataKey="score" radius={[4, 4, 0, 0]} maxBarSize={36}>
                {chartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.prediction === 'normal' ? '#10b981' : '#ef4444'} 
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Filter Tabs & History Table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-white">Inspected Records</span>
          <div className="flex gap-1 bg-dark-bg p-0.5 rounded-lg border border-dark-border">
            <button
              onClick={() => setFilterType('all')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${filterType === 'all' ? 'bg-primary text-white' : 'text-dark-muted hover:text-white'}`}
            >
              All ({totalCount})
            </button>
            <button
              onClick={() => setFilterType('normal')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${filterType === 'normal' ? 'bg-status-normal text-white' : 'text-dark-muted hover:text-white'}`}
            >
              Normal ({normalCount})
            </button>
            <button
              onClick={() => setFilterType('anomalous')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${filterType === 'anomalous' ? 'bg-status-anomaly text-white' : 'text-dark-muted hover:text-white'}`}
            >
              Defects ({anomalousCount})
            </button>
          </div>
        </div>

        <div className="border border-dark-border/80 rounded-xl overflow-hidden overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-dark-bg/80 border-b border-dark-border text-dark-muted text-[11px] uppercase tracking-wider font-semibold">
                <th className="py-2.5 px-3">Preview</th>
                <th className="py-2.5 px-3">File Name</th>
                <th className="py-2.5 px-3">Category</th>
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Result</th>
                <th className="py-2.5 px-3">Anomaly Score</th>
                <th className="py-2.5 px-3">Latency</th>
                <th className="py-2.5 px-3">Time</th>
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-border/50 bg-dark-bg/20">
              {filteredHistory.map((item) => {
                const isDefect = item.prediction === 'anomalous';
                return (
                  <tr key={item.id} className="hover:bg-dark-card/60 transition-colors group">
                    <td className="py-2 px-3">
                      <div className="w-8 h-8 rounded-lg overflow-hidden border border-dark-border bg-dark-card shrink-0">
                        {item.images?.original ? (
                          <img 
                            src={`data:image/png;base64,${item.images.original}`} 
                            alt={item.fileName} 
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-[10px] text-dark-muted">IMG</div>
                        )}
                      </div>
                    </td>
                    <td className="py-2 px-3 font-medium text-white max-w-[160px] truncate" title={item.fileName}>
                      {item.fileName}
                    </td>
                    <td className="py-2 px-3 text-dark-muted capitalize">
                      {item.category}
                    </td>
                    <td className="py-2 px-3">
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-dark-border text-dark-muted">
                        {item.model}
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                        isDefect ? 'bg-status-anomaly/10 text-status-anomaly border border-status-anomaly/20' : 'bg-status-normal/10 text-status-normal border border-status-normal/20'
                      }`}>
                        {isDefect ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
                        {item.prediction}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-mono font-semibold text-white">
                      <span className={item.anomalyScore > item.threshold ? 'text-status-anomaly' : 'text-white'}>
                        {item.anomalyScore.toFixed(3)}
                      </span>
                      <span className="text-[10px] text-dark-muted font-normal ml-1">
                        / {item.threshold}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-dark-muted tabular-nums">
                      {item.inferenceMs.toFixed(1)} ms
                    </td>
                    <td className="py-2 px-3 text-dark-muted text-[11px] tabular-nums">
                      {item.timestamp}
                    </td>
                    <td className="py-2 px-3 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => onSelectRecord(item)}
                          className="p-1 rounded text-dark-muted hover:text-white hover:bg-dark-border transition-colors"
                          title="Load this result and heatmaps back into main viewer"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => onRemoveRecord(item.id)}
                          className="p-1 rounded text-dark-muted hover:text-status-anomaly hover:bg-status-anomaly/10 transition-colors"
                          title="Remove from session log"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
