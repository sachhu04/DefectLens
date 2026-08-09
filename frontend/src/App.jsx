import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Camera, Activity, BarChart2, Zap, Settings, Play, Image as ImageIcon, Upload } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('inspection');
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('bottle');
  const [selectedModel, setSelectedModel] = useState('mctf');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    axios.get(`${API_BASE}/categories`).then(res => {
      if (res.data.categories.length > 0) {
        setCategories(res.data.categories);
        setSelectedCategory(res.data.categories[0]);
      }
    }).catch(err => console.error(err));
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
    }
  };

  const handleInspect = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', selectedCategory);
    formData.append('model_type', selectedModel);

    try {
      const res = await axios.post(`${API_BASE}/predict`, formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert('Inference failed. Check console.');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col font-sans bg-dark-bg text-dark-text selection:bg-primary/30">
      {/* Navbar */}
      <nav className="sticky top-0 z-50 bg-dark-bg/80 backdrop-blur-xl border-b border-dark-border px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Camera className="text-primary w-5 h-5" />
            </div>
            <span className="text-xl font-bold tracking-tight">DefectLens</span>
            <span className="hidden md:inline-block px-2 py-0.5 ml-2 text-xs font-medium bg-dark-border text-dark-muted rounded-full">Efficient ViT AD</span>
          </div>
          <div className="flex gap-2 bg-dark-card p-1 rounded-lg border border-dark-border">
            <button onClick={() => setActiveTab('inspection')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-300 ${activeTab === 'inspection' ? 'bg-primary text-white shadow-lg shadow-primary/25' : 'text-dark-muted hover:text-white hover:bg-dark-border'}`}>Inspection</button>
            <button onClick={() => setActiveTab('benchmark')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-300 ${activeTab === 'benchmark' ? 'bg-primary text-white shadow-lg shadow-primary/25' : 'text-dark-muted hover:text-white hover:bg-dark-border'}`}>Benchmark</button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 w-full max-w-7xl mx-auto p-6 animate-fade-in">
        {activeTab === 'inspection' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Sidebar Controls */}
            <div className="lg:col-span-4 space-y-6">
              <div className="bg-dark-card/50 backdrop-blur-sm p-6 rounded-2xl border border-dark-border shadow-sm animate-slide-up">
                <div className="flex items-center gap-2 mb-6">
                  <Settings className="w-4 h-4 text-primary" />
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-dark-muted">Configuration</h2>
                </div>
              
                <div className="mb-5">
                  <label className="block text-xs font-medium text-dark-muted uppercase tracking-wider mb-2">Category</label>
                  <div className="relative">
                    <select 
                      value={selectedCategory} 
                      onChange={e => setSelectedCategory(e.target.value)}
                      className="w-full bg-dark-bg border border-dark-border rounded-lg p-3 text-sm text-white appearance-none outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                    >
                      {categories.map(c => <option key={c} value={c}>{c}</option>)}
                      {categories.length === 0 && <option value="bottle">bottle (Default)</option>}
                    </select>
                  </div>
                </div>

                <div className="mb-6">
                  <label className="block text-xs font-medium text-dark-muted uppercase tracking-wider mb-2">Model Pipeline</label>
                  <select 
                    value={selectedModel} 
                    onChange={e => setSelectedModel(e.target.value)}
                    className="w-full bg-dark-bg border border-dark-border rounded-lg p-3 text-sm text-white appearance-none outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                  >
                    <option value="mctf">Efficient ViT (MCTF)</option>
                    <option value="baseline">Standard ViT (Baseline)</option>
                  </select>
                </div>

                <div className="mb-6">
                  <label className="block text-xs font-medium text-dark-muted uppercase tracking-wider mb-2">Input Image</label>
                  <label className="flex flex-col items-center justify-center w-full h-32 bg-dark-bg border border-dark-border border-dashed rounded-xl cursor-pointer hover:bg-dark-border/50 hover:border-primary/50 transition-all duration-300 group">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-6 h-6 mb-3 text-dark-muted group-hover:text-primary transition-colors" />
                      <p className="text-sm font-medium text-dark-muted group-hover:text-white transition-colors">Select image to inspect</p>
                    </div>
                    <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                  </label>
                </div>

                <button 
                  onClick={handleInspect}
                  disabled={!file || loading}
                  className="w-full bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium py-3 px-4 rounded-xl shadow-lg shadow-primary/20 flex items-center justify-center gap-2 transition-all duration-300"
                >
                  {loading ? <Activity className="animate-spin w-4 h-4"/> : <Play className="w-4 h-4"/>}
                  {loading ? 'Analyzing...' : 'Run Inspection'}
                </button>
              </div>
            </div>

            {/* Results Area */}
            <div className="lg:col-span-8 space-y-6">
              
              {/* Metrics */}
              {result && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 animate-slide-up" style={{ animationDelay: '0.1s' }}>
                  <div className={`p-5 rounded-2xl border flex flex-col justify-center relative overflow-hidden ${result.prediction === 'normal' ? 'bg-status-normal/5 border-status-normal/30' : 'bg-status-anomaly/5 border-status-anomaly/30'}`}>
                    <div className={`absolute top-0 right-0 w-16 h-16 blur-2xl rounded-full opacity-20 ${result.prediction === 'normal' ? 'bg-status-normal' : 'bg-status-anomaly'}`}></div>
                    <span className="text-xs font-semibold uppercase text-dark-muted tracking-widest mb-1 z-10">Prediction</span>
                    <span className={`text-2xl font-bold uppercase tracking-tight z-10 ${result.prediction === 'normal' ? 'text-status-normal' : 'text-status-anomaly'}`}>{result.prediction}</span>
                  </div>
                  <div className="bg-dark-card/50 backdrop-blur-sm border border-dark-border p-5 rounded-2xl flex flex-col justify-center">
                    <span className="text-xs font-semibold uppercase text-dark-muted tracking-widest flex items-center gap-1.5 mb-1"><Activity className="w-3.5 h-3.5"/> Score</span>
                    <span className="text-2xl font-bold tracking-tight text-white">{result.anomaly_score.toFixed(3)}</span>
                  </div>
                  <div className="bg-dark-card/50 backdrop-blur-sm border border-dark-border p-5 rounded-2xl flex flex-col justify-center">
                    <span className="text-xs font-semibold uppercase text-dark-muted tracking-widest flex items-center gap-1.5 mb-1"><Zap className="w-3.5 h-3.5"/> Latency</span>
                    <span className="text-2xl font-bold tracking-tight text-white">{result.inference_ms.toFixed(1)} <span className="text-sm font-normal text-dark-muted">ms</span></span>
                  </div>
                  <div className="bg-dark-card/50 backdrop-blur-sm border border-dark-border p-5 rounded-2xl flex flex-col justify-center">
                    <span className="text-xs font-semibold uppercase text-dark-muted tracking-widest flex items-center gap-1.5 mb-1"><BarChart2 className="w-3.5 h-3.5"/> Token Reduction</span>
                    <span className="text-2xl font-bold tracking-tight text-primary">{result.token_reduction_percent.toFixed(1)}<span className="text-sm font-normal">%</span></span>
                  </div>
                </div>
              )}

              {/* Visualizations */}
              <div className={`bg-dark-card/30 backdrop-blur-sm border border-dark-border rounded-2xl overflow-hidden min-h-[480px] flex items-center justify-center p-8 shadow-sm ${result ? 'animate-slide-up' : 'animate-fade-in'}`} style={{ animationDelay: result ? '0.2s' : '0s' }}>
                {!preview && !result && (
                  <div className="text-dark-muted flex flex-col items-center">
                    <div className="w-20 h-20 bg-dark-bg rounded-full flex items-center justify-center mb-6 border border-dark-border shadow-inner">
                      <ImageIcon className="w-8 h-8 opacity-50" />
                    </div>
                    <p className="text-sm font-medium">Awaiting inspection image...</p>
                  </div>
                )}
                
                {preview && !result && (
                  <img src={preview} alt="Preview" className="max-h-[500px] object-contain rounded-xl border border-dark-border/50 shadow-2xl animate-fade-in" />
                )}

                {result && (
                  <div className="grid grid-cols-2 gap-6 w-full animate-fade-in">
                    <div className="flex flex-col gap-3 group">
                      <span className="text-xs font-semibold text-dark-muted uppercase tracking-widest pl-1">Original</span>
                      <div className="rounded-xl border border-dark-border overflow-hidden relative shadow-lg group-hover:border-primary/30 transition-colors">
                        <img src={`data:image/png;base64,${result.images.original}`} className="w-full object-cover transform group-hover:scale-[1.02] transition-transform duration-500" />
                      </div>
                    </div>
                    <div className="flex flex-col gap-3 group">
                      <span className="text-xs font-semibold text-dark-muted uppercase tracking-widest pl-1">Heatmap</span>
                      <div className="rounded-xl border border-dark-border overflow-hidden relative shadow-lg group-hover:border-primary/30 transition-colors">
                        <img src={`data:image/png;base64,${result.images.heatmap}`} className="w-full object-cover transform group-hover:scale-[1.02] transition-transform duration-500" />
                      </div>
                    </div>
                    <div className="flex flex-col gap-3 group">
                      <span className="text-xs font-semibold text-dark-muted uppercase tracking-widest pl-1">Overlay</span>
                      <div className="rounded-xl border border-dark-border overflow-hidden relative shadow-lg group-hover:border-primary/30 transition-colors">
                        <img src={`data:image/png;base64,${result.images.overlay}`} className="w-full object-cover transform group-hover:scale-[1.02] transition-transform duration-500" />
                      </div>
                    </div>
                    <div className="flex flex-col gap-3 group">
                      <span className="text-xs font-semibold text-dark-muted uppercase tracking-widest pl-1">Suspicious Region</span>
                      <div className="rounded-xl border border-dark-border overflow-hidden relative shadow-lg group-hover:border-primary/30 transition-colors">
                        <img src={`data:image/png;base64,${result.images.region}`} className="w-full object-cover transform group-hover:scale-[1.02] transition-transform duration-500" />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'benchmark' && (
          <div className="bg-dark-card/30 backdrop-blur-sm border border-dark-border p-12 rounded-2xl text-center shadow-sm animate-fade-in max-w-3xl mx-auto mt-10">
            <div className="w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-8 shadow-inner border border-primary/20">
              <BarChart2 className="w-10 h-10 text-primary" />
            </div>
            <h2 className="text-3xl font-bold tracking-tight mb-4 text-white">Benchmark Dashboard</h2>
            <p className="text-dark-muted text-lg mb-10 leading-relaxed">
              Run automated evaluations on the entire dataset to compare Baseline ViT vs MCTF across AUROC, latency, and FLOPs.
            </p>
            <div className="p-6 bg-dark-bg border border-dark-border border-dashed rounded-xl inline-block">
               <span className="text-sm font-medium text-dark-muted">(Charts populated from results/summary.csv)</span>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
