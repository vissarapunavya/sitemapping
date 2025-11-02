import React, { useEffect, useState } from 'react'
import axios from 'axios'
import './App.css';

const api = axios.create({ baseURL: '/api' }) // uses Vite proxy

export default function App() {
  const [apiStatus, setApiStatus] = useState('checking')
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('soil')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const init = async () => {
      try { await api.get('/health'); setApiStatus('connected') } catch { setApiStatus('disconnected') }
      try { const { data } = await api.get('/models'); setModels(data.models || []) } catch {}
    }
    init()
  }, [])

  const onChooseFile = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    if (!f.type.startsWith('image/')) { alert('Select an image'); return }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResults(null)
  }

  const predict = async () => {
    if (!file) return
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('model_type', selectedModel)
      const { data } = await api.post('/predict', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResults(data)
    } catch (e) {
      alert(e?.response?.data?.detail || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => { setFile(null); setPreview(null); setResults(null) }

  const selectedModelInfo = models.find(m => m.id === selectedModel)

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">🌱</span>
          <div>
            <h1>AI Detection Lab</h1>
            <p>Soil Type + Vegetation Land</p>
          </div>
        </div>
        <div className={`status ${apiStatus}`}>{apiStatus === 'connected' ? '✓ API Connected' : apiStatus === 'disconnected' ? '✕ API Disconnected' : '… Checking'}</div>
      </header>

      <main className="layout">
        <aside className="panel">
          <h2>Select Model</h2>
          <div className="model-list">
            {models.map(m => (
              <label key={m.id} className={`model-item ${selectedModel === m.id ? 'active' : ''} ${!m.available ? 'disabled' : ''}`}>
                <input type="radio" name="model" value={m.id} checked={selectedModel === m.id} onChange={() => setSelectedModel(m.id)} disabled={!m.available} />
                <div>
                  <strong>{m.name}</strong>
                  <div className="muted">{m.description}{!m.available && ' • Unavailable'}</div>
                </div>
              </label>
            ))}
          </div>

          <h2>Upload Image</h2>
          <label className="upload">
            <input type="file" accept="image/*" onChange={onChooseFile} hidden />
            {preview ? <img className="thumb" src={preview} alt="preview" /> : <div className="placeholder">📁 Choose image</div>}
          </label>

          <div className="actions">
            <button className="primary" onClick={predict} disabled={!file || loading || apiStatus !== 'connected'}>{loading ? 'Analyzing…' : 'Analyze'}</button>
            <button onClick={reset} disabled={!file && !results}>Reset</button>
          </div>
        </aside>

        <section className="panel">
          {!results ? (
            <div className="empty">
              <div className="big">🔍</div>
              <p>Upload an image and run analysis</p>
              {selectedModelInfo && <p className="hint">{selectedModelInfo.description}</p>}
            </div>
          ) : (
            <>
              {results.summary && <div className="summary-banner"><strong>{results.summary}</strong></div>}
              <div className="grid">
                <div><h3>Original</h3><div className="imgbox"><img src={preview} alt="original" /></div></div>
                <div><h3>Output</h3><div className="imgbox"><img src={`data:image/png;base64,${results.annotated_image}`} alt="annotated" /></div></div>
              </div>
              <div className="stats">
                <div className="stat"><div className="kpi">{results.total_detections}</div><div className="label">Detections</div></div>
                <div className="stat"><div className="kpi">{Object.keys(results.class_counts || {}).length}</div><div className="label">{results.model_type === 'soil' ? 'Soil Types' : 'Vegetation Areas'}</div></div>
              </div>
              {Object.keys(results.class_counts || {}).length > 0 && (
                <div className="detection-summary">
                  <h3>{results.model_type === 'soil' ? 'Detected Soil Types' : 'Vegetation Areas'}</h3>
                  <div className="classes">
                    {Object.entries(results.class_counts).map(([k, v]) => (
                      <div key={k} className="chip"><span className="class-name">{k}</span><span className="count">{v}</span></div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  )
}
