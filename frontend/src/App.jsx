import { useEffect, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`)
        if (!res.ok) throw new Error('Health check failed')
        const data = await res.json()
        if (data.status === 'ok') {
          setBackendStatus('online')
        } else {
          setBackendStatus('offline')
        }
      } catch {
        setBackendStatus('offline')
      }
    }
    checkHealth()
  }, [])

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0]
    if (!selected) {
      setFile(null)
      setPreviewUrl(null)
      return
    }
    setError(null)
    setResult(null)
    setFile(selected)
    const url = URL.createObjectURL(selected)
    setPreviewUrl(url)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)

    if (!file) {
      setError('Please select a histopathology image to upload.')
      return
    }

    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_BASE_URL}/predict?explain=true&task=stage`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || 'Prediction request failed')
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error during prediction.')
    } finally {
      setLoading(false)
    }
  }

  const statusLabel =
    backendStatus === 'checking'
      ? 'Checking...'
      : backendStatus === 'online'
        ? 'Online'
        : 'Offline'

  const statusColor =
    backendStatus === 'online'
      ? 'bg-emerald-500/10 text-emerald-400 ring-emerald-500/40'
      : backendStatus === 'checking'
        ? 'bg-amber-500/10 text-amber-400 ring-amber-500/40'
        : 'bg-rose-500/10 text-rose-400 ring-rose-500/40'

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top navigation */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-500 via-sky-400 to-emerald-400 shadow-lg shadow-indigo-500/40">
              <span className="text-lg font-semibold text-slate-950">CS</span>
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">
                Cancer Stage Prediction
              </p>
              <p className="text-xs text-slate-400">
                Histopathology images with Grad-CAM explainability
              </p>
            </div>
          </div>

          <nav className="hidden items-center gap-6 text-sm text-slate-300 md:flex">
            <span className="hover:text-white">Predict</span>
            <span className="hover:text-white">Explainability</span>
            <span className="hover:text-white">Metrics</span>
          </nav>
        </div>
      </header>

      {/* Main layout */}
      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-6 md:flex-row md:px-6 md:py-8">
        {/* Sidebar */}
        <aside className="flex w-full flex-col gap-4 md:w-64">
          <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-lg shadow-slate-900/60">
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
              System status
            </h2>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Backend API</span>
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${statusColor}`}
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-current" />
                  {statusLabel}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Last prediction</span>
                <span className="text-xs text-slate-400">{result ? 'Completed' : 'Not yet run'}</span>
              </div>
            </div>
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-300 shadow-lg shadow-slate-900/60">
            <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
              How to use
            </h2>
            <ol className="space-y-2 list-decimal list-inside">
              <li>Select a histopathology image (PNG/JPEG).</li>
              <li>Click “Run prediction”.</li>
              <li>Review the predicted stage and risk.</li>
              <li>Inspect the Grad-CAM heatmap overlay.</li>
            </ol>
          </section>
        </aside>

        {/* Main content */}
        <section className="flex-1 space-y-6">
          {/* Input panel */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-xl shadow-slate-900/70 md:p-6">
            <div className="mb-4 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <h1 className="text-lg font-semibold tracking-tight md:text-xl">
                  Upload histopathology image
                </h1>
                <p className="text-xs text-slate-400 md:text-sm">
                  The backend model will predict cancer stage and progression risk and generate a Grad-CAM heatmap.
                </p>
              </div>
              <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/10 px-3 py-1 text-[11px] font-medium text-indigo-400 ring-1 ring-indigo-500/40">
                React · FastAPI · PyTorch
              </span>
            </div>

            <form className="grid gap-4 md:grid-cols-2" onSubmit={handleSubmit}>
              <div className="space-y-3 md:col-span-1">
                <label className="text-xs font-medium text-slate-300">
                  Histopathology image
                </label>
                <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-700 bg-slate-900/70 px-4 py-6 text-center text-xs text-slate-400 hover:border-indigo-400 hover:text-indigo-300">
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                  <span className="mb-1 font-medium text-slate-200">
                    {file ? file.name : 'Click to select an image'}
                  </span>
                  <span>PNG, JPG, or JPEG • up to a few MB</span>
                </label>
                {error && <p className="text-xs text-rose-400">{error}</p>}
              </div>

              <div className="space-y-3 md:col-span-1">
                {previewUrl && (
                  <div className="space-y-1.5">
                    <p className="text-xs font-medium text-slate-300">Preview</p>
                    <div className="overflow-hidden rounded-xl border border-slate-700 bg-slate-950/60">
                      <img
                        src={previewUrl}
                        alt="Selected histopathology"
                        className="h-48 w-full object-cover"
                      />
                    </div>
                  </div>
                )}

                <div className="flex justify-end pt-2">
                  <button
                    type="submit"
                    disabled={loading || backendStatus !== 'online'}
                    className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 via-sky-500 to-emerald-400 px-4 py-2 text-xs font-semibold text-slate-950 shadow-lg shadow-indigo-500/40 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {loading ? 'Running prediction...' : 'Run prediction'}
                  </button>
                </div>
              </div>
            </form>
          </div>

          {/* Results panel */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200 shadow-lg shadow-slate-900/70 md:p-5">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h2 className="text-[13px] font-semibold tracking-tight">
                Prediction and explainability
              </h2>
              <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">
                {result ? 'Latest prediction' : 'Waiting for first run'}
              </span>
            </div>

            {!result && (
              <p className="text-[11px] text-slate-400">
                Upload a histopathology image and run a prediction to see the predicted cancer
                stage, progression risk, and Grad-CAM heatmap overlay here.
              </p>
            )}

            {result && (
              <div className="grid gap-4 md:grid-cols-3 md:gap-6">
                <div className="space-y-3 md:col-span-1">
                  <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Stage prediction
                  </h3>
                  <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 text-[11px]">
                    <p className="mb-1 text-slate-300">
                      Predicted stage:{' '}
                      <span className="font-semibold text-indigo-300">
                        {result.stage.label}
                      </span>
                    </p>
                    <p className="mb-2 text-slate-400">
                      Index: <span className="font-mono">{result.stage.index}</span>
                    </p>
                    <p className="mb-1 text-slate-400">Stage probabilities:</p>
                    <ul className="space-y-1 text-[11px] text-slate-300">
                      {result.stage.probabilities.map((p, idx) => (
                        <li key={idx} className="flex items-center justify-between">
                          <span>Stage {idx}</span>
                          <span className="font-mono">{(p * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <h3 className="mt-4 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Progression risk
                  </h3>
                  <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 text-[11px]">
                    <p className="mb-1 text-slate-300">
                      Predicted risk:{' '}
                      <span className="font-semibold text-emerald-300">
                        {result.progression.label}
                      </span>
                    </p>
                    <p className="mb-2 text-slate-400">
                      Index: <span className="font-mono">{result.progression.index}</span>
                    </p>
                    <p className="mb-1 text-slate-400">Risk probabilities:</p>
                    <ul className="space-y-1 text-[11px] text-slate-300">
                      {result.progression.probabilities.map((p, idx) => (
                        <li key={idx} className="flex items-center justify-between">
                          <span>
                            {idx === 0 ? 'Low' : idx === 1 ? 'Medium' : 'High'} risk
                          </span>
                          <span className="font-mono">{(p * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="space-y-3 md:col-span-2">
                  <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Grad-CAM heatmap
                  </h3>
                  <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950/60">
                    {result.explainability?.heatmap ? (
                      <img
                        src={result.explainability.heatmap}
                        alt="Grad-CAM heatmap overlay"
                        className="w-full object-contain"
                      />
                    ) : (
                      <div className="flex h-48 items-center justify-center text-[11px] text-slate-500">
                        No heatmap available for this prediction.
                      </div>
                    )}
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Colored regions indicate areas that most strongly influenced the model&apos;s
                    {result.explainability?.task === 'progression'
                      ? ' progression risk'
                      : ' stage'}{' '}
                    prediction. Use this to qualitatively assess whether the model is focusing on
                    tumor regions.
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="border-t border-slate-900/80 bg-slate-950/80">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-2 px-4 py-4 text-[11px] text-slate-500 md:flex-row md:px-6">
          <p>AI Tool &mdash; Cancer Stage Prediction with Explainable AI</p>
          <p>
            Backend:{' '}
            <code className="text-slate-400">
              {API_BASE_URL}
            </code>
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
