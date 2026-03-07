import React, { useState } from "react"

// API base (can be set in Vite env as VITE_API_BASE). Default to backend on localhost:8000.
const API_BASE = (window.__env__ && window.__env__.VITE_API_BASE) || import.meta.env.VITE_API_BASE || "http://localhost:8000"

const RESOLUTIONS = [
  "256 × 256",
  "512 × 512",
  "512 × 768",
  "768 × 512",
  "768 × 1024",
  "1024 × 768",
  "1024 × 1024",
  "1280 × 720",
  "1280 × 1280",
  "1536 × 1024",
  "1024 × 1536",
  "1920 × 1080",
  "2048 × 2048"
]

const FORMATS = ["PNG (lossless, stores prompt)", "JPG (quality 100)", "JPEG (quality 100)", "WebP (quality 90)"]

const EXAMPLE_PROMPTS = [
  "Contemporary glass-and-concrete museum, dramatic golden hour lighting, ultra-detailed architectural photography, 8k",
  "Baroque cathedral interior, warm volumetric light, intricate stone carvings, high dynamic range",
  "Futuristic vertical city tower, green terraces, evening neon accents, realistic rendering, 8k",
  "Minimalist Japanese teahouse by a pond, soft natural light, photorealistic, high detail",
  "Modern sustainable office building, glass facade, reflections, cinematic overcast lighting, 8k"
]

export default function GenerateForm({ onAccepted }) {
  const [prompt, setPrompt] = useState("")
  const [negative, setNegative] = useState("")
  const [resolution, setResolution] = useState("1024 × 1024")
  const [steps, setSteps] = useState(30)
  const [guidance, setGuidance] = useState(7.5)
  const [batchSize, setBatchSize] = useState(1)
  const [format, setFormat] = useState(FORMATS[0])
  const [loading, setLoading] = useState(false)
  const [statusMsg, setStatusMsg] = useState("")
  const [error, setError] = useState(null)

  function mapResolution(label) {
    return label.replace(/ × /g, "x").replace(/ /g, "")
  }

  function mapFormat(label) {
    if (label.startsWith("PNG")) return "png"
    if (label.startsWith("JPG")) return "jpg"
    if (label.startsWith("JPEG")) return "jpeg"
    if (label.startsWith("WebP")) return "webp"
    return "png"
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!prompt.trim()) return setStatusMsg("Please enter a prompt.")
    setLoading(true)
    setStatusMsg("Submitting job...")
    try {
      const fd = new FormData()
      fd.append("prompt", prompt)
      fd.append("negative_prompt", negative)
      fd.append("resolution", mapResolution(resolution))
      fd.append("steps", steps)
      fd.append("guidance_scale", guidance)
      fd.append("batch_size", batchSize)
      fd.append("output_format", mapFormat(format))

        const res = await fetch(`${API_BASE}/api/v1/generate`, {
          method: 'POST',
          body: fd,
        })

        // Some error responses may not be valid JSON (HTML or empty). Read text and try parse.
        const text = await res.text()
        let data = null
        try {
          data = text ? JSON.parse(text) : null
        } catch (e) {
          data = { detail: text }
        }

        if (!res.ok) {
          const msg = data && data.detail ? data.detail : res.statusText
          setError(`Failed to submit job: ${msg}`)
          setLoading(false)
          return
        }

        // Notify parent that job was accepted so it can start polling / show results
        setError(null)
        setStatusMsg("Job accepted; queued for processing")
        if (typeof onAccepted === "function") onAccepted(data.job_id, `${API_BASE}${data.poll_url}`)
    } catch (err) {
      console.error(err)
      setError("Failed to submit job: " + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form className="form" onSubmit={handleSubmit}>
      <label>Prompt</label>
      <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={4} placeholder="Describe the scene..." />
      <div className="examples">
        <small>Examples: </small>
        {EXAMPLE_PROMPTS.map((p, i) => (
          <button type="button" key={i} className="link" onClick={() => setPrompt(p)}>{`Example ${i+1}`}</button>
        ))}
      </div>

      <label>Negative prompt (optional)</label>
      <input value={negative} onChange={e => setNegative(e.target.value)} />

      <div className="row">
        <div>
          <label>Resolution</label>
          <select value={resolution} onChange={e => setResolution(e.target.value)}>
            {RESOLUTIONS.map(r => <option key={r}>{r}</option>)}
          </select>
        </div>
        <div>
          <label>Format</label>
          <select value={format} onChange={e => setFormat(e.target.value)}>
            {FORMATS.map(f => <option key={f}>{f}</option>)}
          </select>
        </div>
      </div>

      <div className="row">
        <div>
          <label>Steps</label>
          <input type="number" min={1} max={200} value={steps} onChange={e => setSteps(Number(e.target.value))} />
        </div>
        <div>
          <label>Guidance</label>
          <input type="number" step="0.1" min={0} max={30} value={guidance} onChange={e => setGuidance(Number(e.target.value))} />
        </div>
        <div>
          <label>Batch</label>
          <input type="number" min={1} max={100} value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} />
        </div>
      </div>

      <div className="actions">
        <button type="submit" disabled={loading}>{loading ? "Submitting…" : "Generate"}</button>
      </div>
      {statusMsg && <div className="status">{statusMsg}</div>}
      {error && <div className="status" style={{color: 'crimson'}}>{error}</div>}
    </form>
  )
}
