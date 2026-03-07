import React, { useState, useEffect } from "react"
import GenerateForm from "./components/GenerateForm"
import ResultsGallery from "./components/ResultsGallery"

const API_BASE = (window.__env__ && window.__env__.VITE_API_BASE) || import.meta.env.VITE_API_BASE || "http://localhost:8000"

export default function App() {
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [files, setFiles] = useState([])
  const [message, setMessage] = useState("")

  useEffect(() => {
    let timer = null
      if (jobId) {
      setStatus("queued")
      timer = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/v1/job/${jobId}`)
          if (!res.ok) throw new Error("failed to fetch job")
          const data = await res.json()
          setStatus(data.status)

          // show partial / final files as they become available
          if (data.result && data.result.files) {
            setFiles(data.result.files.map(f => ({ ...f, url: API_BASE + f.url })))
          }

          // update simple progress message from job meta if present
          if (data.meta) {
            const completed = data.meta.completed || data.meta.get?.("completed") || 0
            const batch = data.meta.batch_size || data.meta.get?.("batch_size") || null
            if (batch) setMessage(`${completed}/${batch} images ready`)
          }

          if (data.status === "finished") {
            setMessage("Job finished")
            clearInterval(timer)
          }
          if (data.status === "failed") {
            setMessage(data.error || "Job failed")
            clearInterval(timer)
          }
        } catch (err) {
          console.error(err)
        }
      }, 2000)
    }
    return () => clearInterval(timer)
  }, [jobId])

  return (
    <div className="app">
      <header className="header">
        <h1>Chronora — SDXL Direct</h1>
        <p className="muted">Generate images (text→image). Jobs are processed in background.</p>
      </header>

      <main className="main">
        <div className="column">
          <GenerateForm onAccepted={(jid, msg) => { setJobId(jid); setMessage(msg); setFiles([]) }} />
          {message && <div className="message">{message}</div>}
        </div>
        <div className="column">
          <h3>Job status: {status || "idle"}</h3>
          <ResultsGallery files={files} />
        </div>
      </main>

      <footer className="footer">
        <small>Tip: open network/worker terminal to view progress and errors.</small>
      </footer>
    </div>
  )
}
