import React from "react"

export default function ResultsGallery({ files = [] }) {
  if (!files || files.length === 0) return <div className="results-empty">No results yet.</div>

  return (
    <div className="results-gallery">
      {files.map((f, i) => (
        <div className="results-item" key={i}>
          <a href={f.url} target="_blank" rel="noreferrer">
            <img
              className="result-image"
              src={f.url}
              alt={f.filename || `result-${i}`}
            />
          </a>
          <div className="result-caption">
            <a className="result-link" href={f.url} download>
              {f.filename || f.url}
            </a>
          </div>
        </div>
      ))}
    </div>
  )
}
