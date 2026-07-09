import { useState } from "react";
import { ask } from "./api.js";
import Source from "./Source.jsx";

const MODES = [
  { value: "dense", label: "Dense (rapide)" },
  { value: "rerank", label: "Rerank (plus précis, plus lent)" },
];

function formatElapsed(ms) {
  if (typeof ms !== "number") return "";
  const s = ms / 1000;
  // 1 décimale, virgule française.
  return `${s.toFixed(1).replace(".", ",")} s`;
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("dense");
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    if (!question.trim() || status === "loading") return;
    setStatus("loading");
    setResult(null);
    setError(null);
    const res = await ask({ question: question.trim(), mode });
    if (res.ok) {
      setResult(res.data);
      setStatus("success");
    } else {
      setError(res.error);
      setStatus("error");
    }
  }

  const busy = status === "loading";

  return (
    <main className="app">
      <header className="app-header">
        <h1>RastaFaRAG</h1>
        <p className="app-tagline">
          Réponses sourcées sur le corpus. Rien d'inventé hors contexte.
        </p>
      </header>

      <form className="ask-form" onSubmit={onSubmit}>
        <label className="field">
          <span className="field-label">Question</span>
          <textarea
            className="question-input"
            rows={3}
            placeholder="Ex. Quel équipement en UMTS joue le rôle du BSC en GSM ?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={busy}
          />
        </label>

        <fieldset className="field mode-field" disabled={busy}>
          <legend className="field-label">
            Choix de l'effort{" "}
            <span className="field-hint">
              — dense = rapide · rerank = plus précis, plus lent
            </span>
          </legend>
          {MODES.map((m) => (
            <label key={m.value} className="mode-option">
              <input
                type="radio"
                name="mode"
                value={m.value}
                checked={mode === m.value}
                onChange={(e) => setMode(e.target.value)}
              />
              <span>{m.label}</span>
            </label>
          ))}
        </fieldset>

        <button
          type="submit"
          className="submit-btn"
          disabled={busy || !question.trim()}
        >
          {busy ? "Recherche en cours…" : "Poser la question"}
        </button>
      </form>

      <section className="result-zone" aria-live="polite">
        {status === "loading" && (
          <div className="loading">
            <span className="spinner" aria-hidden="true" />
            <span className="loading-text">
              Recherche et génération en cours…
              {mode === "rerank" && (
                <em className="loading-hint">
                  {" "}
                  (le mode rerank peut prendre ~10 s sur CPU)
                </em>
              )}
            </span>
          </div>
        )}

        {status === "error" && (
          <div className="error" role="alert">
            <strong>Erreur :</strong> {error}
          </div>
        )}

        {status === "success" && result && (
          <article className="answer-block">
            <h2 className="answer-heading">Réponse</h2>
            <p className="answer-text">{result.answer}</p>
            <p className="meta">
              mode : <code>{result.mode_used}</code>
              {" · "}
              temps : {formatElapsed(result.elapsed_ms)}
            </p>

            {result.sources && result.sources.length > 0 && (
              <div className="sources-block">
                <h3 className="sources-heading">Sources</h3>
                <ol className="sources-list">
                  {result.sources.map((s, i) => (
                    <Source key={i} index={i + 1} source={s} />
                  ))}
                </ol>
              </div>
            )}
          </article>
        )}
      </section>
    </main>
  );
}
