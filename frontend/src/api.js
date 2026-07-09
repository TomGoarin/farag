// URL configurable via variable Vite (VITE_API_URL) ou constante par défaut.
export const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

/**
 * Interroge POST /ask.
 * Retourne { ok: true, data } en cas de succès,
 *          { ok: false, error, status } en cas d'erreur.
 */
export async function ask({ question, mode }) {
  let response;
  try {
    response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode }),
    });
  } catch (e) {
    return {
      ok: false,
      status: 0,
      error: `Impossible de joindre le backend (${API_URL}). Est-il démarré ?`,
    };
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    /* body non-JSON — on tombera sur les branches ci-dessous */
  }

  if (response.ok) {
    return { ok: true, data: payload };
  }

  // Backend structuré : { detail: { error } } (503) ou { detail: [...] } (422).
  const detail = payload && payload.detail;
  let message = "Erreur inconnue.";
  if (detail && typeof detail === "object" && !Array.isArray(detail) && detail.error) {
    message = detail.error;
  } else if (Array.isArray(detail) && detail.length > 0) {
    message = detail.map((d) => d.msg || JSON.stringify(d)).join(" · ");
  } else if (payload && payload.error) {
    message = payload.error;
  } else {
    message = `Erreur HTTP ${response.status}`;
  }
  return { ok: false, status: response.status, error: message };
}
