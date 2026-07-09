// Rendu d'une source. Composant isolé pour accueillir plus tard un
// éventuel `chunk_text` (repliable) sans refondre la liste.
export default function Source({ index, source }) {
  const { doc_id, section_path, pages } = source;
  const pagesLabel =
    Array.isArray(pages) && pages.length > 0 ? `p. ${pages.join(", ")}` : null;

  return (
    <li className="source">
      <span className="source-index">[{index}]</span>
      <span className="source-doc-id">{doc_id}</span>
      <span className="source-sep"> · </span>
      <span className="source-section">{section_path || "(racine)"}</span>
      {pagesLabel && (
        <>
          <span className="source-sep"> · </span>
          <span className="source-pages">{pagesLabel}</span>
        </>
      )}
    </li>
  );
}
