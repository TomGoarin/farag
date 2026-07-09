// Rendu d'une source. Composant isolé pour accueillir plus tard un
// éventuel `chunk_text` (repliable) sans refondre la liste.
export default function Source({ index, source }) {
  const { source_file, section_path, pages } = source;
  const hasSection = section_path && section_path.length > 0;
  const hasPages = Array.isArray(pages) && pages.length > 0;

  return (
    <li className="source">
      <span className="source-index">[{index}]</span>
      <span className="source-doc">{source_file}</span>
      {hasSection && (
        <>
          <span className="source-sep"> · </span>
          <span className="source-section">{section_path}</span>
        </>
      )}
      {hasPages && (
        <>
          <span className="source-sep"> · </span>
          <span className="source-pages">p. {pages.join(", ")}</span>
        </>
      )}
    </li>
  );
}
