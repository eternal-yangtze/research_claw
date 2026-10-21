---
name: document-ingest
description: "Import existing local PDF, Markdown or text documents into the research knowledge base, retaining source hashes, pages, parser provenance and automatic bounded formula checks. Use for 导入论文、入库本地资料、建立文献库. Do not use to find or download new papers; use evidence-retrieve for an already indexed corpus."
---

# Ingest local research documents

1. Obtain an existing local file/directory path from the user or a previous acquisition artifact. Never invent a path. Require it to be inside a host-configured readable root.
2. Read the exact input schema from load_skill. Invoke `document-ingest` with the loaded hash and `arguments={"path": <absolute existing path>}`.
3. Let DocumentIngestor perform parsing, source hashing, chunk persistence and formula checks. Do not duplicate parser code or delete prior parser revisions.
4. Report document count, chunk count, parser, quality status, parsed artifact and formula status. A PyMuPDF fallback marked `not_assessed` does not establish correct formulas or their absence. Preserve unresolved and deferred formulas.
5. Stop with the host error on unsupported types, unreadable paths or parsing failure. Correct the input before retrying; do not expand path permissions yourself.

Output contract: the unchanged list of ingestion results in `result.output`; each includes `paper_id`, `chunks`, `parser`, `quality_status`, `path`, and optional `parsed_path`/`formula_validation`. A repeated import must not create duplicate scientific evidence.
