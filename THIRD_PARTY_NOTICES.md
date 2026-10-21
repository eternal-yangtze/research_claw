# Third-party code notices

ResearchOps includes small adapted portions of the following local reference projects:

- TaleClaw (`reference_coding/TaleClaw`), MIT License: `.env` loading design;
  copied `runtime/trace/workspace.py`, `applications/coding/handoff.py`, and
  adapted `evaluation/metrics.py` (removed its failure-enum dependency) into
  `src/researchops/vendor/taleclaw/`. The upstream MIT notice is bundled there.
  The coding application below is a local adapter using ResearchOps model/tools,
  not a claim that the complete upstream TaleClaw runtime has been copied.
- paper-search-mcp (`references2_structure/.../paper-search-mcp-main`), MIT License:
  arXiv query normalization, metadata mapping, pacing and PDF acquisition design;
  OpenAlex abstract/OA mapping, anonymous public OpenReview filtering, and DBLP metadata mapping.
- PaperSeek (`references2_structure/.../paperseek-main`), Apache License 2.0:
  source-specific query planning and bounded literature-query portfolio design.

The complete upstream license files remain in their respective reference directories.
ResearchOps adapters use their own internal data contracts and do not require Semantic Scholar.

Dataset preparation uses installed torchvision (BSD 3-Clause) for native dataset
readers and adapts the official URL/checksum/layout metadata from its CIFAR,
Flowers102, OxfordIIITPet and DTD modules. The upstream license is distributed with
torchvision; see https://github.com/pytorch/vision/blob/main/LICENSE. Individual
datasets have their own source terms and citations, recorded in each manifest.
