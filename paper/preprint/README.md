# Preprint build

`LIP_PREPRINT_001.md` is the canonical, human-editable manuscript. The local
renderer creates the editorial preview at `output/pdf/LIP_PREPRINT_001.pdf`.
`LIP_PREPRINT_001.tex` is the publication-ready LaTeX mirror and must remain
scientifically synchronized with the canonical Markdown. Its verified Overleaf
build is archived at
`output/pdf/Receiver-Anchored_Tests_for_Latent_Communication_v0.1.pdf`; this is
the PDF intended for the Zenodo preprint deposit.

```powershell
python paper/preprint/render_preprint.py
```

Where a TeX distribution is available, compile the LaTeX mirror twice so that references settle:

```powershell
New-Item -ItemType Directory -Force output/latex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory output/latex paper/preprint/LIP_PREPRINT_001.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory output/latex paper/preprint/LIP_PREPRINT_001.tex
```

The two PDFs have distinct roles. `LIP_PREPRINT_001.pdf` is the ReportLab
editorial preview rendered from the canonical Markdown.
`Receiver-Anchored_Tests_for_Latent_Communication_v0.1.pdf` is the
Overleaf-compiled archival candidate rendered from the LaTeX mirror. Treat the
LaTeX file as a second publication source: any scientific, authorship,
disclosure, or licensing edit must be applied to both source files in the same
change.

The paper is intentionally bounded to `LIP-PROTO-013` and `LIP-PROTO-014`. `result_snapshot.json` records the exact paper-facing counts, effects, hashes, and unsupported claims. `CLAIM_LEDGER.md` is the editorial guardrail for abstracts, repository copy, DOI metadata, and public posts.

Licensing is deliberately separated: manuscript text and figures are CC BY
4.0; repository code is MIT. `zenodo_preprint_metadata.json` is for a manual
publication deposit. The root `.zenodo.json` describes a GitHub software
release and must not be reused as the preprint metadata.

The published Zenodo record for version 0.1 is `21943476`. Its registered
version DOI is `10.5281/zenodo.21943476`, and its concept DOI is
`10.5281/zenodo.21943475`. The version DOI is synchronized into both manuscript
sources and the verified archival PDF.

The version 0.1 deposit contains the archival PDF, its LaTeX source, and the
machine-readable `result_snapshot.json`. The canonical Markdown, claim ledger,
and license file remain available through the linked GitHub repository rather
than being duplicated in the Zenodo file bundle. The Zenodo `References` field
mirrors all 13 works cited in the manuscript; the repository alone is recorded
under `Related works` with the relation `Is supplemented by`.

For future archival versions:

1. review every scientific statement against the frozen protocols;
2. reserve the version DOI before compiling the archival PDF;
3. deposit the archival PDF and LaTeX source, and bind supporting code and
   claim-level artifacts through the linked public repository;
4. keep the versioned record immutable after publication; issue a new version for corrections.
