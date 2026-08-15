# Preprint build

`LIP_PREPRINT_001.md` is the canonical, human-editable manuscript. The renderer creates one paper artifact at `output/pdf/LIP_PREPRINT_001.pdf`. `LIP_PREPRINT_001.tex` is the publication-ready LaTeX mirror and must remain scientifically synchronized with the canonical Markdown.

```powershell
python paper/preprint/render_preprint.py
```

Where a TeX distribution is available, compile the LaTeX mirror twice so that references settle:

```powershell
New-Item -ItemType Directory -Force output/latex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory output/latex paper/preprint/LIP_PREPRINT_001.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory output/latex paper/preprint/LIP_PREPRINT_001.tex
```

The checked-in PDF is rendered from the canonical Markdown. Treat the LaTeX file as a second publication source: any scientific, authorship, disclosure, or licensing edit must be applied to both files in the same change.

The paper is intentionally bounded to `LIP-PROTO-013` and `LIP-PROTO-014`. `result_snapshot.json` records the exact paper-facing counts, effects, hashes, and unsupported claims. `CLAIM_LEDGER.md` is the editorial guardrail for abstracts, repository copy, DOI metadata, and public posts.

Licensing is deliberately separated: manuscript text and figures are CC BY
4.0; repository code is MIT. `zenodo_preprint_metadata.json` is for a manual
publication deposit. The root `.zenodo.json` describes a GitHub software
release and must not be reused as the preprint metadata.

Before an archival upload:

1. review every scientific statement against the frozen protocols;
2. replace no metadata with a DOI until the DOI actually exists;
3. deposit the PDF together with the code commit and redistributable claim-level artifacts;
4. keep the versioned record immutable after publication; issue a new version for corrections.
