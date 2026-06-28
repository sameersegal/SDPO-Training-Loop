---
name: read-arxiv-paper
description: Use this skill when asked to read an arxiv paper given an arxiv URL
---

You will be given a URL of an arxiv paper, for example:

https://www.arxiv.org/abs/2601.07372

### Part 1: Normalize the URL

The goal is to fetch the TeX Source of the paper (not the PDF!), the URL always looks like this:

https://www.arxiv.org/src/2601.07372

Notice the /src/ in the url. Once you have the URL:

### Part 2: Download the paper source

Fetch the url to a local .tar.gz file. A good location is `~/.cache/sparkycoder/knowledge/{arxiv_id}.tar.gz`.

(If the file already exists, there is no need to re-download it).

### Part 3: Unpack the file in that folder

Unpack the contents into `~/.cache/sparkycoder/knowledge/{arxiv_id}` directory.

### Part 4: Locate the entrypoint

Every latex source usually has an entrypoint, such as `main.tex` or something like that.

### Part 5: Read the paper

Once you've found the entrypoint, Read the contents and then recurse through all other relevant source files to read the paper.

### Part 5b: Extract the ONE figure that matters (figure discipline)

The summary is a **reading document**, not a figure gallery. A wall of plots actively *hinders*
reading — the prose and any tables already carry the results. So be ruthless:

- **Default: embed exactly ONE figure** — the single image that lets a reader understand something
  *faster than the prose can say it*. Usually that's the **method/mechanism diagram** (what the paper
  *does*) or the one plot that *is* the paper's thesis (e.g. an entropy-collapse curve for a paper about
  collapse). Pick the "money figure."
- **Hard cap: two.** Use a second only when a mechanism diagram **and** one decisive result each add
  *distinct* understanding. If you're unsure, it's one.
- **Never embed:** multi-panel result grids, legend-only images, appendix or per-model duplicate panels,
  or any figure whose point a single sentence or an existing markdown table already makes. A result
  number belongs in prose/a table, not a screenshot of a bar chart.
- **Litmus test before embedding:** "Does this image make the reader *faster*, or is it decoration?" If
  decoration, cut it.

Extraction (do this **only for the 1–2 figures you will actually embed** — don't dump every figure to disk):

1. Find the figure's `\includegraphics[...]{path}` in the source and resolve `path` to a real file in
   `~/.cache/sparkycoder/knowledge/{arxiv_id}/` (extension is often omitted — `.pdf`, `.png`, `.jpg`,
   `.eps`). Skip logos/icons/`.../logo/...`/inline marks.
2. Name it `knowledge/images/{tag}_figure_{number}.{ext}` (create `knowledge/images/` if missing), using
   the **same `{tag}`** as the summary and the paper's own figure number. A composite figure stays ONE
   file — do **not** split it into per-panel `a`/`b`/… images for a summary.
   - **Raster (PNG/JPG/GIF/WebP/SVG):** copy as-is.
   - **PDF/EPS (vector — browsers can't show inline):** convert with poppler:
     `pdftoppm -png -r 150 -singlefile "<src>.pdf" "knowledge/images/{tag}_figure_{number}"`.
     The server (`src/knowledge_reports.py`) only serves `png/jpg/jpeg/gif/webp/svg`.
   - **TikZ/PGF figure (drawn in LaTeX — has a `\begin{tikzpicture}` / `pgfplots` and NO `\includegraphics`):**
     don't crop it out of the paper PDF (low quality). **Compile it** to get a crisp image (needs a LaTeX
     toolchain — `pdflatex` + `texlive-pictures` for tikz + `texlive-latex-extra` for the `standalone`
     class; ask the user to `sudo apt-get install` these if missing). Build a minimal `standalone` wrapper
     — do NOT reuse the paper's full preamble (it pulls in unrelated, often-missing packages like
     `algorithm`/`dsfont`). Instead take just: `\documentclass[border=10pt]{standalone}`,
     `\usepackage{amsmath,amssymb,amsfonts,bm}`, `\usepackage{xcolor}`, `\usepackage{tikz}` +
     **the figure's `\usetikzlibrary{…}` lines**, the paper's **`\definecolor{…}` lines** it uses, and any
     **`\newcommand` macros** it uses (copy whole lines — nested braces like `\KL{D_{\mathrm{KL}}}` break a
     naive regex). Then `\begin{document}` + the `tikzpicture` (only — not the `\caption`/`figure` wrapper)
     + `\end{document}`. Compile **in the paper's source dir** (so a custom `\usepackage{mylatexstyle}` etc.
     resolves) and rasterize at high DPI:
     `pdflatex -interaction=nonstopmode -halt-on-error fig.tex && pdftoppm -png -r 300 -singlefile fig.pdf "knowledge/images/{tag}_figure_{number}"`.
     Iterate on `Undefined control sequence` errors by adding the missing package/macro. Drop the temp
     `fig.{tex,pdf,aux,log}` afterward. (If no LaTeX toolchain is available and can't be installed, fall
     back to a high-DPI page render + crop, and note the lower quality.)
3. Embed it where you discuss it, image then one-line caption, **relative** `images/` path:
   ```
   ![Figure 2: self-teacher mechanism](images/{tag}_figure_2.png)
   *Figure 2 — the model re-scores its own rollout under feedback, then distills that back (from the paper).*
   ```
   Many strong summaries cite figures by number in prose **without** embedding them — that's fine and
   often better than another image.

### Part 6: Report

Once you've read the paper, produce a summary of the paper into a markdown file at `./knowledge/summary_{tag}.md`. Notice that 1) use the local knowledge directory here (it's easier for me to open and reference here), not in `~/.cache`, and 2) generate some reasonable `tag` like e.g. `conditional_memory` or whatever seems appropriate given the paper. Probably make sure that the tag doesn't exist yet so you're not overwriting files.

As for the summary itself, remember that you're processing this paper within the context of the sparkycoder repository, so most often we will be interested in how to apply the paper and its lessons to the sparkycoder project. Therefore, you should feel free to "remind yourself" of the related sparkycoder code by reading the relevant parts, and then explicitly make the connection of how this paper might relate to sparkycoder or what are things we might be inspired about or try.

### Math equation formatting (IMPORTANT — the summaries are read through a KaTeX viewer)

The `summary_*.md` reports are rendered by `src/knowledge_reports.py` (markdown via marked.js, math via **KaTeX**). Write math so it actually renders:

- **Delimiters:** inline math `$ ... $`, display math `$$ ... $$` (on its own line). `\( ... \)` and `\[ ... \]` also work. Keep each inline `$...$` span on a single line.
- **Only standard KaTeX commands.** Papers define their own macros in the LaTeX preamble (e.g. `\Acc`, `\E`, `\dkl`, `\sg`) via `\newcommand` — KaTeX does **not** know these. **Expand them yourself** when copying math: e.g. `\Acc` → `\mathrm{Acc}`, a custom `\E` → `\mathbb{E}`, `\sg[x]` → `\mathrm{sg}[x]`. If a symbol renders red/as an error, it's almost always an unexpanded custom macro. (See the [KaTeX supported-functions list](https://katex.org/docs/supported.html) when unsure.)
- **Don't write bare `$` for currency or lone numbers.** Inline math that starts with a digit (e.g. `$0.1$`, `$1.0$`) collides with currency detection — write such scalars as plain text (`0.1`, `1.0`) or include a non-digit lead (`$\beta=0.1$`). For real dollar amounts, escape as `\$5`.
- `$` **inside code spans/blocks is safe** — the viewer protects code, so a regex like `` `...proj$` `` won't be treated as math. You don't need to escape those.

Spot-check a new report in the viewer (`python src/knowledge_reports.py knowledge --serve`) — unrendered `$`…`$` or red error text means a delimiter or macro needs fixing.
