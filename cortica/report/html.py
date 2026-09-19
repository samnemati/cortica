"""Render a pipeline + dataset summary into a self-contained HTML report.

This first version is deliberately dependency-light (standard library only): it
captures the reproducible artifact — the ordered steps, their parameters, a dataset
summary, and a provenance block. Rich per-step figures (via ``mne.Report``) are a
later slice.
"""
from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from pathlib import Path

_CSS = """
  :root { color-scheme: light dark; }
  body { margin: 0; background: #f6f8fa; color: #14202b;
         font: 15px/1.5 system-ui, -apple-system, sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 32px 20px 56px; }
  h1 { margin: 0 0 2px; font-size: 26px; letter-spacing: -.01em; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
       color: #5a6b7b; border-bottom: 1px solid #dde4ea; padding-bottom: 6px;
       margin: 30px 0 12px; }
  .sub { color: #5a6b7b; margin: 0 0 8px; }
  dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 20px; margin: 0; }
  dt { color: #5a6b7b; } dd { margin: 0; font-variant-numeric: tabular-nums; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e6ecf1;
           vertical-align: top; }
  th { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #5a6b7b; }
  td.n { color: #8a99a8; font-variant-numeric: tabular-nums; width: 28px; }
  td.p { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 13px; }
  .id { color: #8a99a8; font-family: ui-monospace, monospace; font-size: 11px; }
  footer { margin-top: 34px; color: #8a99a8; font-size: 12px; }
  .figs { display: flex; flex-wrap: wrap; gap: 16px; }
  figure { margin: 0; }
  figure img { max-width: 100%; border: 1px solid #e6ecf1; border-radius: 8px; }
  figcaption { color: #5a6b7b; font-size: 12px; margin-top: 4px; }
  @media (prefers-color-scheme: dark) {
    body { background: #0e141b; color: #e7edf3; }
    h2 { color: #93a2b1; border-color: #26323f; }
    .sub, dt, td.n, .id, footer, th { color: #93a2b1; }
    th, td { border-color: #212d39; }
  }
"""


def _figures(dataset) -> list:
    """Return ``[(title, base64_png), ...]`` of result figures, or ``[]`` if
    figure libraries are unavailable or the payload can't produce them."""
    try:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from .. import viz
    except Exception:
        return []

    payload = getattr(dataset, "payload", None)
    if payload is None or not hasattr(payload, "get_data"):
        return []

    def encode(fig) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    figures = []
    try:
        freqs, psds = viz.spectrum(payload, fmax=45.0)
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        for row in psds:
            ax.plot(freqs, 10 * np.log10(np.maximum(row, 1e-30)), lw=0.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title("Power spectrum")
        figures.append(("Power spectrum", encode(fig)))
    except Exception:
        pass
    try:
        import mne

        if getattr(payload, "get_montage", lambda: None)() is not None:
            power = viz.band_power(payload, "Alpha")
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            mne.viz.plot_topomap(power, payload.info, axes=ax, show=False, cmap="RdBu_r")
            ax.set_title("Alpha power")
            figures.append(("Alpha head map", encode(fig)))
    except Exception:
        pass
    return figures


def build_report(dataset, pipeline, path: str, title: str = "Cortica report") -> str:
    """Write an HTML report for ``pipeline`` applied to ``dataset``; return the path."""
    import cortica.steps  # noqa: F401  register built-ins so step names resolve
    from cortica.core.registry import default_registry

    meta = dataset.meta
    sfreq = meta.get("sfreq")
    n_ch = meta.get("n_channels")
    payload = dataset.payload
    duration = payload.n_times / sfreq if (sfreq and hasattr(payload, "n_times")) else None

    pdict = pipeline.to_dict()
    version = pdict.get("cortica_version", "")
    modality = pdict.get("modality", dataset.modality)

    rows = []
    for i, pstep in enumerate(pipeline.steps, 1):
        try:
            name = default_registry.get(pstep.step_id).name
        except KeyError:
            name = pstep.step_id
        params = ", ".join(f"{k} = {v}" for k, v in pstep.params.items()) or "—"
        rows.append(
            f"<tr><td class='n'>{i}</td>"
            f"<td>{_html.escape(name)}<div class='id'>{_html.escape(pstep.step_id)}</div></td>"
            f"<td class='p'>{_html.escape(params)}</td></tr>"
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dur_txt = f"{duration:.1f} s" if duration is not None else "—"

    figures = _figures(dataset)
    figures_html = ""
    if figures:
        cards = "".join(
            f'<figure><img alt="{_html.escape(fig_title)}" '
            f'src="data:image/png;base64,{b64}">'
            f"<figcaption>{_html.escape(fig_title)}</figcaption></figure>"
            for fig_title, b64 in figures
        )
        figures_html = f'<section><h2>Results</h2><div class="figs">{cards}</div></section>'

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>{_CSS}</style></head>
<body><main>
  <header>
    <h1>{_html.escape(title)}</h1>
    <p class="sub">Reproducible report · generated {generated}</p>
  </header>
  <section>
    <h2>Dataset</h2>
    <dl>
      <dt>Modality</dt><dd>{_html.escape(str(modality))}</dd>
      <dt>Channels</dt><dd>{n_ch if n_ch is not None else "—"}</dd>
      <dt>Sampling rate</dt><dd>{sfreq if sfreq is not None else "—"} Hz</dd>
      <dt>Duration</dt><dd>{dur_txt}</dd>
    </dl>
  </section>
  {figures_html}
  <section>
    <h2>Pipeline — {len(pipeline.steps)} step(s)</h2>
    <table>
      <thead><tr><th>#</th><th>Step</th><th>Parameters</th></tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </section>
  <footer>Generated by Cortica {_html.escape(str(version))} · research use only.</footer>
</main></body></html>"""

    Path(path).write_text(doc, encoding="utf-8")
    return path
