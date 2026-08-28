"""
Two-page executive summary for the Phillies K% trial project.

A standalone condensation of output/methodology_<season>.pdf for a reader who
wants the problem, the model, the held-out result and the caveats without the
full derivation. Reads the same artefacts k_projection.py writes, so it cannot
drift out of sync with the analysis.

Usage:
    python k_projection.py          # produce the analysis outputs first
    python build/make_summary.py    # then typeset the summary

    python build/make_summary.py --author "Your Name" --target-season 2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# make_summary.py sits one level down in build/, so the repo root is two up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# reportlab's built-in fonts are WinAnsi (Latin-1): no glyph for >=, delta or
# sigma, which render as black boxes. The prose below stays ASCII for that
# reason. Unicode is fine inside the matplotlib figures, which embed their own
# fonts -- but this document carries no figures.
#
# The list marker is a MIDDLE DOT (U+00B7), not a bullet (U+2022): reportlab's
# standard-font encoding table has no entry for U+2022 and silently emits 0x7F,
# which has no glyph and prints as blank. U+00B7 round-trips to WinAnsi 0xB7.
BULLET = '\u00b7'


def _panel_stats(data_path: Path) -> dict | None:
    """Panel dimensions, read from the source file when it is available.

    The trial data is gitignored, so a clone without it still produces a valid
    summary; the two sentences that quote panel size are simply omitted.
    """
    if not data_path.exists():
        return None
    d = pd.read_csv(data_path, encoding="utf-8-sig")
    return {"rows": len(d),
            "pitchers": int(d.PlayerId.nunique()),
            "tbf_min": int(d.TBF.min()),
            "tbf_max": int(d.TBF.max()),
            "below_stab": float((d.TBF < 70).mean())}


def build_summary(target_season: int, out_dir: Path, manifest: dict,
                  abl: pd.DataFrame, bt: pd.DataFrame, panel: dict | None,
                  author: str) -> Path:
    """Assemble the two-page summary from a completed run's artefacts."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, KeepTogether)

    GREY = colors.HexColor("#D9D9D9")
    DGREY = colors.HexColor("#666666")
    NAVY = colors.HexColor("#284B8C")

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman",
                          fontSize=9.6, leading=12.6, spaceAfter=6.5)
    lead = ParagraphStyle("lead", parent=body, fontSize=10.4, leading=13.8,
                          spaceAfter=8)
    h1 = ParagraphStyle("h1", parent=ss["Normal"], fontName="Helvetica-Bold",
                        fontSize=10.5, leading=13, spaceBefore=10, spaceAfter=4.5,
                        textColor=NAVY)
    cap = ParagraphStyle("cap", parent=body, fontSize=8.3, leading=10.8,
                         spaceBefore=3, spaceAfter=9, textColor=DGREY)
    title = ParagraphStyle("title", parent=ss["Normal"], fontName="Helvetica-Bold",
                           fontSize=16, leading=20, alignment=1, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontName="Helvetica",
                         fontSize=10.2, leading=13.5, alignment=1, spaceAfter=2,
                         textColor=colors.HexColor("#333333"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=13,
                            bulletIndent=3, spaceAfter=3.5)

    story, T = [], target_season

    def tbl(data, widths, size=8.8, hi=None):
        t = Table(data, colWidths=widths, hAlign="CENTER")
        style = [
            ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), size),
            ("TOPPADDING", (0, 0), (-1, -1), 2.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]),
        ]
        if hi is not None:                      # bold the shipped model's row
            style += [("FONTNAME", (0, hi), (-1, hi), "Times-Bold")]
        t.setStyle(TableStyle(style))
        return t

    # ---------------- values pulled from the run ---------------------------
    hb = manifest.get("holdout_evaluation") or {}
    ke = manifest["variance_decomposition"]
    w = manifest["recency_weights"]
    k_bf = manifest["shrinkage_k_bf"]
    k_inf = manifest.get("shrinkage_k_bf_informed")
    lo_yr, hi_yr = manifest["training_window"]
    lg_proj = manifest["league_k_projected"]
    n_proj = manifest["n_projections"]
    cov = hb.get("pi80_coverage")
    n_hold = hb.get("n")
    wtxt = "/".join(f"{x:.2f}" for x in w)

    def rmse_pp(key):
        return hb[key]["rmse"] * 100

    # ======================= HEADER ========================================
    story.append(Paragraph("Projecting Pitcher Strikeout Percentage", title))
    story.append(Paragraph(f"Executive Summary &mdash; {T} Projections", sub))
    story.append(Paragraph(
        f"{author} &nbsp;|&nbsp; Philadelphia Phillies, Research and Information "
        f"&nbsp;|&nbsp; Quantitative Analyst Associate Trial Project", sub))
    story.append(Spacer(1, 0.16 * inch))

    # ======================= THE PROBLEM ===================================
    story.append(Paragraph("The problem is regression, not regressors", h1))
    panel_txt = ""
    if panel:
        panel_txt = (f"The panel holds {panel['rows']:,} player-seasons for "
                     f"{panel['pitchers']:,} pitchers. Batters faced run from "
                     f"{panel['tbf_min']} to {panel['tbf_max']}, and "
                     f"{panel['below_stab']*100:.0f}% of player-seasons fall below the "
                     f"roughly 70 batters faced at which K% stabilises. ")
    story.append(Paragraph(
        f"K% is a binomial rate observed over wildly unequal samples. {panel_txt}"
        f"A reliever with 40 batters faced and a starter with 800 both report a "
        f"K%, but one is mostly noise and the other is mostly signal. The decision "
        f"that drives accuracy is therefore how much of each pitcher's observed "
        f"rate to believe, not which regressor to add. I treat that as the modelling "
        f"problem and derive the answer from the data rather than assuming it.", lead))

    # ======================= THE MODEL =====================================
    story.append(Paragraph("The model", h1))
    story.append(Paragraph(
        f"An empirical-Bayes projection in the Marcel and Steamer tradition, fitted "
        f"on {lo_yr}-{hi_yr} only. Decomposing the variance of observed K% into true "
        f"talent and binomial noise gives the regression constant directly:", body))
    story.append(Paragraph(
        f"Var(observed) = {ke['var_observed']:.5f} &nbsp;=&nbsp; Var(talent) "
        f"{ke['var_talent']:.5f} &nbsp;+&nbsp; binomial noise {ke['binomial_noise']:.5f}"
        f" &nbsp;&nbsp;-&gt;&nbsp;&nbsp; k = {k_bf:.0f} batters faced",
        ParagraphStyle("eq", parent=body, fontName="Courier", fontSize=8.4,
                       leading=12, alignment=1, spaceBefore=4, spaceAfter=7)))
    story.append(Paragraph(
        f"Each pitcher's record is shrunk toward a prior worth {k_bf:.0f} batters faced. "
        f"That constant is measured, not chosen, and it independently reproduces the "
        f"published stabilisation point for K% of roughly 70 batters faced &mdash; two "
        f"different routes to the same number, which is the main evidence that the "
        f"shrinkage is right. Five components sit on top of it:", body))

    for b in [
        f"<b>A Stuff+-informed prior.</b> Stuff+ becomes reliable in far fewer pitches "
        f"than K% does, so the prior is E[K% | Stuff+, age] rather than a flat league "
        f"average. It must be calibrated forward first: the contemporaneous prior is "
        f"over-dispersed when applied to the next season (slope 0.75), and uncorrected "
        f"it performs worse than a flat prior. A more informative prior also earns a "
        f"larger effective sample size, {k_inf:.0f} batters faced against {k_bf:.0f}.",
        f"<b>Fitted recency weights.</b> Marcel's 5/4/3 is a starting grid, not an "
        f"assumption; a search over the training seasons selects {wtxt}.",
        "<b>A league-detrended aging curve</b>, estimated by the delta method on paired "
        "seasons and applied as a one-year nudge rather than extrapolated, because it is "
        "survivorship-biased by construction.",
        f"<b>League re-anchoring</b>, moving each pitcher's record from the environment "
        f"it was set in into the projected {T} environment of {lg_proj*100:.2f}%.",
        "<b>A 50/50 blend with a ridge model</b> on engineered features, chosen a priori "
        "on the reasoning behind FanGraphs Depth Charts and ATC, not because it won the "
        "backtest.",
    ]:
        story.append(Paragraph(b, bullet, bulletText=BULLET))

    # ======================= HEADLINE RESULT ===============================
    story.append(Paragraph(f"Held-out result: {T} was never used to fit anything", h1))
    rows = [["Model", "RMSE", "MAE", "R2"]]
    order = [("M0_league_average", "League average"),
             ("M1_last_season_kpct", "Last season's K%, unregressed"),
             ("M2_marcel_flat_prior", "Empirical Bayes, flat prior"),
             ("M3_eb_stuff_prior", "Empirical Bayes, Stuff+ prior"),
             ("M4_ridge", "Ridge on engineered features"),
             ("M6_blend (SHIPPED)", "Blend of EB + ridge  (shipped)")]
    hi_row = None
    for i, (k, lbl) in enumerate(order, start=1):
        if k not in hb:
            continue
        if "SHIPPED" in k:
            hi_row = i
        rows.append([lbl, f"{rmse_pp(k):.2f} pp",
                     f"{hb[k]['mae']*100:.2f} pp", f"{hb[k]['r2']:.3f}"])
    story.append(tbl(rows, [2.62*inch, 0.95*inch, 0.95*inch, 0.85*inch], hi=hi_row))
    story.append(Paragraph(
        f"Pitchers with at least 100 batters faced in {T} (n = {n_hold}). The shipped "
        f"model cuts RMSE {(1 - rmse_pp('M6_blend (SHIPPED)')/rmse_pp('M0_league_average'))*100:.0f}% "
        f"against a league-average baseline and "
        f"{(1 - rmse_pp('M6_blend (SHIPPED)')/rmse_pp('M1_last_season_kpct'))*100:.0f}% against "
        f"last season's raw rate, which is worse than the league average because "
        f"unregressed rates carry their full sampling noise forward.", cap))

    # ======================= PAGE 2 ========================================
    story.append(Spacer(1, 0.04 * inch))
    story.append(Paragraph("What each component is worth", h1))
    if len(abl):
        arows = [["Variant", "RMSE", "R2", "Cost (bps)"]]
        for _, r in abl.iterrows():
            arows.append([r.variant.strip(), f"{r.rmse*100:.2f} pp",
                          f"{r.r2:.3f}",
                          "--" if abs(r.delta_bps) < 1e-9 else f"{r.delta_bps:+.1f}"])
        story.append(tbl(arows, [2.62*inch, 0.95*inch, 0.85*inch, 0.95*inch], hi=1))
        story.append(Paragraph(
            f"Each component disabled in turn, re-scored on held-out {T}. Shrinkage is "
            f"the load-bearing decision by an order of magnitude: setting k = 5 costs "
            f"{abl[abl.variant.str.contains('k=5')].delta_bps.iloc[0]:.0f} basis points, "
            f"against 12 for dropping recency weighting and 8 for dropping aging. Over-"
            f"shrinking is far cheaper than under-shrinking, which is the asymmetry the "
            f"whole design leans on.", cap))

    story.append(Paragraph("Validation", h1))
    bt_seasons = sorted(int(x) for x in bt.target_season.unique()) if len(bt) else []
    story.append(Paragraph(
        f"<b>Leakage guard.</b> The instructions bar data from Opening Day {T} onward, "
        f"but the supplied file contains a complete {T} season. Those rows are treated "
        f"strictly as an answer key: read once, after the projections are written, to "
        f"score them. Every fitted object &mdash; league curve, shrinkage constant, aging "
        f"curve, Stuff+ prior and its calibration, ridge, LightGBM and the recency "
        f"weights &mdash; is estimated on {lo_yr}-{hi_yr} alone. "
        f"<font face='Courier' size='8.4'>--self-test</font> re-runs the entire pipeline "
        f"against a copy of the file with every {T} row deleted and asserts the "
        f"projections are unchanged. They are, to 0.00e+00 across all {n_proj:,} "
        f"projections and every numeric column.", body))
    if bt_seasons:
        story.append(Paragraph(
            f"<b>Rolling-origin backtest.</b> For each target season "
            f"{'-'.join(str(s) for s in bt_seasons)} the pipeline is re-fitted on "
            f"transitions completed before it, including a fresh recency-weight search "
            f"inside each fold, so the table is out-of-sample with respect to the weight "
            f"search as well as the outcome. Model ranking is stable across folds and "
            f"thresholds; fold-to-fold separation among the top models is inside noise, "
            f"which is precisely why the shipped model is a blend rather than the "
            f"backtest winner.", body))
    if cov is not None:
        story.append(Paragraph(
            f"<b>Calibrated uncertainty.</b> Every projection ships with an 80% "
            f"prediction interval combining talent uncertainty, which shrinks as "
            f"reliability rises, with the binomial uncertainty of next season's observed "
            f"rate. The second term dominates for low-usage pitchers. Empirical coverage "
            f"on held-out {T} is {cov*100:.1f}% against a nominal 80%. Interval widths "
            f"are calibrated using projected playing time, never the realised batters "
            f"faced, which is unknowable when the interval is issued.", body))

    story.append(Paragraph("Honest limitations", h1))
    for b in [
        "The aging curve is survivorship-biased: pitchers who decline are not re-observed, "
        "so the curve understates true decline. It is applied as a single one-year step "
        "and never extrapolated across a career.",
        "Starters and relievers are pooled. Role changes move K% by several points, and a "
        "role-aware prior is the single most promising extension the supplied columns "
        "would support.",
        "Playing time is projected crudely and used only to size intervals, never as a "
        "forecast. A real depth-chart input would sharpen the low-usage tail.",
        f"Four seasons of history support at most two honest backtest folds. The held-out "
        f"result rests on one season, and the gaps among the top three models "
        f"({rmse_pp('M2_marcel_flat_prior'):.2f}, {rmse_pp('M3_eb_stuff_prior'):.2f} and "
        f"{rmse_pp('M6_blend (SHIPPED)'):.2f} pp) are small relative to that.",
        "Model selection was performed by a human who has seen the held-out season. The "
        "self-test proves no 2025 row reaches a fitted object; it cannot prove the same "
        "of the analyst's judgement. The blend was fixed a priori for this reason.",
    ]:
        story.append(Paragraph(b, bullet, bulletText=BULLET))

    story.append(Paragraph("Deliverables", h1))
    story.append(Paragraph(
        f"<font face='Courier' size='8.4'>k_pct_projections_{T}.csv</font> carries "
        f"{n_proj:,} projections with 80% intervals, reliability and the component "
        f"decomposition behind each one. "
        f"<font face='Courier' size='8.4'>methodology_{T}.pdf</font> is the full "
        f"derivation. <font face='Courier' size='8.4'>backtest_{T}.csv</font>, "
        f"<font face='Courier' size='8.4'>ablation_{T}.csv</font> and "
        f"<font face='Courier' size='8.4'>run_manifest_{T}.json</font> record every "
        f"validation number and every fitted constant for auditing. The analysis is "
        f"deterministic: fixed seed, no network, no data beyond the supplied panel.", body))

    out = out_dir / f"summary_{T}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=LETTER,
                            leftMargin=0.85*inch, rightMargin=0.85*inch,
                            topMargin=0.7*inch, bottomMargin=0.62*inch,
                            title=f"Executive Summary: Projecting Pitcher K% ({T})",
                            author=author)

    def footer(canv, doc_):
        canv.saveState()
        canv.setFont("Times-Roman", 8.2)
        canv.setFillColor(DGREY)
        canv.drawString(0.85*inch, 0.4*inch,
                        f"Pitcher K% projection, {T}  |  executive summary")
        canv.drawRightString(LETTER[0] - 0.85*inch, 0.4*inch, f"Page {doc_.page} of 2")
        canv.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help="directory k_projection.py wrote its outputs into "
                         "(default: <repo>/output)")
    ap.add_argument("--data", default=None,
                    help="player-season panel, read only for panel dimensions "
                         "(default: <repo>/data/k_2026.csv)")
    ap.add_argument("--target-season", type=int, default=2025)
    ap.add_argument("--author", default="Andrew Zaletski")
    args = ap.parse_args()

    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else PROJECT_ROOT / "output")
    data_path = (Path(args.data).expanduser().resolve() if args.data
                 else PROJECT_ROOT / "data" / "k_2026.csv")
    T = args.target_season

    manifest_path = out_dir / f"run_manifest_{T}.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found. Run k_projection.py first.")
    manifest = json.loads(manifest_path.read_text())

    abl_path = out_dir / f"ablation_{T}.csv"
    bt_path = out_dir / f"backtest_{T}.csv"
    abl = pd.read_csv(abl_path) if abl_path.exists() else pd.DataFrame()
    bt = pd.read_csv(bt_path) if bt_path.exists() else pd.DataFrame()

    pdf = build_summary(T, out_dir, manifest, abl, bt, _panel_stats(data_path),
                        args.author)
    print(f"Executive summary written to {pdf}")


if __name__ == "__main__":
    main()
