from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import src.pipeline.data_integration as data_integration
import src.utils.file_management as filemgmt


# ── Colour palette ───────────────────────────────────────────────────────────
_BABYBLUE = "#B8D4F0"   # gender: Male
_PINK     = "#FF85B3"   # gender: Female
_RED      = "#C00000"   # hand: Right  (intentionally bold — majority)
_DKBLUE   = "#1F4E79"   # hand: Left
_SALMON   = "#FF9999"   # trials: Silence

# ── Per-subplot base colours (kept clearly distinct from bars above) ──────────
_COLOR_SUBJECT  = "white" # "#4472C4"   # blue   — subject-level trait scores
_COLOR_TRIAL    = "white" # "#70AD47"   # green  — trial-level subjective scores
_COLOR_CMC      = "white" # "#E8743B"   # orange — CMC coherence values
_COLOR_ACCURACY = "white" #"#7B2D8B"   # purple — RMS error

# ── Bar colour mappings ───────────────────────────────────────────────────────
_GENDER_COLORS = {"Male": _BABYBLUE, "Female": _PINK}
_HAND_COLORS   = {"Right": _RED,     "Left":   _DKBLUE}
_TRIAL_COLORS  = {"Music": _COLOR_SUBJECT, "Silence": _SALMON}


# ── Global font scaling ───────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 11})


def _compute_age_years(birthdate_value: object, reference_date: date) -> float:
    """Compute integer age in years from a birthdate string; returns NaN on parse failure."""
    if pd.Series([birthdate_value]).isna().iloc[0]:
        return np.nan

    birthdate_text = str(birthdate_value).strip()
    if not birthdate_text:
        return np.nan

    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            born = datetime.strptime(birthdate_text, fmt).date()
            break
        except ValueError:
            born = None
    if born is None:
        parsed = pd.to_datetime(birthdate_text, dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            return np.nan
        born = parsed.date()

    age_years = (
        reference_date.year
        - born.year
        - ((reference_date.month, reference_date.day) < (born.month, born.day))
    )
    return float(age_years)


def _summarize_numeric_series(series: pd.Series) -> dict[str, float]:
    """Return n/range/mean/median/std summary for a numeric series (NaN-safe)."""
    numeric_series = pd.Series(pd.to_numeric(pd.Series(series), errors="coerce"))
    values = numeric_series.dropna()
    if values.empty:
        return {
            "n": 0.0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }

    return {
        "n": float(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
    }


def _print_numeric_summary_line(metric_name: str, series: pd.Series, decimals: int = 3) -> None:
    """Print one formatted summary line for a metric."""
    stats = _summarize_numeric_series(series)
    n = int(stats["n"])
    if n == 0:
        print(f"  {metric_name:<28} n=0   range=[nan, nan]   mean=nan   median=nan   sd=nan")
        return

    fmt = f"{{:.{decimals}f}}"
    print(
        f"  {metric_name:<28} n={n:<3} "
        f"range=[{fmt.format(stats['min'])}, {fmt.format(stats['max'])}]   "
        f"mean={fmt.format(stats['mean'])}   "
        f"median={fmt.format(stats['median'])}   "
        f"sd={fmt.format(stats['std'])}"
    )


def _print_grouped_metric_summary(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    title: str,
    decimals: int = 3,
) -> None:
    """Print per-category summary stats for one metric column."""
    if group_col not in df.columns:
        print(f"\n── {title} ─────────────────────────────────────")
        print(f"  skipped: missing column '{group_col}'")
        return
    if metric_col not in df.columns:
        print(f"\n── {title} ─────────────────────────────────────")
        print(f"  skipped: missing column '{metric_col}'")
        return

    grouped = df[[group_col, metric_col]].copy()
    grouped[metric_col] = pd.to_numeric(grouped[metric_col], errors="coerce")
    grouped = grouped.dropna(subset=[group_col, metric_col])

    print(f"\n── {title} ─────────────────────────────────────")
    if grouped.empty:
        print("  no valid data")
        return

    for category in sorted(grouped[group_col].astype(str).unique()):
        category_values = pd.Series(
            grouped.loc[grouped[group_col].astype(str) == category, metric_col]
        )
        _print_numeric_summary_line(f"{category}", category_values, decimals=decimals)


def _resolve_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    context_name: str,
) -> str:
    """Return the first existing column from candidates, else raise a clear error."""
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    raise KeyError(
        f"Missing column for {context_name}. Tried: {', '.join(candidates)}"
    )


def _print_pearson_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    decimals: int = 3,
) -> None:
    """Print Pearson correlation test for two numeric columns with robust guardrails."""
    print(f"\n── {title} ─────────────────────────────────────────")

    if x_col not in df.columns or y_col not in df.columns:
        print(f"  skipped: missing column(s) '{x_col}' / '{y_col}'")
        return

    pair_df = df[[x_col, y_col]].copy()
    pair_df[x_col] = pd.to_numeric(pair_df[x_col], errors="coerce")
    pair_df[y_col] = pd.to_numeric(pair_df[y_col], errors="coerce")
    pair_df = pair_df.dropna(subset=[x_col, y_col])

    n = len(pair_df)
    if n < 3:
        print(f"  skipped: insufficient valid pairs (n={n}, need >= 3)")
        return

    if pair_df[x_col].nunique() <= 1 or pair_df[y_col].nunique() <= 1:
        print("  skipped: Pearson undefined (at least one variable is constant)")
        return

    r_val, p_val = scipy_stats.pearsonr(pair_df[x_col], pair_df[y_col])
    fmt = f"{{:.{decimals}f}}"
    print(
        f"  {x_col} vs {y_col}: "
        f"r={fmt.format(r_val)}, p={fmt.format(p_val)}, n={n}"
    )

# ════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _compute_scatter_x(
    data: np.ndarray,
    center: float,
    spread: float = 0.12,
    max_spread_group_size: int = 10,
        jitter: float = .0,
) -> np.ndarray:
    """
    Compute x-positions for scatter overlay on a boxplot.

    Uses tie-aware symmetric spreading when the largest group of tied values
    contains at most `max_spread_group_size` points; falls back to seeded
    uniform random jitter for denser data.

    Parameters
    ----------
    data : np.ndarray
        1-D NaN-free numeric array.
    center : float
        Base x-coordinate of the corresponding boxplot.
    spread : float
        Half-width of the horizontal scatter zone.
    max_spread_group_size : int
        Tie groups larger than this threshold trigger random jitter.

    Returns
    -------
    np.ndarray
        X-positions, same length as `data`.
    """
    if jitter > 0:
        rng = np.random.default_rng(seed=42)
        return rng.uniform(center - jitter, center + jitter, size=len(data))

    groups: dict[float, list[int]] = defaultdict(list)
    for i, v in enumerate(data):
        groups[round(float(v), 6)].append(i)

    if max(len(g) for g in groups.values()) <= max_spread_group_size:
        x_pos = np.full(len(data), center, dtype=float)
        for indices in groups.values():
            n = len(indices)
            if n == 1:
                continue
            offsets = np.linspace(-spread * (n - 1) / 2, spread * (n - 1) / 2, n)
            for local_i, global_i in enumerate(indices):
                x_pos[global_i] = center + offsets[local_i]
        return x_pos
    else:
        rng = np.random.default_rng(seed=42)
        return rng.uniform(center - spread, center + spread, size=len(data))


def _draw_stacked_bar(
    ax: plt.Axes,
    counts: dict,
    colors: dict,
    title: str,
    value_format: str = ".0f",
    text_color: str = "white",
) -> None:
    """
    Draw one horizontal stacked bar with one rectangle per category.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    counts : dict
        Ordered mapping of {category_label: numeric_count}.
    colors : dict
        Mapping of {category_label: colour_string}.
    title : str
        Axes title.
    value_format : str, optional
        Format spec for the count annotation. Default '.0f' (integer).
    text_color : str, optional
        Annotation text colour. Default 'white'; pass 'black' for light bars.
    """
    total = sum(counts.values())
    left  = 0.0
    for label, count in counts.items():
        ax.barh(
            0, count, left=left,
            color=colors.get(label, "#CCCCCC"),
            edgecolor="white", linewidth=2, height=0.6,
        )
        ax.text(
            left + count / 2, 0,
            f"{count:{value_format}}\n{label}",
            ha="center", va="center", fontsize=9, color=text_color,
            fontweight="bold",
        )
        left += count
    ax.set_xlim(0, total)
    ax.set_ylim(-0.65, 0.65)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=6)



def _draw_multi_boxplot(
    ax: plt.Axes,
    panels: list[tuple[str, pd.Series]],
    color: str,
    ylabel: str = "",
        boxwidth: float = 0.8,
    alpha_range: tuple[float, float] = (0.45, 0.85),
    max_spread_group_size: int = 10,
    subject_metadata: pd.DataFrame | None = None,
    # subject_metadata: DataFrame indexed by Subject ID,
    # must contain 'Gender' and 'Dominant hand' columns
        jitter: float = 0.0,
) -> None:
    _HAND_COLOR  = {"Right": "#C00000", "Left": "#1F4E79"}
    _GENDER_MARK = {"Male": "^", "Female": "o"}
    _FALLBACK_COLOR  = "#1A1A2E"
    _FALLBACK_MARKER = "o"

    n = len(panels)
    alphas = (np.linspace(alpha_range[0], alpha_range[1], n)
              if n > 1 else [sum(alpha_range) / 2])

    legend_handles: dict[str, plt.Artist] = {}

    for pos, ((title, series), alpha) in enumerate(zip(panels, alphas), start=1):
        data       = series.dropna()
        values     = data.values
        subj_ids   = data.index      # Series index must be Subject ID

        ax.boxplot(
            values,
            positions=[pos],
            patch_artist=True,
            widths=boxwidth,
            showfliers=False,
            medianprops=dict(color="#E74C3C", linewidth=2.5),
            boxprops=dict(facecolor=color, alpha=alpha),
            whiskerprops=dict(linewidth=1.2, color="#555555"),
            capprops=dict(linewidth=1.2, color="#555555"),
        )

        x_scatter = _compute_scatter_x(
            values, center=pos, max_spread_group_size=max_spread_group_size, jitter=jitter,
        )

        if subject_metadata is not None:
            # Group by (hand, gender) to batch scatter calls — one call per combo
            for (hand, gender), group_idx in (
                pd.Series(subj_ids)
                .to_frame("sid")
                .assign(
                    hand   = lambda d: d["sid"].map(subject_metadata["Dominant hand"]),
                    gender = lambda d: d["sid"].map(subject_metadata["Gender"]),
                )
                .groupby(["hand", "gender"]).groups.items()
            ):
                c  = _HAND_COLOR.get(hand,   _FALLBACK_COLOR)
                mk = _GENDER_MARK.get(gender, _FALLBACK_MARKER)
                sc = ax.scatter(
                    x_scatter[group_idx], values[group_idx],
                    alpha=0.75, color=c, marker=mk, s=32,
                    zorder=3, edgecolors="white", linewidths=0.4,
                )
                # Collect one handle per unique combo for the legend
                legend_key = f"{gender} / {hand}"
                if legend_key not in legend_handles:
                    legend_handles[legend_key] = plt.scatter(
                        [], [], color=c, marker=mk, s=32,
                        label=legend_key, edgecolors="white", linewidths=0.4,
                    )
        else:
            ax.scatter(x_scatter, values, alpha=0.55,
                       color=_FALLBACK_COLOR, s=22, zorder=3)

    ax.set_xlim(0.35, n + 0.65)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([p[0] for p in panels], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("black")
    ax.tick_params(axis="both", colors="black")
    ax.yaxis.label.set_color("black")

    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            fontsize=7, loc="best", framealpha=0.7,
            handletextpad=0.4, borderpad=0.5,
        )



# ════════════════════════════════════════════════════════════════════════════
# PUBLIC MOSAIC FIGURE
# ════════════════════════════════════════════════════════════════════════════

def plot_combined_descriptive_mosaic(
    personal_df: pd.DataFrame,
    musical_skill_series: pd.Series,
    athleticism_series: pd.Series,
    dancing_habit_series: pd.Series,
    fatigue_series: pd.Series,
    pleasure_series: pd.Series,
    stats_df: pd.DataFrame,
    liking_series: pd.Series,
    familiarity_series: pd.Series,
    cmc_flexor_beta_series: pd.Series,
    cmc_flexor_gamma_series: pd.Series,
    cmc_extensor_beta_series: pd.Series,
    cmc_extensor_gamma_series: pd.Series,
    accuracy_series: pd.Series,
    save_path: Path | None = None,
        suptitle: str | None = None,
        subject_metadata: pd.DataFrame | None = None,

) -> plt.Figure:
    """
    Render a single slide-ready figure combining all descriptive statistics.

    Layout:
      Row 1 — Participant characteristics:
        left  : Gender bar (top) + Dominant Hand bar (bottom)
        right : Musical Skill / Athleticism / Dancing Habit /
                Total Fatigue / Total Pleasure boxplots
      Row 2 — Study results:
        col 1 : Liking + Familiarity boxplots
        col 2 : CMC Flexor β / γ + Extensor β / γ boxplots
        col 3 : Task Accuracy boxplot

    Box colour encodes measurement type:
      blue   — subject-level trait scores
      green  — trial-level subjective scores
      orange — CMC coherence values
      purple — RMS error

    Parameters
    ----------
    personal_df : pd.DataFrame
        One row per subject; must contain 'Gender' and 'Dominant hand'.
    musical_skill_series, athleticism_series, dancing_habit_series,
    fatigue_series, pleasure_series : pd.Series
        Subject-level trait / post-study scores.
    stats_df : pd.DataFrame
        Full 1-segment statistics frame.
    liking_series, familiarity_series : pd.Series
        Subjective scores [0-7], music trials only.
    cmc_flexor_beta_series, cmc_flexor_gamma_series,
    cmc_extensor_beta_series, cmc_extensor_gamma_series : pd.Series
        CMC coherence values.
    accuracy_series : pd.Series
        RMS task accuracy values.
    save_path : Path, optional
        Destination file; saved at 150 dpi when provided.

    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=(13, 8))
    fig.subplots_adjust(top=0.92, bottom=0.09, left=0.06, right=0.98,
                        hspace=0.20)

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.0, 1.05],
        hspace=0.20,
    )

    # ── ROW 1: personal characteristics ─────────────────────────────────────
    row1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[2.2, 7.8],
        wspace=0.15,
    )
    row1_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=row1[0], hspace=0.60,
    )
    ax_gender = fig.add_subplot(row1_left[0])
    ax_hand   = fig.add_subplot(row1_left[1])
    ax_traits = fig.add_subplot(row1[1])

    # ── ROW 2: ratings + CMC + accuracy ──────────────────────────────────────
    row2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        width_ratios=[3.0, 5.5, 1.5],
        wspace=0.45,
    )
    ax_ratings = fig.add_subplot(row2[0])
    ax_cmc     = fig.add_subplot(row2[1])
    ax_acc     = fig.add_subplot(row2[2])

    # ── DRAW ROW 1 ───────────────────────────────────────────────────────────
    ordered_gender = {k: v for k, v in personal_df["Gender"].value_counts().items()
                      if k in _GENDER_COLORS}
    _draw_stacked_bar(ax_gender, ordered_gender, _GENDER_COLORS, "Gender", text_color='black')

    ordered_hand = {k: v for k, v in personal_df["Dominant hand"].value_counts().items()
                    if k in _HAND_COLORS}
    _draw_stacked_bar(ax_hand, ordered_hand, _HAND_COLORS, "Dominant Hand")

    # five subject-level scores on one shared blue axis
    _draw_multi_boxplot(ax_traits, [
        ("Musical Skill",  musical_skill_series),
        ("Athleticism",    athleticism_series),
        ("Dancing Habit",  dancing_habit_series),
        ("Total Fatigue",   fatigue_series),
        ("Total Pleasure",  pleasure_series),
    ], color=_COLOR_SUBJECT, ylabel="Score [0-7]", boxwidth=.5,  # width slightly reduced here, since subplot is wider
                        subject_metadata=subject_metadata)

    # ── DRAW ROW 2 ───────────────────────────────────────────────────────────
    _draw_multi_boxplot(ax_ratings, [
        ("Liking",      liking_series),
        ("Familiarity", familiarity_series),
    ], color=_COLOR_TRIAL, ylabel="Score [0-7]", subject_metadata=subject_metadata)

    _draw_multi_boxplot(ax_cmc, [
        ("Flexor β", cmc_flexor_beta_series),
        ("Flexor γ", cmc_flexor_gamma_series),
        ("Extensor β", cmc_extensor_beta_series),
        ("Extensor γ", cmc_extensor_gamma_series),
    ], color=_COLOR_CMC, ylabel="Coherence", subject_metadata=subject_metadata,
                        jitter=.05,
                        # add jitter here, because continuous y-axis prevents coinciding samples (no width spreading)
                        )

    _draw_multi_boxplot(ax_acc, [
        ("Accuracy", accuracy_series),
    ], color=_COLOR_ACCURACY, ylabel="Task Error (RMSE)", subject_metadata=subject_metadata,
                        jitter=.1,
                        # same jitter argumentation here
                        )

    # ── SECTION LABELS ───────────────────────────────────────────────────────
    for y_fig, label in [(0.955, "Participant Characteristics"),
                         (0.495, "Study Results")]:
        fig.text(0.05, y_fig, label, fontsize=10, color="black",
                 va="bottom", style="normal")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=13, y=0.99)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig




# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    ROOT                = Path().resolve().parent
    DATA                = ROOT / "data"
    OUTPUT              = ROOT / "output"
    EXPERIMENT_DATA     = DATA / "experiment_results"
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"
    DESCRIPTIVE_OUTPUT  = OUTPUT / "descriptive_statistics"
    filemgmt.assert_dir(DESCRIPTIVE_OUTPUT)

    N_SUBJECTS: int = 12

    # ── Personal data ────────────────────────────────────────────────────────
    personal_records: list[dict] = []
    for subject_ind in range(N_SUBJECTS):
        subject_dir   = EXPERIMENT_DATA / f"subject_{subject_ind:02}"
        personal_data = data_integration.fetch_personal_data(
            subject_dir, include_name_and_birthdate=True,
        )
        personal_records.append({
            "Subject ID":    subject_ind,
            "Birthdate":     personal_data.get("Birthdate"),
            "Gender":        personal_data["Gender"],
            "Dominant hand": personal_data["Dominant hand"],
            "Listening habit [0-3]": personal_data["Listening habit [0-3]"],
            "Musical skill": personal_data["Musical skill"],
            "Athleticism":   personal_data["Athleticism"],
            "Dancing habit": personal_data["Dancing habit"],
            "Total fatigue": personal_data["Total fatigue"],
            "Total pleasure": personal_data["Total pleasure"],
        })

    personal_df          = pd.DataFrame(personal_records)
    # subject_metadata indexed by Subject ID
    subject_metadata = personal_df.set_index("Subject ID")[["Gender", "Dominant hand"]]

    # Subject-level series — index them by Subject ID
    musical_skill_series = personal_df.set_index("Subject ID")["Musical skill"]
    listening_habit_series = personal_df.set_index("Subject ID")["Listening habit [0-3]"]
    athleticism_series = personal_df.set_index("Subject ID")["Athleticism"]
    dancing_habit_series = personal_df.set_index("Subject ID")["Dancing habit"]
    fatigue_series = personal_df.set_index("Subject ID")["Total fatigue"]
    pleasure_series = personal_df.set_index("Subject ID")["Total pleasure"]
    age_series = personal_df["Birthdate"].apply(
        lambda value: _compute_age_years(value, reference_date=date.today())
    )

    print("\n── Personal data ───────────────────────────────────────────")
    print(personal_df.to_string(index=False))

    # ── Study structure ──────────────────────────────────────────────────────
    stats_df: pd.DataFrame = pd.read_csv(
        filemgmt.most_recent_file(FEATURE_OUTPUT_DATA, ".csv", ["Combined Statistics 1seg"])
    )

    # Trial-level series — index by Subject ID so each row maps to a subject
    cmc_flexor_beta_series = stats_df.set_index("Subject ID")["CMC_Flexor_mean_beta"].dropna()
    cmc_flexor_gamma_series = stats_df.set_index("Subject ID")["CMC_Flexor_mean_gamma"].dropna()
    cmc_extensor_beta_series = stats_df.set_index("Subject ID")["CMC_Extensor_mean_beta"].dropna()
    cmc_extensor_gamma_series = stats_df.set_index("Subject ID")["CMC_Extensor_mean_gamma"].dropna()

    music_df = stats_df.loc[stats_df["Music Listening"].astype(bool)]
    liking_col = _resolve_existing_column(
        music_df,
        candidates=["Liking [0-7]", "Liking"],
        context_name="liking",
    )
    familiarity_col = _resolve_existing_column(
        music_df,
        candidates=["Familiarity [0-7]", "Familiarity"],
        context_name="familiarity",
    )
    perceived_category_col = _resolve_existing_column(
        music_df,
        candidates=["Perceived Category", "Category or Silence"],
        context_name="perceived category",
    )

    liking_series = music_df.set_index("Subject ID")[liking_col].dropna()
    familiarity_series = music_df.set_index("Subject ID")[familiarity_col].dropna()
    accuracy_series = stats_df.set_index("Subject ID")["RMS_Accuracy"].dropna()

    """musical_skill_series = personal_df["Musical skill"]
    athleticism_series   = personal_df["Athleticism"]
    dancing_habit_series = personal_df["Dancing habit"]
    fatigue_series       = personal_df["Total fatigue"]
    pleasure_series      = personal_df["Total pleasure"]

    

    
    liking_series      = music_df["Liking [0-7]"].dropna()
    familiarity_series = music_df["Familiarity [0-7]"].dropna()"""

    print("\n── Trial counts ────────────────────────────────────────────")
    print(f"  Total:   {len(stats_df)}")
    print(f"  Music:   {stats_df['Music Listening'].astype(bool).sum()}")
    print(f"  Silence: {(~stats_df['Music Listening'].astype(bool)).sum()}")

    """# ── Study results ────────────────────────────────────────────────────────
    cmc_flexor_beta_series    = stats_df["CMC_Flexor_mean_beta"].dropna()
    cmc_flexor_gamma_series   = stats_df["CMC_Flexor_mean_gamma"].dropna()
    cmc_extensor_beta_series  = stats_df["CMC_Extensor_mean_beta"].dropna()
    cmc_extensor_gamma_series = stats_df["CMC_Extensor_mean_gamma"].dropna()
    accuracy_series           = stats_df["RMS_Accuracy"].dropna()"""

    # ── Final numeric summary output (all boxplot metrics + requested extras) ──
    print("\n── Final metric summary (Range / Mean / Median / SD) ────────")
    boxplot_metrics: dict[str, pd.Series] = {
        "Musical Skill": musical_skill_series,
        "Listening habit [0-3]": listening_habit_series,
        "Athleticism": athleticism_series,
        "Dancing Habit": dancing_habit_series,
        "Total Fatigue": fatigue_series,
        "Total Pleasure": pleasure_series,
        "Liking [0-7]": liking_series,
        "Familiarity [0-7]": familiarity_series,
        "CMC Flexor beta": cmc_flexor_beta_series,
        "CMC Flexor gamma": cmc_flexor_gamma_series,
        "CMC Extensor beta": cmc_extensor_beta_series,
        "CMC Extensor gamma": cmc_extensor_gamma_series,
        "RMS Accuracy": accuracy_series,
    }
    for metric_name, metric_series in boxplot_metrics.items():
        _print_numeric_summary_line(metric_name, metric_series)

    print("\n── Participant age (years; based on Birthdate) ───────────────")
    _print_numeric_summary_line("Age", age_series, decimals=2)

    music_perceived_df = stats_df.loc[stats_df["Music Listening"].astype(bool)].copy()
    _print_grouped_metric_summary(
        music_perceived_df,
        group_col=perceived_category_col,
        metric_col=liking_col,
        title="Music liking by perceived category",
    )
    _print_grouped_metric_summary(
        music_perceived_df,
        group_col=perceived_category_col,
        metric_col=familiarity_col,
        title="Music familiarity by perceived category",
    )
    _print_pearson_correlation(
        stats_df,
        x_col=liking_col,
        y_col=familiarity_col,
        title="Liking - Familiarity Pearson correlation (all subjects/trials)",
    )

    # ── Render ───────────────────────────────────────────────────────────────
    plot_combined_descriptive_mosaic(
        personal_df,
        musical_skill_series, athleticism_series, dancing_habit_series,
        fatigue_series, pleasure_series,
        stats_df,
        liking_series, familiarity_series,
        cmc_flexor_beta_series,    cmc_flexor_gamma_series,
        cmc_extensor_beta_series,  cmc_extensor_gamma_series,
        accuracy_series,
        save_path=DESCRIPTIVE_OUTPUT / "descriptive_statistics_mosaic.png",
        subject_metadata=subject_metadata,
    )

    plt.show()
