"""Generate self-contained HTML reports for execution training artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import math
import os
from pathlib import Path

import polars as pl

from denoisr_crypto.training.baseline import (
    feature_columns,
    load_training_frame,
    split_indices,
)
from denoisr_crypto.types import StorageLayout

_PRIMARY = "#0f766e"
_SECONDARY = "#0f172a"
_ACCENT = "#b45309"
_NEGATIVE = "#b91c1c"
_GRID = "#d6d3d1"
_BG = "#f8fafc"
_PANEL = "#ffffff"
_TEXT = "#111827"
_MUTED = "#475569"


@dataclass(frozen=True)
class VisualizationArtifacts:
    combined_report_path: Path
    symbol_report_paths: dict[str, Path]


def build_visualization_reports(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    max_points: int = 720,
) -> VisualizationArtifacts | None:
    symbol_paths: dict[str, Path] = {}
    for symbol in symbols:
        symbol_path = build_symbol_report(
            layout=layout,
            symbol=symbol,
            max_points=max_points,
        )
        if symbol_path is None:
            return None
        symbol_paths[symbol] = symbol_path
    combined_report_path = build_combined_training_report(
        layout=layout,
        symbols=symbols,
        max_points=max_points,
        symbol_report_paths=symbol_paths,
    )
    if combined_report_path is None:
        return None
    return VisualizationArtifacts(
        combined_report_path=combined_report_path,
        symbol_report_paths=symbol_paths,
    )


def build_symbol_report(
    *,
    layout: StorageLayout,
    symbol: str,
    max_points: int = 720,
) -> Path | None:
    source_path = layout.feature_path(symbol, "features_multi_interval.parquet")
    if not source_path.exists():
        return None
    output_path = layout.report_symbol_dir(symbol) / "training_data_report.html"
    frame = pl.read_parquet(source_path).sort("open_time")
    if frame.is_empty():
        raise ValueError(f"No feature rows available for {symbol}")

    downsampled = _downsample(
        frame.select(
            "open_time",
            "close_1m",
            "dollar_volume_1m",
            "rolling_vol_15_1m",
            "target_return_5m",
            "target_vol_15m",
        ),
        max_points=max_points,
    )
    correlation_items = _top_correlations(
        frame=frame,
        target_column="target_return_5m",
        limit=10,
    )

    stats = {
        "Rows": _format_int(frame.height),
        "Start": _format_timestamp(frame["open_time"][0]),
        "End": _format_timestamp(frame["open_time"][-1]),
        "Positive 5m target ratio": _format_pct(
            frame.select((pl.col("target_return_5m") > 0).mean()).item()
        ),
        "Mean 15m target vol": _format_float(frame["target_vol_15m"].mean()),
        "Feature columns": _format_int(len(feature_columns(frame))),
    }
    body = "\n".join(
        [
            _summary_cards(stats),
            _section(
                "Market Path",
                _two_up(
                    _line_chart_card(
                        "Close Price 1m",
                        x_labels=_timestamp_labels(downsampled["open_time"]),
                        values=downsampled["close_1m"].to_list(),
                        color=_PRIMARY,
                    ),
                    _line_chart_card(
                        "Dollar Volume 1m",
                        x_labels=_timestamp_labels(downsampled["open_time"]),
                        values=downsampled["dollar_volume_1m"].to_list(),
                        color=_ACCENT,
                    ),
                ),
            ),
            _section(
                "Targets",
                _two_up(
                    _histogram_card(
                        "Target Return 5m",
                        values=frame["target_return_5m"].to_list(),
                        color=_PRIMARY,
                    ),
                    _histogram_card(
                        "Target Volatility 15m",
                        values=frame["target_vol_15m"].to_list(),
                        color=_ACCENT,
                    ),
                ),
            ),
            _section(
                "State Features",
                _two_up(
                    _line_chart_card(
                        "Rolling Volatility 15",
                        x_labels=_timestamp_labels(downsampled["open_time"]),
                        values=downsampled["rolling_vol_15_1m"].to_list(),
                        color=_SECONDARY,
                    ),
                    _bar_chart_card(
                        "Top Feature Correlations vs Target Return 5m",
                        labels=[label for label, _ in correlation_items],
                        values=[value for _, value in correlation_items],
                    ),
                ),
            ),
            _section(
                "Sample Rows",
                _data_table(
                    frame.select(
                        "open_time",
                        "close_1m",
                        "dollar_volume_1m",
                        "rolling_vol_15_1m",
                        "target_return_5m",
                        "target_vol_15m",
                    ).head(8)
                ),
            ),
        ]
    )
    html = _render_page(
        title=f"{symbol} Training Data Report",
        subtitle="Execution POC feature artifact visualization",
        body=body,
        links=[
            (
                "Combined training report",
                _relative_href(output_path, layout.reports_dir() / "training_data_report.html"),
            ),
            ("Feature parquet", _relative_href(output_path, source_path)),
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def build_combined_training_report(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    max_points: int = 720,
    symbol_report_paths: dict[str, Path] | None = None,
) -> Path | None:
    frame = load_training_frame(layout, symbols)
    if frame is None:
        return None
    frame = frame.drop_nulls(subset=["target_return_5m", "target_vol_15m"])
    if frame.is_empty():
        raise ValueError("No training rows available")
    feature_count = len(feature_columns(frame))
    train_end, val_end = split_indices(frame.height)
    train_last = frame["open_time"][train_end - 1]
    val_last = frame["open_time"][val_end - 1]
    split_items = [
        ("Train", train_end, _PRIMARY),
        ("Validation", val_end - train_end, _ACCENT),
        ("Test", frame.height - val_end, _SECONDARY),
    ]

    counts_by_symbol = (
        frame.group_by("symbol")
        .len()
        .sort("symbol")
        .rename({"len": "row_count"})
    )
    downsampled = _downsample(
        frame.select("open_time", "target_return_5m", "target_vol_15m"),
        max_points=max_points,
    )
    correlation_items = _top_correlations(
        frame=frame,
        target_column="target_return_5m",
        limit=12,
    )
    stats = {
        "Symbols": ", ".join(symbols),
        "Rows": _format_int(frame.height),
        "Feature columns": _format_int(feature_count),
        "Start": _format_timestamp(frame["open_time"][0]),
        "End": _format_timestamp(frame["open_time"][-1]),
        "Train / Val / Test": f"{train_end:,} / {val_end - train_end:,} / {frame.height - val_end:,}",
    }
    output_path = layout.reports_dir() / "training_data_report.html"
    extra_links: list[tuple[str, str]] = []
    if symbol_report_paths:
        extra_links.extend(
            (
                f"{symbol} symbol report",
                _relative_href(output_path, path),
            )
            for symbol, path in sorted(symbol_report_paths.items())
        )

    body_parts = [
        _summary_cards(stats),
        _section(
            "Temporal Split",
            _stacked_split_card(
                items=split_items,
                start=_format_timestamp(frame["open_time"][0]),
                train_end=_format_timestamp(train_last),
                val_end=_format_timestamp(val_last),
                end=_format_timestamp(frame["open_time"][-1]),
            ),
        ),
        _section(
            "Target Behavior",
            _two_up(
                _line_chart_card(
                    "Target Return 5m Over Time",
                    x_labels=_timestamp_labels(downsampled["open_time"]),
                    values=downsampled["target_return_5m"].to_list(),
                    color=_PRIMARY,
                ),
                _line_chart_card(
                    "Target Volatility 15m Over Time",
                    x_labels=_timestamp_labels(downsampled["open_time"]),
                    values=downsampled["target_vol_15m"].to_list(),
                    color=_ACCENT,
                ),
            ),
        ),
        _section(
            "Dataset Composition",
            _two_up(
                _bar_chart_card(
                    "Rows per Symbol",
                    labels=counts_by_symbol["symbol"].to_list(),
                    values=counts_by_symbol["row_count"].cast(pl.Float64).to_list(),
                ),
                _bar_chart_card(
                    "Top Absolute Correlations vs Target Return 5m",
                    labels=[label for label, _ in correlation_items],
                    values=[abs(value) for _, value in correlation_items],
                ),
            ),
        ),
    ]
    html = _render_page(
        title="Execution Training Data Report",
        subtitle="Combined multi-symbol feature dataset for the Binance spot POC",
        body="\n".join(body_parts),
        links=extra_links,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def _top_correlations(
    *,
    frame: pl.DataFrame,
    target_column: str,
    limit: int,
) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for column in feature_columns(frame):
        series = frame.select(column, target_column).drop_nulls()
        if series.height < 3:
            continue
        corr = series.select(pl.corr(column, target_column)).item()
        if corr is None or math.isnan(float(corr)):
            continue
        items.append((column, float(corr)))
    items.sort(key=lambda item: abs(item[1]), reverse=True)
    return items[:limit]


def _downsample(frame: pl.DataFrame, *, max_points: int) -> pl.DataFrame:
    if frame.height <= max_points:
        return frame
    step = max(1, math.ceil(frame.height / max_points))
    return frame.gather_every(step)


def _timestamp_labels(values: pl.Series) -> list[str]:
    return [_format_timestamp(value) for value in values]


def _relative_href(report_path: Path, target_path: Path) -> str:
    return os.path.relpath(target_path, start=report_path.parent).replace("\\", "/")


def _render_page(
    *,
    title: str,
    subtitle: str,
    body: str,
    links: list[tuple[str, str]],
) -> str:
    link_html = "".join(
        f'<a class="link-pill" href="{escape(href)}">{escape(label)}</a>'
        for label, href in links
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: {_BG};
      --panel: {_PANEL};
      --text: {_TEXT};
      --muted: {_MUTED};
      --border: {_GRID};
      --primary: {_PRIMARY};
      --secondary: {_SECONDARY};
      --accent: {_ACCENT};
      --negative: {_NEGATIVE};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 24rem),
        linear-gradient(180deg, #fefce8 0%, var(--bg) 22rem);
      color: var(--text);
    }}
    main {{ max-width: 1360px; margin: 0 auto; padding: 32px 24px 56px; }}
    header {{
      background: rgba(255,255,255,0.82);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(214,211,209,0.9);
      border-radius: 24px;
      padding: 24px 28px;
      box-shadow: 0 18px 50px rgba(15,23,42,0.08);
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 600; line-height: 1.1; }}
    h1 {{ font-size: 2.35rem; letter-spacing: -0.04em; }}
    h2 {{ font-size: 1.45rem; letter-spacing: -0.02em; margin-bottom: 14px; }}
    h3 {{ font-size: 1rem; margin-bottom: 10px; }}
    p {{ margin: 0; color: var(--muted); }}
    .meta {{ margin-top: 10px; font-size: 0.95rem; }}
    .link-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
    .link-pill {{
      color: var(--secondary);
      text-decoration: none;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.95rem;
    }}
    .summary-grid, .two-up, .metric-grid {{
      display: grid;
      gap: 16px;
    }}
    .summary-grid {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-top: 26px;
    }}
    .two-up {{
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    }}
    .metric-grid {{
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    }}
    .card, section {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(214,211,209,0.9);
      border-radius: 22px;
      box-shadow: 0 16px 40px rgba(15,23,42,0.06);
    }}
    section {{ margin-top: 22px; padding: 22px; }}
    .card {{ padding: 18px; }}
    .card .label {{
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .card .value {{
      margin-top: 8px;
      font-size: 1.55rem;
      letter-spacing: -0.03em;
      color: var(--secondary);
    }}
    .chart-card {{ padding: 18px; }}
    .chart-note {{ margin-top: 8px; font-size: 0.92rem; }}
    svg {{ width: 100%; height: auto; display: block; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
      overflow: hidden;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(214,211,209,0.8);
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .split-bar {{
      display: flex;
      height: 24px;
      border-radius: 999px;
      overflow: hidden;
      margin: 14px 0 16px;
      border: 1px solid rgba(214,211,209,0.9);
    }}
    .split-segment {{
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-size: 0.84rem;
      white-space: nowrap;
    }}
    .split-meta {{
      display: grid;
      gap: 6px;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    @media (max-width: 720px) {{
      main {{ padding: 20px 14px 40px; }}
      header, section {{ padding: 18px; }}
      h1 {{ font-size: 1.9rem; }}
      .two-up {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{escape(title)}</h1>
      <p class="meta">{escape(subtitle)}</p>
      <p class="meta">Generated at {escape(generated_at)}</p>
      <div class="link-row">{link_html}</div>
    </header>
    {body}
  </main>
</body>
</html>
"""


def _section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _summary_cards(items: dict[str, str]) -> str:
    cards = "".join(
        f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'
        for label, value in items.items()
    )
    return f'<div class="summary-grid">{cards}</div>'


def _two_up(left: str, right: str) -> str:
    return f'<div class="two-up">{left}{right}</div>'


def _line_chart_card(title: str, *, x_labels: list[str], values: list[float | None], color: str) -> str:
    svg = _svg_line_chart(
        x_labels=x_labels,
        values=[_float_or_none(value) for value in values],
        color=color,
    )
    return f'<div class="card chart-card"><h3>{escape(title)}</h3>{svg}</div>'


def _histogram_card(title: str, *, values: list[float | None], color: str) -> str:
    svg = _svg_histogram(values=[_float_or_none(value) for value in values], color=color)
    return f'<div class="card chart-card"><h3>{escape(title)}</h3>{svg}</div>'


def _bar_chart_card(title: str, *, labels: list[str], values: list[float]) -> str:
    svg = _svg_bar_chart(labels=labels, values=values)
    return f'<div class="card chart-card"><h3>{escape(title)}</h3>{svg}</div>'


def _stacked_split_card(
    *,
    items: list[tuple[str, int, str]],
    start: str,
    train_end: str,
    val_end: str,
    end: str,
) -> str:
    total = sum(value for _, value, _ in items) or 1
    segments = "".join(
        (
            f'<div class="split-segment" style="width:{(value / total) * 100:.2f}%;background:{color};">'
            f"{escape(label)} {value:,}"
            "</div>"
        )
        for label, value, color in items
    )
    meta = (
        f'<div class="split-meta">'
        f"<div>Start: {escape(start)}</div>"
        f"<div>Train end: {escape(train_end)}</div>"
        f"<div>Validation end: {escape(val_end)}</div>"
        f"<div>End: {escape(end)}</div>"
        f"</div>"
    )
    return f'<div class="card"><div class="split-bar">{segments}</div>{meta}</div>'


def _data_table(frame: pl.DataFrame) -> str:
    columns = frame.columns
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    rows = []
    for row in frame.iter_rows(named=True):
        cells = "".join(
            f"<td>{escape(_stringify_metric(row[column]))}</td>"
            for column in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    return f'<div class="card"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def _svg_line_chart(*, x_labels: list[str], values: list[float | None], color: str) -> str:
    filtered = [(x, y) for x, y in zip(x_labels, values, strict=False) if y is not None]
    if len(filtered) < 2:
        return '<p class="chart-note">Not enough data to render chart.</p>'
    width = 760
    height = 280
    pad_left = 54
    pad_right = 18
    pad_top = 16
    pad_bottom = 30
    xs = [item[0] for item in filtered]
    ys = [item[1] for item in filtered]
    y_min = min(ys)
    y_max = max(ys)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    x_step = (width - pad_left - pad_right) / max(len(filtered) - 1, 1)
    points = []
    for idx, (_, value) in enumerate(filtered):
        x = pad_left + idx * x_step
        y = pad_top + (height - pad_top - pad_bottom) * (1 - ((value - y_min) / (y_max - y_min)))
        points.append(f"{x:.2f},{y:.2f}")
    grid = "".join(
        f'<line x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" stroke="{_GRID}" stroke-width="1" />'
        for y in (
            pad_top,
            pad_top + (height - pad_top - pad_bottom) / 2,
            height - pad_bottom,
        )
    )
    labels = (
        f'<text x="{pad_left}" y="{height - 8}" fill="{_MUTED}" font-size="12">{escape(xs[0])}</text>'
        f'<text x="{width - pad_right}" y="{height - 8}" text-anchor="end" fill="{_MUTED}" font-size="12">{escape(xs[-1])}</text>'
        f'<text x="8" y="{pad_top + 8}" fill="{_MUTED}" font-size="12">{escape(_format_float(y_max))}</text>'
        f'<text x="8" y="{height - pad_bottom + 4}" fill="{_MUTED}" font-size="12">{escape(_format_float(y_min))}</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="line chart">'
        f"{grid}"
        f'<polyline fill="none" stroke="{escape(color)}" stroke-width="2.5" points="{" ".join(points)}" />'
        f"{labels}</svg>"
    )


def _svg_histogram(*, values: list[float | None], color: str) -> str:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return '<p class="chart-note">Not enough data to render histogram.</p>'
    width = 760
    height = 280
    pad_left = 48
    pad_right = 18
    pad_top = 16
    pad_bottom = 30
    bins = min(30, max(10, int(math.sqrt(len(clean)))))
    value_min = min(clean)
    value_max = max(clean)
    if math.isclose(value_min, value_max):
        value_min -= 1.0
        value_max += 1.0
    bin_width = (value_max - value_min) / bins
    counts = [0 for _ in range(bins)]
    for value in clean:
        index = min(int((value - value_min) / bin_width), bins - 1)
        counts[index] += 1
    max_count = max(counts) or 1
    chart_width = width - pad_left - pad_right
    bar_width = chart_width / bins
    rects = []
    for index, count in enumerate(counts):
        bar_height = (height - pad_top - pad_bottom) * (count / max_count)
        x = pad_left + index * bar_width
        y = height - pad_bottom - bar_height
        rects.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 1, 1):.2f}" height="{bar_height:.2f}" fill="{escape(color)}" opacity="0.82" />'
        )
    labels = (
        f'<text x="{pad_left}" y="{height - 8}" fill="{_MUTED}" font-size="12">{escape(_format_float(value_min))}</text>'
        f'<text x="{width - pad_right}" y="{height - 8}" text-anchor="end" fill="{_MUTED}" font-size="12">{escape(_format_float(value_max))}</text>'
        f'<text x="8" y="{pad_top + 8}" fill="{_MUTED}" font-size="12">{max_count}</text>'
    )
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="histogram">{"".join(rects)}{labels}</svg>'


def _svg_bar_chart(*, labels: list[str], values: list[float]) -> str:
    if not labels or not values:
        return '<p class="chart-note">Not enough data to render bars.</p>'
    width = 760
    height = max(220, 34 * len(labels) + 24)
    pad_left = 180
    pad_right = 30
    pad_top = 12
    pad_bottom = 18
    max_abs_value = max(abs(value) for value in values) or 1.0
    bar_area = width - pad_left - pad_right
    has_negative = min(values) < 0
    zero_x = pad_left if not has_negative else pad_left + bar_area / 2
    usable_width = bar_area * 0.92 if not has_negative else (bar_area / 2) * 0.92
    row_height = (height - pad_top - pad_bottom) / max(len(labels), 1)
    rows = []
    for index, (label, value) in enumerate(zip(labels, values, strict=False)):
        y = pad_top + index * row_height
        scaled = (abs(value) / max_abs_value) * usable_width
        if value >= 0:
            x = zero_x
            color = _PRIMARY
        else:
            x = zero_x - scaled
            color = _NEGATIVE
        rows.append(
            f'<text x="{pad_left - 10}" y="{y + row_height * 0.6:.2f}" text-anchor="end" fill="{_MUTED}" font-size="12">{escape(label)}</text>'
            f'<rect x="{x:.2f}" y="{y + row_height * 0.18:.2f}" width="{scaled:.2f}" height="{row_height * 0.56:.2f}" rx="6" fill="{color}" opacity="0.88" />'
            f'<text x="{(x + scaled + 6) if value >= 0 else (x - 6):.2f}" y="{y + row_height * 0.6:.2f}" fill="{_TEXT}" font-size="12" text-anchor="{"start" if value >= 0 else "end"}">{escape(_format_float(value))}</text>'
        )
    axis = f'<line x1="{zero_x:.2f}" y1="{pad_top}" x2="{zero_x:.2f}" y2="{height - pad_bottom}" stroke="{_GRID}" stroke-width="1" />'
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="bar chart">{axis}{"".join(rows)}</svg>'


def _format_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _format_float(value: object) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 1:
        return f"{number:,.4f}"
    return f"{number:.6f}"


def _format_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_int(value: int) -> str:
    return f"{value:,}"


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def _stringify_metric(value: object) -> str:
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, datetime):
        return _format_timestamp(value)
    return str(value)
