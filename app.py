import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
import re
import io
import base64
import time
import datetime
from io import StringIO

st.set_page_config(page_title="ETF Holdings Analyzer", page_icon="📊", layout="wide")
st.markdown('<style>[data-testid="stMetricValue"]{font-size:1.4rem}</style>', unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SECTOR_COLORS = {
    "Technology": "#4C72B0",
    "Financial Services": "#55A868",
    "Financials": "#55A868",
    "Healthcare": "#C44E52",
    "Consumer Cyclical": "#8172B2",
    "Consumer Defensive": "#937860",
    "Industrials": "#DA8BC3",
    "Communication Services": "#8C8C8C",
    "Energy": "#CCB974",
    "Basic Materials": "#64B5CD",
    "Real Estate": "#FF8C42",
    "Utilities": "#6BCDC9",
    "Unknown": "#CCCCCC",
}

PERIOD_MAP = {
    "1D":  ("1d",  "5m"),
    "1W":  ("5d",  "1h"),
    "1M":  ("1mo", "1d"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
    "5Y":  ("5y",  "1wk"),
    "YTD": ("ytd", "1d"),
}

# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(url: str) -> requests.Response | None:
    try:
        r = requests.get(url, headers=SCRAPE_HEADERS, timeout=15)
        return r if r.status_code == 200 else None
    except Exception:
        return None


# ── Yahoo Finance v8 range API (no crumb / no auth needed) ────────────────────
#
# Using ?range=1mo&interval=1d is a different code path from the old
# ?period1=...&period2=...&crumb=... approach.  No session management,
# no cookies, no API key — just a plain GET.

def _yf_range_fetch(ticker: str, range_: str, interval: str) -> pd.DataFrame | None:
    """Fetch price history via Yahoo Finance v8 range endpoint (free, unauthenticated)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": range_, "interval": interval, "includePrePost": "false"}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=SCRAPE_HEADERS, timeout=15)
            if r.status_code == 429:
                if attempt < 2:
                    time.sleep(2 ** attempt + 1)
                continue
            if r.status_code != 200:
                return None
            result = r.json().get("chart", {}).get("result", [])
            if not result:
                return None
            rc = result[0]
            timestamps = rc.get("timestamp", [])
            adjclose = rc.get("indicators", {}).get("adjclose", [])
            if adjclose and "adjclose" in adjclose[0]:
                closes = adjclose[0]["adjclose"]
            else:
                closes = rc.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            if not timestamps or not closes:
                return None
            idx = pd.to_datetime(timestamps, unit="s", utc=True)
            df = pd.DataFrame({"Close": closes}, index=idx).dropna()
            return df if not df.empty else None
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return None


# ── Stockanalysis.com scrapers ─────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_info(ticker: str) -> dict:
    result = {"name": ticker, "aum": None, "expense_ratio": None, "num_holdings": None}
    resp = _get(f"https://stockanalysis.com/etf/{ticker.lower()}/")
    if resp is None:
        return result
    soup = BeautifulSoup(resp.text, "html.parser")
    h1 = soup.find("h1")
    if h1:
        result["name"] = re.sub(r"\s*\([^)]+\)\s*$", "", h1.get_text(strip=True))
    try:
        for t in pd.read_html(StringIO(resp.text)):
            if t.shape[1] == 2:
                t.columns = ["key", "val"]
                lk = dict(zip(t["key"].astype(str), t["val"].astype(str)))
                if "Assets" in lk:
                    result["aum"] = lk["Assets"]
                if "Expense Ratio" in lk:
                    result["expense_ratio"] = lk["Expense Ratio"]
                if "Holdings" in lk:
                    result["num_holdings"] = lk["Holdings"]
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_holdings(ticker: str) -> pd.DataFrame | None:
    resp = _get(f"https://stockanalysis.com/etf/{ticker.lower()}/holdings/")
    if resp is None:
        return None
    try:
        tables = pd.read_html(StringIO(resp.text))
        return max(tables, key=len) if tables else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_meta(ticker: str) -> dict:
    """Return sector and industry scraped from stockanalysis.com stock page."""
    sym = str(ticker).strip()
    if not sym or sym in ("", "—", "N/A", "nan"):
        return {"sector": "Unknown", "industry": "Unknown"}
    resp = _get(f"https://stockanalysis.com/stocks/{sym.lower()}/")
    if resp is None:
        return {"sector": "Unknown", "industry": "Unknown"}
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        result = {"sector": "Unknown", "industry": "Unknown"}
        for el in soup.find_all("span", class_=lambda c: c and "font-semibold" in c):
            label = el.get_text(strip=True)
            if label in ("Sector", "Industry"):
                sibling = el.find_next_sibling()
                val = sibling.get_text(strip=True) if sibling else None
                if not val:
                    link = el.parent.find("a")
                    val = link.get_text(strip=True) if link else None
                if val:
                    if label == "Sector":
                        result["sector"] = val
                    else:
                        result["industry"] = val
        return result
    except Exception:
        return {"sector": "Unknown", "industry": "Unknown"}


# ── Ticker classification + name lookup ───────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def classify_ticker(ticker: str) -> str:
    resp = _get(f"https://stockanalysis.com/etf/{ticker.lower()}/holdings/")
    if resp is None:
        return "stock"
    try:
        tables = pd.read_html(StringIO(resp.text))
        if tables and len(max(tables, key=len)) > 2:
            return "etf"
    except Exception:
        pass
    return "stock"


@st.cache_data(ttl=300, show_spinner=False)
def get_current_price(ticker: str) -> float | None:
    df = _yf_range_fetch(ticker, "5d", "1d")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_name(ticker: str) -> str:
    resp = _get(f"https://stockanalysis.com/stocks/{ticker.lower()}/")
    if resp is None:
        return ticker
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        h1 = soup.find("h1")
        if h1:
            return re.sub(r"\s*\([^)]+\)\s*$", "", h1.get_text(strip=True))
    except Exception:
        pass
    return ticker


# ── Price fetchers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_history(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    return _yf_range_fetch(ticker, period, interval)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_prices_batch(tickers: tuple) -> pd.DataFrame | None:
    if not tickers:
        return None
    results: dict = {}
    for ticker in tickers:
        df = _yf_range_fetch(ticker, "1y", "1d")
        if df is not None:
            results[ticker] = df["Close"]
    return pd.DataFrame(results) if results else None


# ── Normalisation ──────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_holdings(raw: pd.DataFrame) -> pd.DataFrame | None:
    df = raw.copy()
    sym_col  = _col(df, "symbol", "ticker", "sym", "no.")
    name_col = _col(df, "name", "company", "holding", "security", "description", "holdingName")
    wt_col   = _col(df, "weight", "% weight", "% of assets", "allocation", "port. weight",
                    "holdingPercent", "% net assets", "weightings", "% port.")
    if wt_col is None:
        return None
    rename = {}
    if sym_col:
        rename[sym_col] = "Symbol"
    if name_col:
        rename[name_col] = "Name"
    rename[wt_col] = "Weight"
    df = df.rename(columns=rename)
    if "Symbol" not in df.columns:
        df["Symbol"] = ""
    if "Name" not in df.columns:
        df["Name"] = df["Symbol"]
    df["Weight"] = (
        df["Weight"].astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    valid = df["Weight"].dropna()
    if len(valid) > 0 and valid.max() <= 2.0:
        df["Weight"] = df["Weight"] * 100
    df = (
        df[["Symbol", "Name", "Weight"]]
        .dropna(subset=["Weight"])
        .query("Weight > 0")
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1
    return df


# ── Sparkline generator ────────────────────────────────────────────────────────

def make_sparkline(prices: pd.Series) -> str | None:
    if prices is None or len(prices) < 5:
        return None
    try:
        start, end = prices.iloc[0], prices.iloc[-1]
        color = "#2ca02c" if end >= start else "#d62728"
        n = len(prices)
        xs = range(n)
        arr = prices.values

        fig, ax = plt.subplots(figsize=(2.2, 0.5), dpi=100)
        ax.plot(xs, arr, color=color, linewidth=1.4, solid_capstyle="round")
        ax.fill_between(xs, arr, start, where=(arr >= start), color="#2ca02c", alpha=0.12)
        ax.fill_between(xs, arr, start, where=(arr < start),  color="#d62728", alpha=0.12)
        ax.axhline(start, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
        # Tick marks: 6M ≈ 50%, 3M ≈ 75%, 1M ≈ 92%
        for frac in (0.5, 0.75, 0.917):
            ax.axvline(int(n * frac), color="gray", linewidth=0.5, alpha=0.35, linestyle=":")
        ax.set_xlim(0, n - 1)
        ax.axis("off")
        ax.margins(y=0.18)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    transparent=True, pad_inches=0.01)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    except Exception:
        return None


# ── ETF price chart ───────────────────────────────────────────────────────────

def make_etf_chart(hist: pd.DataFrame, ticker: str, period: str) -> go.Figure:
    prices = hist["Close"]
    start, current = prices.iloc[0], prices.iloc[-1]
    color = "#2ca02c" if current >= start else "#d62728"

    fig = go.Figure(go.Scatter(
        x=hist.index,
        y=prices,
        mode="lines",
        line=dict(color=color, width=2),
        hovertemplate="<b>%{x}</b>: $%{y:.2f}<extra></extra>",
        name=ticker,
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        xaxis=dict(showgrid=False, zeroline=False, rangeslider=dict(visible=False)),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(200,200,200,0.25)",
            tickprefix="$", zeroline=False,
            range=[prices.min() * 0.997, prices.max() * 1.003],
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def make_compare_chart(
    hist_a: pd.DataFrame, hist_b: pd.DataFrame, ta: str, tb: str
) -> go.Figure:
    pa = hist_a["Close"]
    pb = hist_b["Close"]
    na = (pa / pa.iloc[0]) * 100
    nb = (pb / pb.iloc[0]) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_a.index, y=na.round(2),
        mode="lines", name=ta,
        line=dict(color="#4C72B0", width=2),
        hovertemplate="<b>" + ta + "</b>: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hist_b.index, y=nb.round(2),
        mode="lines", name=tb,
        line=dict(color="#C44E52", width=2),
        hovertemplate="<b>" + tb + "</b>: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        xaxis=dict(showgrid=False, zeroline=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.25)",
                   zeroline=False, title="Indexed to 100"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0),
    )
    return fig


# ── Holdings bar chart ────────────────────────────────────────────────────────

def make_bar_chart(df: pd.DataFrame) -> go.Figure:
    bar = df.head(25).copy()
    bar["Label"] = bar.apply(lambda r: r["Symbol"] if r["Symbol"] else r["Name"][:20], axis=1)

    has_industry = "Industry" in bar.columns
    hover = (
        "<b>%{y}</b><br>Weight: %{x:.2f}%<br>Sector: %{customdata[0]}"
        + ("<br>Industry: %{customdata[1]}" if has_industry else "")
        + "<extra></extra>"
    )
    custom = (
        list(zip(bar["Sector"], bar["Industry"])) if has_industry
        else list(zip(bar["Sector"],))
    )

    fig = go.Figure(go.Bar(
        x=bar["Weight"], y=bar["Label"],
        orientation="h",
        marker_color=[SECTOR_COLORS.get(s, "#CCCCCC") for s in bar["Sector"]],
        customdata=custom,
        hovertemplate=hover,
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis=dict(title="Weight (%)"),
        margin=dict(l=0, r=20, t=10, b=40),
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Sector donut + industry drill-down ────────────────────────────────────────

def make_sector_donut(df: pd.DataFrame) -> go.Figure:
    """Clean donut showing sector weights only."""
    agg = df.groupby("Sector")["Weight"].sum().reset_index().sort_values("Weight", ascending=False)
    fig = go.Figure(go.Pie(
        labels=agg["Sector"],
        values=agg["Weight"].round(2),
        marker_colors=[SECTOR_COLORS.get(s, "#CCCCCC") for s in agg["Sector"]],
        hole=0.42,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
        sort=False,
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_industry_bar(df: pd.DataFrame, sector: str) -> go.Figure:
    """Horizontal bar chart of industry breakdown within a sector."""
    sub = df[df["Sector"] == sector]
    agg = (
        sub.groupby("Industry")["Weight"]
        .sum()
        .reset_index()
        .sort_values("Weight", ascending=False)
    )
    color = SECTOR_COLORS.get(sector, "#CCCCCC")
    fig = go.Figure(go.Bar(
        x=agg["Weight"].round(2),
        y=agg["Industry"],
        orientation="h",
        marker_color=color,
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Weight (%)"),
        title=dict(text=f"{sector} — Industry Mix", font=dict(size=13)),
        margin=dict(l=0, r=20, t=40, b=30),
        height=max(180, len(agg) * 38 + 70),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_sector_breakdown(df: pd.DataFrame, selectbox_key: str) -> None:
    """Donut for sectors + selectbox to drill into industry mix."""
    st.plotly_chart(make_sector_donut(df), use_container_width=True)
    sectors = (
        df.groupby("Sector")["Weight"].sum()
        .reset_index()
        .sort_values("Weight", ascending=False)["Sector"]
        .tolist()
    )
    chosen = st.selectbox(
        "Select a sector to see its industry breakdown:",
        options=["—"] + sectors,
        key=selectbox_key,
    )
    if chosen != "—":
        st.plotly_chart(make_industry_bar(df, chosen), use_container_width=True)


def sector_agg(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Sector")["Weight"].sum().reset_index()


def _industry_hover_text(df: pd.DataFrame, sectors: list[str]) -> list[str]:
    """Build per-sector industry breakdown strings for grouped bar chart hover."""
    if "Industry" not in df.columns:
        return ["" for _ in sectors]
    ind_agg = df.groupby(["Sector", "Industry"])["Weight"].sum().reset_index()
    result = []
    for sec in sectors:
        rows = ind_agg[ind_agg["Sector"] == sec].sort_values("Weight", ascending=False)
        if rows.empty:
            result.append("")
        else:
            lines = "<br>".join(f"  {r['Industry']}: {r['Weight']:.1f}%" for _, r in rows.iterrows())
            result.append(lines)
    return result


# ── Shared data pipeline ───────────────────────────────────────────────────────

def load_etf(ticker: str, top_n: int, progress_label: str = "") -> tuple[dict, pd.DataFrame | None]:
    info = fetch_etf_info(ticker)
    raw = fetch_holdings(ticker)
    if raw is None:
        return info, None
    holdings = normalize_holdings(raw)
    if holdings is None or holdings.empty:
        return info, None

    display = holdings.head(top_n).copy()
    symbols = display["Symbol"].tolist()
    prefix = f"{progress_label}: " if progress_label else ""

    bar = st.progress(0, text=f"{prefix}Looking up sector & industry…")
    sectors, industries = [], []
    for i, sym in enumerate(symbols):
        meta = get_stock_meta(sym)
        sectors.append(meta["sector"])
        industries.append(meta["industry"])
        bar.progress((i + 1) / len(symbols), text=f"{prefix}Metadata {i+1}/{len(symbols)}")
    bar.empty()

    display["Sector"]   = sectors
    display["Industry"] = industries

    valid_syms = tuple(s for s in symbols if s and str(s) not in ("", "nan", "N/A"))
    price_df = fetch_stock_prices_batch(valid_syms) if valid_syms else None

    bar2 = st.progress(0, text=f"{prefix}Generating price charts…")
    charts = []
    for i, sym in enumerate(symbols):
        if price_df is not None and sym in price_df.columns:
            charts.append(make_sparkline(price_df[sym].dropna()))
        else:
            charts.append(None)
        bar2.progress((i + 1) / len(symbols))
    bar2.empty()

    display["1Y Chart"] = charts
    display["Weight"]   = display["Weight"].round(4)
    return info, display


# ── Portfolio analysis pipeline ───────────────────────────────────────────────

def run_portfolio_analysis(
    tickers: list[str],
    amounts: list[float],
    input_type: str,
) -> dict | None:
    # ── Step 1: portfolio weights ──────────────────────────────────────────────
    if input_type == "shares":
        bar = st.progress(0, text="Fetching current prices…")
        market_values = []
        for i, (t, sh) in enumerate(zip(tickers, amounts)):
            price = get_current_price(t) or 0.0
            market_values.append(price * sh)
            bar.progress((i + 1) / len(tickers))
        bar.empty()
        amounts_w = market_values
    else:
        amounts_w = amounts

    total = sum(amounts_w)
    if total <= 0:
        return None
    weights = [a / total * 100 for a in amounts_w]

    # ── Step 2: classify + aggregate ──────────────────────────────────────────
    exposure: dict[str, dict] = {}
    summary_rows: list[dict]  = []
    coverage_notes: list[str] = []

    bar2 = st.progress(0, text="Analyzing holdings…")
    for i, (ticker, amount, w) in enumerate(zip(tickers, amounts, weights)):
        bar2.progress((i + 0.5) / len(tickers), text=f"Analyzing {ticker}…")
        t_type = classify_ticker(ticker)

        if t_type == "etf":
            etf_info     = fetch_etf_info(ticker)
            display_name = etf_info["name"]
            raw          = fetch_holdings(ticker)
            if raw is not None:
                h = normalize_holdings(raw)
                if h is not None and not h.empty:
                    known_pct = h["Weight"].sum()
                    coverage_notes.append(
                        f"**{ticker}**: top {len(h)} holdings cover {known_pct:.1f}% of ETF weight"
                    )
                    for _, hr in h.iterrows():
                        sym = str(hr["Symbol"]).strip()
                        if not sym:
                            continue
                        exp = w * hr["Weight"] / 100
                        if sym not in exposure:
                            exposure[sym] = {"name": hr["Name"], "total": 0.0, "sources": {}}
                        exposure[sym]["total"] += exp
                        exposure[sym]["sources"][ticker] = (
                            exposure[sym]["sources"].get(ticker, 0.0) + exp
                        )
        else:
            display_name = get_stock_name(ticker)
            if ticker not in exposure:
                exposure[ticker] = {"name": display_name, "total": 0.0, "sources": {}}
            exposure[ticker]["total"] += w
            exposure[ticker]["sources"]["Direct"] = (
                exposure[ticker]["sources"].get("Direct", 0.0) + w
            )

        summary_rows.append({
            "Ticker":           ticker,
            "Name":             display_name,
            "Portfolio Weight": w,
            "Amount":           amount,
        })
        bar2.progress((i + 1) / len(tickers))
    bar2.empty()

    # ── Step 3: build exposure DataFrame ──────────────────────────────────────
    rows = []
    for sym, data in exposure.items():
        bd = " | ".join(
            f"{src}: {v:.2f}%"
            for src, v in sorted(data["sources"].items(), key=lambda x: -x[1])
        )
        rows.append({
            "Symbol":         sym,
            "Name":           data["name"],
            "Total Exposure": round(data["total"], 4),
            "Breakdown":      bd,
        })
    exp_df = (
        pd.DataFrame(rows)
        .sort_values("Total Exposure", ascending=False)
        .reset_index(drop=True)
    )
    exp_df.index += 1

    # ── Step 4: sector + industry for top 50 ──────────────────────────────────
    top_n_meta = min(50, len(exp_df))
    top_syms   = exp_df["Symbol"].head(top_n_meta).tolist()

    bar3 = st.progress(0, text="Fetching sector & industry…")
    sec_map, ind_map = {}, {}
    for i, sym in enumerate(top_syms):
        meta = get_stock_meta(sym)
        sec_map[sym] = meta["sector"]
        ind_map[sym] = meta["industry"]
        bar3.progress((i + 1) / top_n_meta, text=f"Metadata {i+1}/{top_n_meta}")
    bar3.empty()

    exp_df["Sector"]   = exp_df["Symbol"].map(lambda s: sec_map.get(s, "Unknown"))
    exp_df["Industry"] = exp_df["Symbol"].map(lambda s: ind_map.get(s, "Unknown"))

    # ── Step 5: sparklines ────────────────────────────────────────────────────
    valid_syms = tuple(s for s in top_syms if s and str(s) not in ("", "nan"))
    price_df   = fetch_stock_prices_batch(valid_syms) if valid_syms else None

    bar4 = st.progress(0, text="Generating price charts…")
    charts = []
    for i, sym in enumerate(exp_df["Symbol"]):
        if price_df is not None and sym in price_df.columns:
            charts.append(make_sparkline(price_df[sym].dropna()))
        else:
            charts.append(None)
        if i < top_n_meta:
            bar4.progress((i + 1) / top_n_meta)
    bar4.empty()
    exp_df["1Y Chart"] = charts

    return {
        "portfolio_summary": pd.DataFrame(summary_rows),
        "exposure_df":       exp_df,
        "coverage_notes":    coverage_notes,
    }


# ── Rendering helpers ──────────────────────────────────────────────────────────

def render_info_metrics(info: dict) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric("AUM", info["aum"] or "N/A")
    m2.metric("Expense Ratio", info["expense_ratio"] or "N/A")
    m3.metric("# Holdings", info["num_holdings"] or "N/A")


def render_price_section(ticker: str, period_key: str) -> None:
    period = st.radio(
        "Period", list(PERIOD_MAP.keys()),
        horizontal=True, key=period_key, index=0,
    )
    p, iv = PERIOD_MAP[period]
    hist = fetch_price_history(ticker, p, iv)

    if hist is None or hist.empty:
        st.info("Price chart unavailable — data is temporarily unavailable. "
                "Try again in a few minutes.")
        return

    prices = hist["Close"]
    current   = prices.iloc[-1]
    start     = prices.iloc[0]
    pct_chg   = (current / start - 1) * 100
    chg_color = "normal" if pct_chg >= 0 else "inverse"

    pm1, pm2 = st.columns(2)
    pm1.metric("Current Price", f"${current:.2f}")
    pm2.metric(f"{period} Return", f"{pct_chg:+.2f}%",
               delta=f"{pct_chg:+.2f}%", delta_color=chg_color)
    st.plotly_chart(make_etf_chart(hist, ticker, period), use_container_width=True)


def render_holdings_table(df: pd.DataFrame, ticker: str) -> None:
    styled = df.copy()
    styled.index.name = "Rank"
    styled["Weight"] = styled["Weight"].map(lambda x: f"{x:.2f}%")

    col_config = {
        "Symbol":   st.column_config.TextColumn("Symbol",   width="small"),
        "Name":     st.column_config.TextColumn("Company",  width="medium"),
        "Weight":   st.column_config.TextColumn("Weight",   width="small"),
        "Sector":   st.column_config.TextColumn("Sector",   width="small"),
        "Industry": st.column_config.TextColumn("Industry", width="medium"),
    }
    if "1Y Chart" in styled.columns:
        col_config["1Y Chart"] = st.column_config.ImageColumn("1Y Trend", width="medium")

    st.dataframe(styled, use_container_width=True, column_config=col_config)
    csv = df.drop(columns=["1Y Chart"], errors="ignore").to_csv(index_label="Rank")
    st.download_button(
        f"Download {ticker} CSV", data=csv,
        file_name=f"{ticker}_holdings.csv", mime="text/csv",
        key=f"dl_{ticker}_{id(df)}",
    )


# ════════════════════════════════════════════════════════════════════════════════
# App
# ════════════════════════════════════════════════════════════════════════════════

st.title("ETF Holdings Analyzer")
tab_single, tab_compare, tab_portfolio = st.tabs(["Single ETF", "Compare ETFs", "My Portfolio"])


# ── Tab 1: Single ETF ─────────────────────────────────────────────────────────

with tab_single:
    st.caption("Enter any ETF ticker to explore its holdings, sector allocation, and price history.")

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        s_ticker = st.text_input("ETF Ticker", placeholder="e.g. SPY, QQQ, VTI",
                                 label_visibility="collapsed", key="s_ticker_in")
    with c2:
        s_analyze = st.button("Analyze", type="primary", use_container_width=True, key="s_btn")
    with c3:
        s_top_n = st.slider("Top N holdings", 10, 100, 25, 5, key="s_n")

    if s_analyze and s_ticker:
        ticker = s_ticker.strip().upper()
        with st.spinner(f"Loading {ticker}…"):
            info, df = load_etf(ticker, s_top_n)
        st.session_state.s_data   = (info, df)
        st.session_state.s_ticker = ticker
    elif s_analyze:
        st.warning("Please enter an ETF ticker.")

    if "s_data" in st.session_state and st.session_state.s_data is not None:
        info, df = st.session_state.s_data
        ticker   = st.session_state.s_ticker

        if df is None:
            st.error("Could not retrieve holdings. Check that this is a valid ETF ticker.")
        else:
            st.subheader(info["name"])
            render_info_metrics(info)
            st.divider()

            render_price_section(ticker, "s_period")
            st.divider()

            st.subheader(f"Top {len(df)} Holdings")
            render_holdings_table(df, ticker)
            st.divider()

            ch1, ch2 = st.columns([3, 2])
            with ch1:
                st.subheader("Concentration by Holding")
                st.plotly_chart(make_bar_chart(df), use_container_width=True)
            with ch2:
                st.subheader("Sector & Industry Breakdown")
                st.caption("Select a sector below to see its industry mix.")
                render_sector_breakdown(df, "s_sector_select")

            agg = sector_agg(df)
            top5, top10 = df["Weight"].head(5).sum(), df["Weight"].head(10).sum()
            st.divider()
            st.subheader("Concentration Summary")
            s1, s2, s3 = st.columns(3)
            s1.metric("Top 5 holdings",  f"{top5:.1f}% of fund")
            s2.metric("Top 10 holdings", f"{top10:.1f}% of fund")
            ts = agg.sort_values("Weight", ascending=False).iloc[0]
            s3.metric("Largest sector", f"{ts['Sector']} ({ts['Weight']:.1f}%)")


# ── Tab 2: Compare ETFs ───────────────────────────────────────────────────────

with tab_compare:
    st.caption("Compare two ETFs side-by-side — price performance, holdings, sectors, and overlap.")

    cc1, cc2, cc3, cc4 = st.columns([2, 2, 1, 2])
    with cc1:
        c_ta = st.text_input("ETF A", placeholder="e.g. SPY",
                             label_visibility="collapsed", key="c_ta_in")
    with cc2:
        c_tb = st.text_input("ETF B", placeholder="e.g. QQQ",
                             label_visibility="collapsed", key="c_tb_in")
    with cc3:
        c_compare = st.button("Compare", type="primary", use_container_width=True, key="c_btn")
    with cc4:
        c_top_n = st.slider("Top N holdings", 10, 100, 25, 5, key="c_n")

    if c_compare and c_ta and c_tb:
        ta = c_ta.strip().upper()
        tb = c_tb.strip().upper()
        if ta == tb:
            st.warning("Enter two different tickers.")
        else:
            with st.spinner(f"Loading {ta}…"):
                info_a, df_a = load_etf(ta, c_top_n, progress_label=ta)
            with st.spinner(f"Loading {tb}…"):
                info_b, df_b = load_etf(tb, c_top_n, progress_label=tb)
            st.session_state.c_data    = (info_a, df_a, info_b, df_b)
            st.session_state.c_tickers = (ta, tb)
    elif c_compare:
        st.warning("Please enter both ETF tickers.")

    if "c_data" in st.session_state and st.session_state.c_data is not None:
        info_a, df_a, info_b, df_b = st.session_state.c_data
        ta, tb = st.session_state.c_tickers

        if df_a is None:
            st.error(f"Could not retrieve holdings for {ta}.")
        elif df_b is None:
            st.error(f"Could not retrieve holdings for {tb}.")
        else:
            # ── Fund overview table ────────────────────────────────────────────
            st.subheader("Fund Overview")
            overview = pd.DataFrame({
                "":   ["Name", "AUM", "Expense Ratio", "# Holdings"],
                ta:   [info_a["name"], info_a["aum"] or "N/A",
                       info_a["expense_ratio"] or "N/A", info_a["num_holdings"] or "N/A"],
                tb:   [info_b["name"], info_b["aum"] or "N/A",
                       info_b["expense_ratio"] or "N/A", info_b["num_holdings"] or "N/A"],
            }).set_index("")
            st.dataframe(overview, use_container_width=True,
                         column_config={
                             ta: st.column_config.TextColumn(ta, width="medium"),
                             tb: st.column_config.TextColumn(tb, width="medium"),
                         })

            st.divider()

            # ── Price comparison chart ─────────────────────────────────────────
            st.subheader("Price Performance")
            cmp_period = st.radio(
                "Period", list(PERIOD_MAP.keys()),
                horizontal=True, key="c_period", index=4,   # default 1Y
            )
            cp, civ = PERIOD_MAP[cmp_period]
            hist_a = fetch_price_history(ta, cp, civ)
            hist_b = fetch_price_history(tb, cp, civ)

            if hist_a is not None and hist_b is not None:
                pm1, pm2 = st.columns(2)
                with pm1:
                    cur_a  = hist_a["Close"].iloc[-1]
                    ret_a  = (cur_a / hist_a["Close"].iloc[0] - 1) * 100
                    st.metric(f"{ta} Price", f"${cur_a:.2f}",
                              delta=f"{ret_a:+.2f}%",
                              delta_color="normal" if ret_a >= 0 else "inverse")
                with pm2:
                    cur_b  = hist_b["Close"].iloc[-1]
                    ret_b  = (cur_b / hist_b["Close"].iloc[0] - 1) * 100
                    st.metric(f"{tb} Price", f"${cur_b:.2f}",
                              delta=f"{ret_b:+.2f}%",
                              delta_color="normal" if ret_b >= 0 else "inverse")
                st.plotly_chart(
                    make_compare_chart(hist_a, hist_b, ta, tb),
                    use_container_width=True,
                )
            else:
                st.info("Price chart unavailable — data is temporarily unavailable. "
                        "Try again in a few minutes.")

            st.divider()

            # ── Holdings side by side ──────────────────────────────────────────
            st.subheader(f"Top {c_top_n} Holdings")
            ha, hb = st.columns(2)
            with ha:
                st.markdown(f"**{ta} — {info_a['name']}**")
                render_holdings_table(df_a, ta)
            with hb:
                st.markdown(f"**{tb} — {info_b['name']}**")
                render_holdings_table(df_b, tb)

            st.divider()

            # ── Sector comparison ──────────────────────────────────────────────
            st.subheader("Sector & Industry Breakdown")
            st.caption("Hover over bars to see industry detail within each sector.")
            agg_a = sector_agg(df_a).rename(columns={"Weight": ta})
            agg_b = sector_agg(df_b).rename(columns={"Weight": tb})
            msec  = (
                agg_a.merge(agg_b, on="Sector", how="outer")
                .fillna(0)
                .sort_values(ta, ascending=False)
            )
            sector_list = msec["Sector"].tolist()
            ind_hover_a = _industry_hover_text(df_a, sector_list)
            ind_hover_b = _industry_hover_text(df_b, sector_list)

            fig_sec = go.Figure()
            fig_sec.add_trace(go.Bar(
                name=ta, x=msec["Sector"], y=msec[ta].round(2),
                marker_color="#4C72B0",
                customdata=[[t] for t in ind_hover_a],
                hovertemplate=(
                    "<b>%{x}</b><br>" + ta + ": %{y:.2f}%"
                    "<br><br><b>Industries:</b><br>%{customdata[0]}<extra></extra>"
                ),
            ))
            fig_sec.add_trace(go.Bar(
                name=tb, x=msec["Sector"], y=msec[tb].round(2),
                marker_color="#C44E52",
                customdata=[[t] for t in ind_hover_b],
                hovertemplate=(
                    "<b>%{x}</b><br>" + tb + ": %{y:.2f}%"
                    "<br><br><b>Industries:</b><br>%{customdata[0]}<extra></extra>"
                ),
            ))
            fig_sec.update_layout(
                barmode="group",
                xaxis=dict(title="Sector", tickangle=-30),
                yaxis=dict(title="Weight (%)"),
                legend=dict(orientation="h", y=1.08),
                margin=dict(l=0, r=0, t=40, b=80),
                height=420,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sec, use_container_width=True)

            sec_tbl = msec.copy()
            sec_tbl[ta] = sec_tbl[ta].map(lambda x: f"{x:.2f}%")
            sec_tbl[tb] = sec_tbl[tb].map(lambda x: f"{x:.2f}%")
            st.dataframe(sec_tbl.set_index("Sector"), use_container_width=True,
                         column_config={
                             ta: st.column_config.TextColumn(ta, width="small"),
                             tb: st.column_config.TextColumn(tb, width="small"),
                         })

            st.divider()

            # ── Shared holdings ────────────────────────────────────────────────
            st.subheader("Shared Holdings")
            st.caption("Stocks present in both ETFs' displayed holdings, ranked by combined weight.")

            overlap = (
                df_a[["Symbol", "Name", "Weight"]].rename(columns={"Weight": f"Weight in {ta}"})
                .merge(
                    df_b[["Symbol", "Weight"]].rename(columns={"Weight": f"Weight in {tb}"}),
                    on="Symbol",
                )
            )

            if overlap.empty:
                st.info("No stocks in common within the displayed top holdings.")
            else:
                overlap["Combined Weight"] = (
                    overlap[f"Weight in {ta}"] + overlap[f"Weight in {tb}"]
                )
                overlap = overlap.sort_values("Combined Weight", ascending=False).reset_index(drop=True)
                overlap.index += 1

                charts_a = df_a.set_index("Symbol")["1Y Chart"].to_dict() if "1Y Chart" in df_a.columns else {}
                charts_b = df_b.set_index("Symbol")["1Y Chart"].to_dict() if "1Y Chart" in df_b.columns else {}
                overlap["1Y Chart"] = overlap["Symbol"].map(lambda s: {**charts_b, **charts_a}.get(s))

                disp_ov = overlap.copy()
                for col in [f"Weight in {ta}", f"Weight in {tb}", "Combined Weight"]:
                    disp_ov[col] = disp_ov[col].map(lambda x: f"{x:.2f}%")

                ov_col_cfg = {
                    "Symbol":               st.column_config.TextColumn("Symbol",          width="small"),
                    "Name":                 st.column_config.TextColumn("Company",         width="medium"),
                    f"Weight in {ta}":      st.column_config.TextColumn(f"Wt {ta}",        width="small"),
                    f"Weight in {tb}":      st.column_config.TextColumn(f"Wt {tb}",        width="small"),
                    "Combined Weight":      st.column_config.TextColumn("Combined",        width="small"),
                    "1Y Chart":             st.column_config.ImageColumn("1Y Trend",       width="medium"),
                }
                st.dataframe(disp_ov, use_container_width=True, column_config=ov_col_cfg)

                ov_a = overlap[f"Weight in {ta}"].sum()
                ov_b = overlap[f"Weight in {tb}"].sum()
                st.caption(
                    f"**{len(overlap)}** stocks in common · "
                    f"Overlap accounts for {ov_a:.1f}% of {ta} "
                    f"and {ov_b:.1f}% of {tb} (within displayed top holdings)"
                )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — My Portfolio
# ════════════════════════════════════════════════════════════════════════════════

with tab_portfolio:
    st.caption(
        "Enter your holdings — ETFs and/or individual stocks — to see your true "
        "aggregate stock exposure across your whole portfolio, ranked highest to lowest."
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    p_type_label = st.radio(
        "Amount type",
        ["$ Market Value", "Number of Shares", "% of Portfolio"],
        horizontal=True,
        key="p_type",
    )
    col_label = {
        "$ Market Value":    "Market Value ($)",
        "Number of Shares":  "Shares Held",
        "% of Portfolio":    "% of Portfolio",
    }[p_type_label]

    st.markdown("**Current Holdings**")
    starter_df = pd.DataFrame({"Ticker": [""] * 6, "Amount": [0.0] * 6})
    portfolio_input = st.data_editor(
        starter_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker", help="ETF or stock symbol (e.g. SPY, QQQ, NVDA)"
            ),
            "Amount": st.column_config.NumberColumn(col_label, min_value=0, format="%.4f"),
        },
        key="p_editor",
    )

    st.markdown("**Simulate Adding (optional)** — leave blank to skip")
    sim_starter = pd.DataFrame({"Ticker": [""] * 3, "Amount": [0.0] * 3})
    sim_input = st.data_editor(
        sim_starter,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker", help="Positions you're considering adding"
            ),
            "Amount": st.column_config.NumberColumn(col_label, min_value=0, format="%.4f"),
        },
        key="sim_editor",
    )

    p_btn = st.button("Analyze Portfolio", type="primary", key="p_btn")

    if p_btn:
        type_map = {
            "$ Market Value":   "usd",
            "Number of Shares": "shares",
            "% of Portfolio":   "pct",
        }
        input_type = type_map[p_type_label]

        def _clean(df):
            out = df.copy()
            out.columns = ["Ticker", "Amount"]
            out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
            return out[(out["Ticker"].str.len() > 0) & (out["Amount"] > 0)]

        clean     = _clean(portfolio_input)
        sim_clean = _clean(sim_input)

        if len(clean) == 0:
            st.warning("Please add at least one holding with a positive amount.")
        else:
            with st.spinner("Analyzing current portfolio…"):
                result = run_portfolio_analysis(
                    tickers=clean["Ticker"].tolist(),
                    amounts=clean["Amount"].tolist(),
                    input_type=input_type,
                )
            if result is None:
                st.error("Could not compute portfolio weights — check your amounts.")
            else:
                st.session_state.portfolio_result = result
                st.session_state.p_orig_tickers   = clean["Ticker"].tolist()
                st.session_state.p_orig_amounts   = clean["Amount"].tolist()
                st.session_state.p_orig_type      = input_type

                if len(sim_clean) > 0:
                    with st.spinner("Running simulation…"):
                        sim_res = run_portfolio_analysis(
                            tickers=clean["Ticker"].tolist() + sim_clean["Ticker"].tolist(),
                            amounts=clean["Amount"].tolist() + sim_clean["Amount"].tolist(),
                            input_type=input_type,
                        )
                    st.session_state.sim_result = sim_res
                else:
                    st.session_state.sim_result = None

    # ── Results ────────────────────────────────────────────────────────────────
    if "portfolio_result" in st.session_state and st.session_state.portfolio_result:
        res       = st.session_state.portfolio_result
        exp_df    = res["exposure_df"]
        summary   = res["portfolio_summary"]
        cov_notes = res["coverage_notes"]

        st.divider()
        st.subheader("Portfolio Summary")
        sum_display = summary.copy()
        sum_display["Portfolio Weight"] = sum_display["Portfolio Weight"].map(lambda x: f"{x:.2f}%")
        st.dataframe(
            sum_display,
            use_container_width=True,
            column_config={
                "Ticker":           st.column_config.TextColumn("Ticker",  width="small"),
                "Name":             st.column_config.TextColumn("Name",    width="medium"),
                "Portfolio Weight": st.column_config.TextColumn("Weight",  width="small"),
                "Amount":           st.column_config.NumberColumn("Amount",width="small", format="%.2f"),
            },
        )

        if cov_notes:
            with st.expander("ℹ️ ETF holdings coverage"):
                st.caption(
                    "Only the top holdings from each ETF are available from stockanalysis.com. "
                    "The remainder of each ETF's weight is not broken down."
                )
                for note in cov_notes:
                    st.markdown(f"- {note}")

        st.divider()
        st.subheader(f"Aggregate Stock Exposure — {len(exp_df)} positions")
        st.caption(
            "Each row is one underlying stock. Exposure is its effective share of your total portfolio, "
            "accounting for weight inside each ETF multiplied by that ETF's portfolio weight."
        )

        exp_display = exp_df.copy()
        exp_display.index.name = "Rank"
        exp_display["Total Exposure"] = exp_display["Total Exposure"].map(lambda x: f"{x:.2f}%")

        exp_col_cfg: dict = {
            "Symbol":         st.column_config.TextColumn("Symbol",   width="small"),
            "Name":           st.column_config.TextColumn("Company",  width="medium"),
            "Total Exposure": st.column_config.TextColumn("Exposure", width="small"),
            "Breakdown":      st.column_config.TextColumn("Sources",  width="medium"),
            "Sector":         st.column_config.TextColumn("Sector",   width="small"),
            "Industry":       st.column_config.TextColumn("Industry", width="medium"),
        }
        if "1Y Chart" in exp_display.columns:
            exp_col_cfg["1Y Chart"] = st.column_config.ImageColumn("1Y Trend", width="medium")

        st.dataframe(exp_display, use_container_width=True, column_config=exp_col_cfg)

        dl_csv = exp_df.drop(columns=["1Y Chart"], errors="ignore").to_csv(index_label="Rank")
        st.download_button(
            "Download Exposure CSV", data=dl_csv,
            file_name="portfolio_exposure.csv", mime="text/csv",
        )

        st.divider()
        st.subheader("Aggregate Sector & Industry Breakdown")
        st.caption("Select a sector below to see its industry mix.")

        sector_df = exp_df[["Sector", "Industry", "Total Exposure"]].rename(
            columns={"Total Exposure": "Weight"}
        )
        render_sector_breakdown(sector_df, "p_sector_select")

        st.divider()
        st.subheader("Concentration Summary")
        sec_agg_p = (
            exp_df.groupby("Sector")["Total Exposure"]
            .sum()
            .reset_index()
            .rename(columns={"Total Exposure": "Weight"})
            .sort_values("Weight", ascending=False)
        )
        top5_e  = exp_df["Total Exposure"].head(5).sum()
        top10_e = exp_df["Total Exposure"].head(10).sum()
        top_s   = sec_agg_p.iloc[0]
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Top 5 positions",  f"{top5_e:.1f}% of portfolio")
        pc2.metric("Top 10 positions", f"{top10_e:.1f}% of portfolio")
        pc3.metric("Largest sector",   f"{top_s['Sector']} ({top_s['Weight']:.1f}%)")

        # ════════════════════════════════════════════════════════════════════
        # SIMULATION RESULTS
        # ════════════════════════════════════════════════════════════════════
        if st.session_state.get("sim_result"):
            sim_res  = st.session_state.sim_result
            orig_res = st.session_state.portfolio_result
            sim_exp  = sim_res["exposure_df"].copy()
            orig_exp = orig_res["exposure_df"].copy()

            st.divider()
            st.subheader("Simulated Portfolio")

            n_new = len(sim_res["portfolio_summary"]) - len(orig_res["portfolio_summary"])
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Original positions", len(orig_res["portfolio_summary"]))
            sm2.metric("Positions added",    n_new)
            sm3.metric("Total positions",    len(sim_res["portfolio_summary"]))

            sim_sum_disp = sim_res["portfolio_summary"].copy()
            sim_sum_disp["Portfolio Weight"] = sim_sum_disp["Portfolio Weight"].map(lambda x: f"{x:.2f}%")
            st.dataframe(
                sim_sum_disp,
                use_container_width=True,
                column_config={
                    "Ticker":           st.column_config.TextColumn("Ticker",  width="small"),
                    "Name":             st.column_config.TextColumn("Name",    width="medium"),
                    "Portfolio Weight": st.column_config.TextColumn("Weight",  width="small"),
                    "Amount":           st.column_config.NumberColumn("Amount",width="small", format="%.2f"),
                },
            )

            # ── Build comparison DataFrame ─────────────────────────────────
            comparison = (
                sim_exp[["Symbol", "Name", "Total Exposure", "Sector", "Industry", "1Y Chart"]]
                .rename(columns={"Total Exposure": "After"})
                .merge(
                    orig_exp[["Symbol", "Total Exposure"]].rename(columns={"Total Exposure": "Before"}),
                    on="Symbol", how="left",
                )
            )
            comparison["Before"] = comparison["Before"].fillna(0.0)
            comparison["Change"] = (comparison["After"] - comparison["Before"]).round(4)
            comparison = comparison.sort_values("After", ascending=False).reset_index(drop=True)
            comparison.index += 1

            new_pos   = comparison[comparison["Before"] == 0]
            increased = comparison[(comparison["Change"] > 0) & (comparison["Before"] > 0)]
            diluted   = comparison[comparison["Change"] < 0]

            km1, km2, km3 = st.columns(3)
            km1.metric("New positions",       len(new_pos))
            km2.metric("Increased positions", len(increased))
            km3.metric("Diluted positions",   len(diluted),
                       help="Existing holdings whose portfolio % shrank because new capital was added")

            top_up   = comparison[comparison["Change"] > 0].nlargest(5, "Change")
            top_down = comparison[comparison["Change"] < 0].nsmallest(5, "Change")

            if not top_up.empty or not top_down.empty:
                with st.expander("Top movers"):
                    mu, md = st.columns(2)
                    with mu:
                        st.markdown("**Biggest increases**")
                        for _, r in top_up.iterrows():
                            st.markdown(f"- **{r['Symbol']}**: {r['Before']:.2f}% → {r['After']:.2f}% "
                                        f"(+{r['Change']:.2f}%)")
                    with md:
                        st.markdown("**Most diluted**")
                        for _, r in top_down.iterrows():
                            st.markdown(f"- **{r['Symbol']}**: {r['Before']:.2f}% → {r['After']:.2f}% "
                                        f"({r['Change']:.2f}%)")

            st.subheader("Full Exposure Comparison")
            cmp_disp = comparison.copy()
            cmp_disp.index.name = "Rank"
            cmp_disp["Before"] = cmp_disp["Before"].map(
                lambda x: "—" if x == 0 else f"{x:.2f}%"
            )
            cmp_disp["After"]  = cmp_disp["After"].map(lambda x: f"{x:.2f}%")
            cmp_disp["Change"] = cmp_disp["Change"].map(
                lambda x: f"+{x:.2f}%" if x > 0.0001 else (f"{x:.2f}%" if x < -0.0001 else "—")
            )
            cmp_disp["Status"] = comparison["Before"].map(lambda x: "NEW" if x == 0 else "")

            cmp_col_cfg: dict = {
                "Symbol":  st.column_config.TextColumn("Symbol",  width="small"),
                "Name":    st.column_config.TextColumn("Company", width="medium"),
                "Before":  st.column_config.TextColumn("Before",  width="small"),
                "After":   st.column_config.TextColumn("After",   width="small"),
                "Change":  st.column_config.TextColumn("Change",  width="small"),
                "Status":  st.column_config.TextColumn("",        width="small"),
                "Sector":  st.column_config.TextColumn("Sector",  width="small"),
                "Industry":st.column_config.TextColumn("Industry",width="medium"),
            }
            if "1Y Chart" in cmp_disp.columns:
                cmp_col_cfg["1Y Chart"] = st.column_config.ImageColumn("1Y Trend", width="medium")

            cols_order = ["Symbol", "Name", "Before", "After", "Change", "Status",
                          "Sector", "Industry"] + (["1Y Chart"] if "1Y Chart" in cmp_disp.columns else [])
            st.dataframe(
                cmp_disp[cols_order],
                use_container_width=True,
                column_config=cmp_col_cfg,
            )

            dl_sim = comparison.drop(columns=["1Y Chart"], errors="ignore").to_csv(index_label="Rank")
            st.download_button(
                "Download Simulation CSV", data=dl_sim,
                file_name="portfolio_simulation.csv", mime="text/csv",
            )

            # ── Sector shift ───────────────────────────────────────────────
            st.subheader("Sector Allocation Shift")
            orig_sec = (
                orig_exp.groupby("Sector")["Total Exposure"]
                .sum().reset_index().rename(columns={"Total Exposure": "Before"})
            )
            sim_sec = (
                sim_exp.groupby("Sector")["Total Exposure"]
                .sum().reset_index().rename(columns={"Total Exposure": "After"})
            )
            sec_shift = (
                orig_sec.merge(sim_sec, on="Sector", how="outer")
                .fillna(0)
                .sort_values("After", ascending=False)
            )
            fig_shift = go.Figure()
            fig_shift.add_trace(go.Bar(
                name="Before", x=sec_shift["Sector"], y=sec_shift["Before"].round(2),
                marker_color="#8C8C8C", opacity=0.75,
                hovertemplate="<b>%{x}</b><br>Before: %{y:.2f}%<extra></extra>",
            ))
            fig_shift.add_trace(go.Bar(
                name="After", x=sec_shift["Sector"], y=sec_shift["After"].round(2),
                marker_color="#4C72B0",
                hovertemplate="<b>%{x}</b><br>After: %{y:.2f}%<extra></extra>",
            ))
            fig_shift.update_layout(
                barmode="group",
                xaxis=dict(title="Sector", tickangle=-30),
                yaxis=dict(title="Weight (%)"),
                legend=dict(orientation="h", y=1.08),
                margin=dict(l=0, r=0, t=40, b=80),
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_shift, use_container_width=True)
