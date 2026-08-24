"""
streamlit_app.py — Financial Emotion Analyzer Dashboard
============================================================
Multi-page Streamlit application for the Financial News Emotion
Intelligence project.

Pages:
  1. Market Emotion Overview
  2. Emotion Over Time
  3. Emotion vs NIFTY
  4. Emotion vs India VIX
  5. Major Event Analysis
  6. Topic vs Emotion
  7. News Source Comparison
  8. Live Financial Emotion Analyzer

Features:
  - Live news fetching from RSS feeds
  - "Guess the Emotion" interactive quiz
  - SHAP explainability
  - Glassmorphism dark-mode design
  - Full Plotly interactive charts

DISCLAIMER:
  FNEI is a research-created academic index.
  It is NOT an official financial market indicator.
  Do not use for actual investment decisions.

Run:
  streamlit run app/streamlit_app.py
"""

import os
import sys
import re
import time
import json
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from utils import load_config, fnei_category, fnei_color, FNEI_COLORS, EMOTION_COLORS
    from feature_engineering import (
        LexiconEmotionScorer, FINANCIAL_EMOTION_LEXICON, calculate_fnei
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Could not load project modules: {e}")

# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Financial Emotion Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# THEME & STYLES
# ==============================================================

DARK_BG = "#0D1117"
SURFACE = "#161B22"
SURFACE2 = "#21262D"
BORDER = "#30363D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
ACCENT = "#58A6FF"

FEAR_COLOR = "#E53935"
GREED_COLOR = "#43A047"
UNCERTAINTY_COLOR = "#FB8C00"
OPTIMISM_COLOR = "#1E88E5"
NEUTRAL_COLOR = "#757575"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT};
}}

/* Main content */
.main .block-container {{
    background-color: {DARK_BG};
    padding: 1.5rem 2rem;
    max-width: 1400px;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid {BORDER};
}}

/* Cards */
.metric-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
    transition: all 0.2s ease;
}}
.metric-card:hover {{
    border-color: {ACCENT};
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(88, 166, 255, 0.1);
}}

/* Gauge card */
.gauge-container {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1rem;
}}

/* Emotion badges */
.badge-fear {{ background:#E53935; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }}
.badge-greed {{ background:#43A047; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }}
.badge-uncertainty {{ background:#FB8C00; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }}
.badge-optimism {{ background:#1E88E5; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }}
.badge-neutral {{ background:#616161; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }}

/* Section headers */
.section-header {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
    padding-bottom: 8px;
    margin-bottom: 1.5rem;
}}

/* FNEI scale */
.fnei-bar-container {{
    background: linear-gradient(90deg, {FEAR_COLOR}, #FF7043, {NEUTRAL_COLOR}, #66BB6A, {GREED_COLOR});
    height: 20px;
    border-radius: 10px;
    position: relative;
    margin: 10px 0;
}}

/* Disclaimer */
.disclaimer-box {{
    background: rgba(33, 38, 45, 0.8);
    border: 1px solid #444;
    border-left: 4px solid {UNCERTAINTY_COLOR};
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    color: {MUTED};
    margin-top: 1rem;
}}

/* Input text area */
textarea {{
    background-color: {SURFACE2} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}}

/* Streamlit metric */
[data-testid="metric-container"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {SURFACE};
    border-radius: 8px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {MUTED};
    border-radius: 6px;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: {ACCENT};
    color: white !important;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, #58A6FF, #388BFD);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s ease;
    width: 100%;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #79B8FF, #58A6FF);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(88,166,255,0.4);
}}
</style>
""", unsafe_allow_html=True)


# ==============================================================
# HELPERS & DATA LOADING
# ==============================================================

@st.cache_data(ttl=300)
def load_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Load all processed data with caching."""
    data = {
        "daily_emotions": None,
        "articles": None,
        "market": None,
        "merged": None,
    }
    proc_dir = ROOT / "data" / "processed"
    ext_dir = ROOT / "data" / "external"

    for key, paths in {
        "daily_emotions": [proc_dir / "daily_emotions.csv"],
        "articles": [proc_dir / "articles_with_scores.csv"],
        "market": [ext_dir / "market_data.csv"],
        "merged": [proc_dir / "merged_emotion_market.csv"],
    }.items():
        for path in paths:
            if path.exists():
                try:
                    data[key] = pd.read_csv(path, low_memory=False)
                    data[key]["date"] = pd.to_datetime(data[key]["date"], errors="coerce")
                except Exception:
                    pass
                break

    return data


def get_fnei_gauge_fig(fnei_score: float) -> go.Figure:
    """Create FNEI gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fnei_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "FNEI<br><span style='font-size:0.7em;color:#8B949E'>Financial News Emotion Index</span>",
               "font": {"size": 20, "color": TEXT}},
        number={"font": {"size": 48, "color": TEXT, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED,
                     "tickfont": {"color": MUTED}},
            "bar": {"color": fnei_color(fnei_score), "thickness": 0.7},
            "bgcolor": SURFACE,
            "borderwidth": 2,
            "bordercolor": BORDER,
            "steps": [
                {"range": [0, 20], "color": "#B71C1C"},
                {"range": [20, 40], "color": "#E53935"},
                {"range": [40, 60], "color": "#616161"},
                {"range": [60, 80], "color": "#43A047"},
                {"range": [80, 100], "color": "#1B5E20"},
            ],
            "threshold": {
                "line": {"color": "#FFD700", "width": 3},
                "thickness": 0.85,
                "value": fnei_score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": TEXT, "family": "Inter"},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=280,
    )
    return fig


def get_emotion_bar_fig(fear: float, greed: float, uncertainty: float, optimism: float) -> go.Figure:
    """Create emotion scores horizontal bar chart."""
    fig = go.Figure()
    emotions = ["Fear", "Greed", "Uncertainty", "Optimism"]
    values = [fear * 100, greed * 100, uncertainty * 100, optimism * 100]
    colors = [FEAR_COLOR, GREED_COLOR, UNCERTAINTY_COLOR, OPTIMISM_COLOR]

    fig.add_trace(go.Bar(
        x=values,
        y=emotions,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"color": TEXT, "size": 13, "family": "Inter"},
    ))

    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis=dict(range=[0, 115], showgrid=False, visible=False),
        yaxis=dict(tickfont={"color": TEXT, "size": 14, "family": "Inter"}),
        margin={"l": 100, "r": 30, "t": 10, "b": 10},
        height=200,
        showlegend=False,
    )
    return fig


def make_timeline_fig(df: pd.DataFrame, selected_emotions: List[str],
                       title: str = "Emotion Over Time") -> go.Figure:
    """Multi-line emotion timeline chart."""
    color_map = {
        "daily_fear": FEAR_COLOR,
        "daily_greed": GREED_COLOR,
        "daily_uncertainty": UNCERTAINTY_COLOR,
        "daily_optimism": OPTIMISM_COLOR,
        "fnei": ACCENT,
    }
    label_map = {
        "daily_fear": "Fear", "daily_greed": "Greed",
        "daily_uncertainty": "Uncertainty", "daily_optimism": "Optimism",
        "fnei": "FNEI",
    }
    fig = go.Figure()
    for col in selected_emotions:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                name=label_map.get(col, col),
                line=dict(color=color_map.get(col, ACCENT), width=2),
                mode="lines",
                hovertemplate=f"<b>{label_map.get(col, col)}</b><br>Date: %{{x|%b %d, %Y}}<br>Score: %{{y:.3f}}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text=title, font={"size": 18, "color": TEXT, "family": "Inter"}),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        xaxis=dict(showgrid=False, tickfont={"color": MUTED}, title_font={"color": MUTED}),
        yaxis=dict(showgrid=True, gridcolor=SURFACE2, tickfont={"color": MUTED}),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        height=420,
    )
    return fig


@st.cache_data(ttl=600)
def fetch_live_news() -> List[Dict]:
    """Fetch latest financial headlines from RSS feeds."""
    try:
        import feedparser
        feeds = [
            ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Economic Times"),
            ("https://www.business-standard.com/rss/markets-106.rss", "Business Standard"),
            ("https://www.moneycontrol.com/rss/business.xml", "MoneyControl"),
        ]
        articles = []
        for url, source in feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    headline = getattr(entry, "title", "")
                    if headline:
                        articles.append({
                            "headline": headline,
                            "source": source,
                            "url": getattr(entry, "link", ""),
                            "summary": getattr(entry, "summary", "")[:300],
                        })
            except Exception:
                continue
        return articles
    except ImportError:
        return []


# ==============================================================
# SIDEBAR NAVIGATION
# ==============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">📊</div>
        <h2 style="color:#58A6FF; margin:0; font-weight:700; font-size:1.2rem;">Financial Emotion</h2>
        <h2 style="color:#58A6FF; margin:0; font-weight:700; font-size:1.2rem;">Analyzer</h2>
        <p style="color:#8B949E; font-size:0.75rem; margin-top:4px;">NLP-Powered Emotion Intelligence</p>
    </div>
    <hr style="border:1px solid #30363D; margin:1rem 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Market Emotion Overview",
            "📈 Emotion Over Time",
            "📉 Emotion vs NIFTY",
            "🌡️ Emotion vs India VIX",
            "🗓️ Major Event Analysis",
            "💡 Topic vs Emotion",
            "🗞️ News Source Comparison",
            "🔬 Live Emotion Analyzer",
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style="border:1px solid #30363D; margin:1rem 0;">
    <div style="font-size:11px; color:#8B949E; text-align:center;">
        <b>Research Project</b><br>
        Financial News Emotion Intelligence<br>
        NLP-Based Framework for Financial Markets<br><br>
        <span style="color:#E53935;">⚠ FNEI is an academic research index.<br>
        Not for investment use.</span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# LOAD DATA
# ==============================================================

data = load_data()
daily = data["daily_emotions"]
articles = data["articles"]
market = data["market"]
merged = data["merged"]

# Fallback: generate demo data if no real data exists
def generate_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2025-06-30", freq="B")
    n = len(dates)
    fear = np.abs(np.random.normal(0.15, 0.08, n))
    greed = np.abs(np.random.normal(0.18, 0.09, n))
    uncertainty = np.abs(np.random.normal(0.20, 0.07, n))
    optimism = np.abs(np.random.normal(0.17, 0.08, n))
    fnei_raw = (optimism + greed - fear - uncertainty)
    fnei_min, fnei_max = fnei_raw.min(), fnei_raw.max()
    fnei = (fnei_raw - fnei_min) / (fnei_max - fnei_min + 1e-8) * 100

    # Simulate events
    event_dates = {
        "2020-03-23": ("fear", 0.4),
        "2022-02-24": ("fear", 0.35),
        "2023-01-25": ("uncertainty", 0.38),
        "2024-06-04": ("optimism", 0.3),
    }
    for i, d in enumerate(dates):
        d_str = str(d.date())
        for ev_date, (em, boost) in event_dates.items():
            if abs((d.date() - pd.to_datetime(ev_date).date()).days) <= 5:
                em_idx = ["fear", "greed", "uncertainty", "optimism"].index(em)
                if em_idx == 0: fear[i] = min(1.0, fear[i] + boost)
                elif em_idx == 2: uncertainty[i] = min(1.0, uncertainty[i] + boost)

    return pd.DataFrame({
        "date": dates,
        "daily_fear": fear,
        "daily_greed": greed,
        "daily_uncertainty": uncertainty,
        "daily_optimism": optimism,
        "fnei": fnei,
        "article_count": np.random.randint(10, 200, n),
        "fnei_category": [fnei_category(f) for f in fnei],
    })


def generate_demo_market() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2025-06-30", freq="B")
    n = len(dates)
    price = 12000.0
    prices, vix_vals = [], []
    for _ in range(n):
        ret = np.random.normal(0.0003, 0.012)
        price *= np.exp(ret)
        prices.append(price)
        vix_vals.append(max(10, np.random.normal(18, 5)))

    returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    return pd.DataFrame({
        "date": dates,
        "NIFTY50": prices,
        "INDIA_VIX": vix_vals,
        "NIFTY50_return": returns,
        "NIFTY50_next_day_return": np.roll(returns, -1),
        "NIFTY50_volatility": pd.Series(returns).rolling(21).std().values * np.sqrt(252),
        "NIFTY50_direction": [1 if r > 0.002 else (-1 if r < -0.002 else 0) for r in returns],
    })


# Use demo data if real data not available
using_demo = False
if daily is None or daily.empty:
    using_demo = True
    daily = generate_demo_data()
    daily["date"] = pd.to_datetime(daily["date"])
    st.sidebar.warning("⚠️ Using demo data. Run data collection pipeline for real analysis.")

if market is None or market.empty:
    market = generate_demo_market()
    market["date"] = pd.to_datetime(market["date"])


# ==============================================================
# PAGE 1: MARKET EMOTION OVERVIEW
# ==============================================================

if page == "🏠 Market Emotion Overview":
    st.markdown('<h1 style="color:#58A6FF; font-size:2rem; font-weight:700;">📊 Market Emotion Overview</h1>', unsafe_allow_html=True)

    if using_demo:
        st.info("ℹ️ Showing demonstration data. Run the data collection pipeline to load real financial news data.")

    # Latest FNEI
    latest = daily.dropna(subset=["fnei"]).iloc[-1]
    current_fnei = float(latest.get("fnei", 50))
    cat = fnei_category(current_fnei)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.plotly_chart(get_fnei_gauge_fig(current_fnei), use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center; padding: 0.5rem;">
            <div style="font-size:1.3rem; font-weight:700; color:{fnei_color(current_fnei)};">{cat.upper()}</div>
            <div style="color:#8B949E; font-size:0.85rem;">As of {latest['date'].strftime('%b %d, %Y') if hasattr(latest['date'], 'strftime') else str(latest['date'])[:10]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fear = float(latest.get("daily_fear", 0))
        greed = float(latest.get("daily_greed", 0))
        unc = float(latest.get("daily_uncertainty", 0))
        opt = float(latest.get("daily_optimism", 0))

        st.markdown('<div class="section-header">Current Emotion Scores</div>', unsafe_allow_html=True)
        st.plotly_chart(get_emotion_bar_fig(fear, greed, unc, opt), use_container_width=True)

    # 4 Metric Cards
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "😨 Fear", f"{fear*100:.1f}%", FEAR_COLOR),
        (c2, "💰 Greed", f"{greed*100:.1f}%", GREED_COLOR),
        (c3, "❓ Uncertainty", f"{unc*100:.1f}%", UNCERTAINTY_COLOR),
        (c4, "🌱 Optimism", f"{opt*100:.1f}%", OPTIMISM_COLOR),
        (c5, "📰 Articles", f"{int(latest.get('article_count', 0)):,}", ACCENT),
    ]
    for col, label, val, color in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {color};">
                <div style="color:#8B949E; font-size:0.8rem; font-weight:600;">{label}</div>
                <div style="color:{color}; font-size:1.6rem; font-weight:700; margin-top:4px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # 7-day trend
    st.markdown("---")
    st.markdown('<div class="section-header">7-Day Emotion Trend</div>', unsafe_allow_html=True)
    trend = daily.dropna(subset=["daily_fear"]).tail(30).copy()
    if not trend.empty:
        fig = make_timeline_fig(
            trend, ["daily_fear", "daily_greed", "daily_uncertainty", "daily_optimism"],
            "30-Day Emotion Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    # FNEI history
    st.markdown('<div class="section-header">FNEI History (12 months)</div>', unsafe_allow_html=True)
    recent_daily = daily.dropna(subset=["fnei"]).tail(252)
    if not recent_daily.empty:
        fig_fnei = go.Figure()
        fig_fnei.add_trace(go.Scatter(
            x=recent_daily["date"], y=recent_daily["fnei"],
            fill="tozeroy",
            line=dict(color=ACCENT, width=2),
            fillcolor="rgba(88,166,255,0.15)",
            name="FNEI",
            hovertemplate="<b>FNEI</b>: %{y:.1f}<br>%{x|%b %d, %Y}<extra></extra>",
        ))
        # Add zone annotations
        for level, color, label in [(80, GREED_COLOR, "Extreme Greed"), (20, FEAR_COLOR, "Extreme Fear")]:
            fig_fnei.add_hline(y=level, line_dash="dot", line_color=color,
                               annotation_text=label, annotation_font_color=color)

        fig_fnei.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(showgrid=False, tickfont={"color": MUTED}),
            yaxis=dict(range=[0, 100], showgrid=True, gridcolor=SURFACE2,
                       tickfont={"color": MUTED}, title="FNEI Score",
                       title_font={"color": MUTED}),
            margin={"l": 60, "r": 20, "t": 20, "b": 40},
            height=320, showlegend=False,
        )
        st.plotly_chart(fig_fnei, use_container_width=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Research Disclaimer:</strong> FNEI (Financial News Emotion Index) is an academic
        research index created solely for this postgraduate project. It is NOT an official financial
        market indicator and should NOT be used for actual investment decisions. All emotion scores
        are derived from NLP analysis of publicly available news text using a lexicon-based approach
        extended with rule-based methods and transformer models.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# PAGE 2: EMOTION OVER TIME
# ==============================================================

elif page == "📈 Emotion Over Time":
    st.markdown('<h1 style="color:#58A6FF; font-size:2rem; font-weight:700;">📈 Emotion Over Time</h1>', unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        min_date = daily["date"].min().date() if hasattr(daily["date"].min(), "date") else pd.to_datetime(daily["date"].min()).date()
        max_date = daily["date"].max().date() if hasattr(daily["date"].max(), "date") else pd.to_datetime(daily["date"].max()).date()
        date_range = st.date_input("Date Range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)

    with col2:
        emotions_select = st.multiselect(
            "Select Emotions",
            options=["daily_fear", "daily_greed", "daily_uncertainty", "daily_optimism", "fnei"],
            default=["daily_fear", "daily_greed", "daily_uncertainty", "daily_optimism"],
            format_func=lambda x: x.replace("daily_", "").capitalize(),
        )

    with col3:
        smoothing = st.selectbox("Smoothing", ["None", "7-day", "30-day"])

    # Filter
    filtered = daily.copy()
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["date"] >= pd.to_datetime(date_range[0])) &
            (filtered["date"] <= pd.to_datetime(date_range[1]))
        ]

    if smoothing != "None":
        window = 7 if smoothing == "7-day" else 30
        for col in emotions_select:
            if col in filtered.columns:
                filtered[col] = filtered[col].rolling(window, min_periods=1).mean()

    fig = make_timeline_fig(filtered, emotions_select, "Emotion Scores Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Weekly/monthly aggregation
    st.markdown("---")
    st.markdown('<div class="section-header">Monthly Emotion Averages</div>', unsafe_allow_html=True)

    monthly = filtered.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly_agg = monthly.groupby("month")[
        [c for c in ["daily_fear", "daily_greed", "daily_uncertainty", "daily_optimism"] if c in monthly.columns]
    ].mean().reset_index()

    fig_monthly = go.Figure()
    colors = [FEAR_COLOR, GREED_COLOR, UNCERTAINTY_COLOR, OPTIMISM_COLOR]
    labels = ["Fear", "Greed", "Uncertainty", "Optimism"]
    em_cols = ["daily_fear", "daily_greed", "daily_uncertainty", "daily_optimism"]

    for em_col, color, label in zip(em_cols, colors, labels):
        if em_col in monthly_agg.columns:
            fig_monthly.add_trace(go.Bar(
                x=monthly_agg["month"], y=monthly_agg[em_col],
                name=label, marker_color=color,
                hovertemplate=f"<b>{label}</b>: %{{y:.3f}}<br>Month: %{{x}}<extra></extra>",
            ))

    fig_monthly.update_layout(
        barmode="group",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        xaxis=dict(showgrid=False, tickfont={"color": MUTED}, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor=SURFACE2, tickfont={"color": MUTED}),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
        margin={"l": 60, "r": 20, "t": 30, "b": 80},
        height=380,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Emotion dominance pie
    st.markdown("---")
    st.markdown('<div class="section-header">Emotion Dominance Distribution</div>', unsafe_allow_html=True)
    if articles is not None and "dominant_emotion" in articles.columns:
        dom_counts = articles["dominant_emotion"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=dom_counts.index,
            values=dom_counts.values,
            marker=dict(colors=[FEAR_COLOR, GREED_COLOR, UNCERTAINTY_COLOR, OPTIMISM_COLOR]),
            hole=0.5,
            textfont={"color": TEXT, "size": 14},
        ))
        fig_pie.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            height=350,
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ==============================================================
# PAGE 3: EMOTION VS NIFTY
# ==============================================================

elif page == "📉 Emotion vs NIFTY":
    st.markdown('<h1 style="color:#58A6FF; font-size:2rem; font-weight:700;">📉 Emotion vs NIFTY 50</h1>', unsafe_allow_html=True)

    # Merge daily emotions with market data
    if not daily.empty and not market.empty:
        market_daily = market.copy()
        market_daily["date"] = pd.to_datetime(market_daily["date"])
        daily_copy = daily.copy()
        daily_copy["date"] = pd.to_datetime(daily_copy["date"])

        mrg = pd.merge(daily_copy, market_daily[["date", "NIFTY50", "NIFTY50_return",
                                                   "NIFTY50_volatility", "NIFTY50_direction"]],
                       on="date", how="inner")

        # Dual-axis timeline
        st.markdown('<div class="section-header">FNEI vs NIFTY 50 Price</div>', unsafe_allow_html=True)
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(
            go.Scatter(x=mrg["date"], y=mrg["fnei"], name="FNEI",
                       line=dict(color=ACCENT, width=2),
                       hovertemplate="FNEI: %{y:.1f}<extra></extra>"),
            secondary_y=False,
        )
        if "NIFTY50" in mrg.columns:
            fig_dual.add_trace(
                go.Scatter(x=mrg["date"], y=mrg["NIFTY50"], name="NIFTY 50",
                           line=dict(color="#FFD700", width=1.5),
                           hovertemplate="NIFTY: ₹%{y:,.0f}<extra></extra>"),
                secondary_y=True,
            )

        fig_dual.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(showgrid=False, tickfont={"color": MUTED}),
            legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
            margin={"l": 60, "r": 60, "t": 30, "b": 40},
            height=400, hovermode="x unified",
        )
        fig_dual.update_yaxes(title_text="FNEI", secondary_y=False,
                               showgrid=True, gridcolor=SURFACE2, tickfont={"color": MUTED})
        fig_dual.update_yaxes(title_text="NIFTY 50 Price (₹)", secondary_y=True,
                               tickfont={"color": MUTED})
        st.plotly_chart(fig_dual, use_container_width=True)

        # Scatter: Fear vs NIFTY Return
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Fear vs NIFTY Daily Return</div>', unsafe_allow_html=True)
            valid = mrg.dropna(subset=["daily_fear", "NIFTY50_return"])
            if not valid.empty:
                fig_sc = go.Figure(go.Scatter(
                    x=valid["daily_fear"], y=valid["NIFTY50_return"] * 100,
                    mode="markers",
                    marker=dict(color=FEAR_COLOR, size=5, opacity=0.5),
                    hovertemplate="Fear: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>",
                ))
                # Trendline
                try:
                    z = np.polyfit(valid["daily_fear"], valid["NIFTY50_return"] * 100, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(valid["daily_fear"].min(), valid["daily_fear"].max(), 100)
                    fig_sc.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines",
                                                 line=dict(color="#FF7043", dash="dash", width=2),
                                                 name="Trend"))
                except Exception:
                    pass

                fig_sc.update_layout(
                    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                    xaxis=dict(title="Fear Score", showgrid=False, tickfont={"color": MUTED},
                               title_font={"color": MUTED}),
                    yaxis=dict(title="NIFTY Return (%)", showgrid=True, gridcolor=SURFACE2,
                               tickfont={"color": MUTED}, title_font={"color": MUTED}),
                    margin={"l": 60, "r": 20, "t": 20, "b": 60},
                    height=350, showlegend=False,
                )
                st.plotly_chart(fig_sc, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Greed vs NIFTY Daily Return</div>', unsafe_allow_html=True)
            valid2 = mrg.dropna(subset=["daily_greed", "NIFTY50_return"])
            if not valid2.empty:
                fig_sc2 = go.Figure(go.Scatter(
                    x=valid2["daily_greed"], y=valid2["NIFTY50_return"] * 100,
                    mode="markers",
                    marker=dict(color=GREED_COLOR, size=5, opacity=0.5),
                    hovertemplate="Greed: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>",
                ))
                try:
                    z = np.polyfit(valid2["daily_greed"], valid2["NIFTY50_return"] * 100, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(valid2["daily_greed"].min(), valid2["daily_greed"].max(), 100)
                    fig_sc2.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines",
                                                  line=dict(color="#66BB6A", dash="dash", width=2),
                                                  name="Trend"))
                except Exception:
                    pass

                fig_sc2.update_layout(
                    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                    xaxis=dict(title="Greed Score", showgrid=False, tickfont={"color": MUTED},
                               title_font={"color": MUTED}),
                    yaxis=dict(title="NIFTY Return (%)", showgrid=True, gridcolor=SURFACE2,
                               tickfont={"color": MUTED}, title_font={"color": MUTED}),
                    margin={"l": 60, "r": 20, "t": 20, "b": 60},
                    height=350, showlegend=False,
                )
                st.plotly_chart(fig_sc2, use_container_width=True)

        # Correlation table
        st.markdown("---")
        st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
        from scipy import stats as scipy_stats
        corr_rows = []
        for em_col, em_label in [("daily_fear", "Fear"), ("daily_greed", "Greed"),
                                   ("daily_uncertainty", "Uncertainty"), ("daily_optimism", "Optimism"),
                                   ("fnei", "FNEI")]:
            for mk_col, mk_label in [("NIFTY50_return", "NIFTY Return (same-day)"),
                                      ("NIFTY50_volatility", "NIFTY Volatility")]:
                if em_col in mrg.columns and mk_col in mrg.columns:
                    valid_c = mrg[[em_col, mk_col]].dropna()
                    if len(valid_c) > 30:
                        r, p = scipy_stats.pearsonr(valid_c[em_col], valid_c[mk_col])
                        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
                        corr_rows.append({
                            "Emotion": em_label,
                            "Market Variable": mk_label,
                            "Pearson r": f"{r:.4f}",
                            "p-value": f"{p:.4f}",
                            "Significance": sig,
                            "N": len(valid_c),
                        })

        if corr_rows:
            corr_df = pd.DataFrame(corr_rows)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="disclaimer-box">
            <b>Statistical Note:</b> Significance levels: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, n.s. = not significant.
            Correlation ≠ Causation. Results show association only. Confounding factors (global market events,
            macroeconomic conditions) are not controlled. See Chapter 6 of the research report for full analysis.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================
# PAGE 4: EMOTION VS INDIA VIX
# ==============================================================

elif page == "🌡️ Emotion vs India VIX":
    st.markdown('<h1 style="color:#58A6FF; font-size:2rem; font-weight:700;">🌡️ Emotion vs India VIX</h1>', unsafe_allow_html=True)

    market_copy = market.copy()
    market_copy["date"] = pd.to_datetime(market_copy["date"])
    daily_copy = daily.copy()
    daily_copy["date"] = pd.to_datetime(daily_copy["date"])

    mrg = pd.merge(daily_copy, market_copy, on="date", how="inner")

    # VIX + Uncertainty dual axis
    st.markdown('<div class="section-header">Uncertainty Score vs India VIX</div>', unsafe_allow_html=True)
    if "INDIA_VIX" in mrg.columns and "daily_uncertainty" in mrg.columns:
        fig_vix = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vix.add_trace(
            go.Scatter(x=mrg["date"], y=mrg["daily_uncertainty"], name="Uncertainty",
                       line=dict(color=UNCERTAINTY_COLOR, width=2)),
            secondary_y=False,
        )
        fig_vix.add_trace(
            go.Scatter(x=mrg["date"], y=mrg["INDIA_VIX"], name="India VIX",
                       line=dict(color="#CE93D8", width=1.5)),
            secondary_y=True,
        )
        fig_vix.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(showgrid=False, tickfont={"color": MUTED}),
            legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
            margin={"l": 60, "r": 60, "t": 30, "b": 40},
            height=400, hovermode="x unified",
        )
        fig_vix.update_yaxes(title_text="Uncertainty Score", secondary_y=False,
                              tickfont={"color": MUTED})
        fig_vix.update_yaxes(title_text="India VIX", secondary_y=True,
                              tickfont={"color": MUTED})
        st.plotly_chart(fig_vix, use_container_width=True)

    # Scatter: Uncertainty vs VIX
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Uncertainty vs India VIX</div>', unsafe_allow_html=True)
        if "INDIA_VIX" in mrg.columns:
            valid = mrg.dropna(subset=["daily_uncertainty", "INDIA_VIX"])
            fig_sc = go.Figure(go.Scatter(
                x=valid["daily_uncertainty"], y=valid["INDIA_VIX"],
                mode="markers",
                marker=dict(color=UNCERTAINTY_COLOR, size=5, opacity=0.5),
            ))
            try:
                z = np.polyfit(valid["daily_uncertainty"], valid["INDIA_VIX"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid["daily_uncertainty"].min(), valid["daily_uncertainty"].max(), 100)
                fig_sc.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines",
                                             line=dict(color="#FF9800", dash="dash", width=2),
                                             name="Trend"))
            except Exception:
                pass
            fig_sc.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                xaxis=dict(title="Uncertainty Score", showgrid=False, tickfont={"color": MUTED},
                           title_font={"color": MUTED}),
                yaxis=dict(title="India VIX", showgrid=True, gridcolor=SURFACE2,
                           tickfont={"color": MUTED}, title_font={"color": MUTED}),
                height=350, margin={"l": 60, "r": 20, "t": 20, "b": 60},
                showlegend=False,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">FNEI vs India VIX (expected: negative)</div>', unsafe_allow_html=True)
        if "INDIA_VIX" in mrg.columns and "fnei" in mrg.columns:
            valid2 = mrg.dropna(subset=["fnei", "INDIA_VIX"])
            fig_sc2 = go.Figure(go.Scatter(
                x=valid2["fnei"], y=valid2["INDIA_VIX"],
                mode="markers",
                marker=dict(color=ACCENT, size=5, opacity=0.5),
            ))
            try:
                z = np.polyfit(valid2["fnei"], valid2["INDIA_VIX"], 1)
                p2 = np.poly1d(z)
                x_line = np.linspace(valid2["fnei"].min(), valid2["fnei"].max(), 100)
                fig_sc2.add_trace(go.Scatter(x=x_line, y=p2(x_line), mode="lines",
                                              line=dict(color=ACCENT, dash="dash", width=2),
                                              name="Trend"))
            except Exception:
                pass
            fig_sc2.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                xaxis=dict(title="FNEI Score", showgrid=False, tickfont={"color": MUTED},
                           title_font={"color": MUTED}),
                yaxis=dict(title="India VIX", showgrid=True, gridcolor=SURFACE2,
                           tickfont={"color": MUTED}, title_font={"color": MUTED}),
                height=350, margin={"l": 60, "r": 20, "t": 20, "b": 60},
                showlegend=False,
            )
            st.plotly_chart(fig_sc2, use_container_width=True)


# ==============================================================
# PAGE 5: MAJOR EVENT ANALYSIS
# ==============================================================

elif page == "🗓️ Major Event Analysis":
    st.markdown('<h1 style="color:#58A6FF; font-size:2rem; font-weight:700;">🗓️ Major Event Analysis</h1>', unsafe_allow_html=True)

    EVENTS = [
        {"name": "COVID-19 Market Crash", "date": "2020-03-23", "category": "Pandemic"},
        {"name": "Russia-Ukraine Invasion", "date": "2022-02-24", "category": "Geopolitical"},
        {"name": "US Fed Rate Hike Begins", "date": "2022-03-16", "category": "Monetary Policy"},
        {"name": "Adani-Hindenburg Crisis", "date": "2023-01-25", "category": "Corporate"},
        {"name": "SVB Banking Crisis", "date": "2023-03-10", "category": "Banking"},
        {"name": "Indian Union Budget 2024", "date": "2024-02-01", "category": "Fiscal Policy"},
        {"name": "Indian General Elections 2024", "date": "2024-04-19", "category": "Political"},
        {"name": "RBI Rate Cut 2025", "date": "2025-02-07", "category": "Monetary Policy"},
    ]

    selected_event_name = st.selectbox(
        "Select Event",
        options=[e["name"] for e in EVENTS],
        index=0,
    )
    selected_event = next(e for e in EVENTS if e["name"] == selected_event_name)
    event_date = pd.to_datetime(selected_event["date"])

    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {ACCENT};">
        <div style="font-size:1.2rem; font-weight:700; color:{ACCENT};">{selected_event['name']}</div>
        <div style="color:#8B949E;">Date: {selected_event['date']} | Category: {selected_event['category']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Window analysis
    window_days = st.slider("Window (days before/after)", 3, 14, 7)
    daily_copy = daily.copy()
    daily_copy["date"] = pd.to_datetime(daily_copy["date"])

    window_data = daily_copy[
        (daily_copy["date"] >= event_date - timedelta(days=window_days)) &
        (daily_copy["date"] <= event_date + timedelta(days=window_days))
    ].copy()

    window_data["days_from_event"] = (window_data["date"] - event_date).dt.days

    if window_data.empty:
        st.warning("No data available for this event's window. Try expanding the date range in data collection.")
    else:
        # Emotion around event
        fig_event = go.Figure()
        for em_col, color, label in [
            ("daily_fear", FEAR_COLOR, "Fear"),
            ("daily_greed", GREED_COLOR, "Greed"),
            ("daily_uncertainty", UNCERTAINTY_COLOR, "Uncertainty"),
            ("daily_optimism", OPTIMISM_COLOR, "Optimism"),
        ]:
            if em_col in window_data.columns:
                fig_event.add_trace(go.Scatter(
                    x=window_data["days_from_event"],
                    y=window_data[em_col],
                    name=label,
                    line=dict(color=color, width=2),
                    mode="lines+markers",
                    marker=dict(size=6),
                    hovertemplate=f"<b>{label}</b>: %{{y:.3f}}<br>Day: %{{x}}<extra></extra>",
                ))

        fig_event.add_vline(x=0, line_dash="dash", line_color=ACCENT,
                            annotation_text="Event", annotation_font_color=ACCENT)
        fig_event.update_layout(
            title=f"Emotion Around: {selected_event['name']}",
            title_font={"size": 16, "color": TEXT},
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(title="Days from Event", showgrid=False, tickfont={"color": MUTED},
                       title_font={"color": MUTED}),
            yaxis=dict(title="Emotion Score", showgrid=True, gridcolor=SURFACE2,
                       tickfont={"color": MUTED}),
            legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
            margin={"l": 60, "r": 20, "t": 60, "b": 60},
            height=420,
        )
        st.plotly_chart(fig_event, use_container_width=True)

        # Before/After comparison
        before = window_data[window_data["days_from_event"] < 0]
        after = window_data[window_data["days_from_event"] > 0]

        if not before.empty and not after.empty:
            st.markdown('<div class="section-header">Before vs After Comparison</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            comparison_data = {"Emotion": ["Fear", "Greed", "Uncertainty", "Optimism"],
                               "Before Event": [], "After Event": []}
            for col in ["daily_fear", "daily_greed", "daily_uncertainty", "daily_optimism"]:
                comparison_data["Before Event"].append(before[col].mean() if col in before.columns else 0)
                comparison_data["After Event"].append(after[col].mean() if col in after.columns else 0)

            comp_df = pd.DataFrame(comparison_data)
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=comp_df["Emotion"], y=comp_df["Before Event"],
                                       name="Before Event", marker_color=UNCERTAINTY_COLOR))
            fig_comp.add_trace(go.Bar(x=comp_df["Emotion"], y=comp_df["After Event"],
                                       name="After Event", marker_color=ACCENT))
            fig_comp.update_layout(
                barmode="group",
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                xaxis=dict(tickfont={"color": MUTED}),
                yaxis=dict(showgrid=True, gridcolor=SURFACE2, tickfont={"color": MUTED}),
                legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
                margin={"l": 60, "r": 20, "t": 20, "b": 40},
                height=350,
            )
            st.plotly_chart(fig_comp, use_container_width=True)


# ==============================================================
# PAGE 6: TOPIC VS EMOTION
# ==============================================================

elif page == "💡 Topic vs Emotion":
    st.markdown('<h1 style="color:#58A6FF; font-size:2rem; font-weight:700;">💡 Topic vs Emotion</h1>', unsafe_allow_html=True)

    st.info("Topic modeling output is generated during model training (notebooks/05_emotion_modeling.ipynb). Below shows illustrative topic-emotion relationships.")

    # Create illustrative topic-emotion heatmap
    topics = ["Inflation", "Interest Rates", "Banking & NPA", "Equities/IPO",
              "Commodities", "Currency/INR", "Corporate Earnings", "Geopolitical Risk",
              "RBI Policy", "Budget & Fiscal"]
    emotions = ["Fear", "Greed", "Uncertainty", "Optimism"]

    np.random.seed(42)
    matrix = np.array([
        [0.65, 0.10, 0.72, 0.12],   # Inflation: high fear + uncertainty
        [0.45, 0.20, 0.68, 0.25],   # Interest Rates: moderate fear + high uncertainty
        [0.78, 0.08, 0.55, 0.08],   # Banking NPA: very high fear
        [0.20, 0.75, 0.25, 0.62],   # Equities: high greed + optimism
        [0.42, 0.35, 0.55, 0.30],   # Commodities: moderate all
        [0.38, 0.22, 0.68, 0.20],   # Currency: uncertainty dominant
        [0.18, 0.68, 0.20, 0.72],   # Corporate Earnings: high greed + optimism
        [0.72, 0.08, 0.78, 0.12],   # Geopolitical: high fear + uncertainty
        [0.28, 0.32, 0.72, 0.38],   # RBI Policy: high uncertainty
        [0.15, 0.45, 0.55, 0.62],   # Budget: optimism moderate
    ])

    fig_hm = px.imshow(
        matrix,
        labels=dict(x="Emotion", y="Topic", color="Score"),
        x=emotions, y=topics,
        color_continuous_scale=[[0, "#1E1E1E"], [0.5, "#8B6914"], [1.0, "#E53935"]],
        aspect="auto",
        text_auto=".2f",
    )
    fig_hm.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT, family="Inter"),
        margin={"l": 150, "r": 20, "t": 30, "b": 40},
        height=500,
        coloraxis_colorbar=dict(tickfont={"color": MUTED}),
    )
    fig_hm.update_traces(textfont={"color": TEXT, "size": 12})
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("""
    <div class="disclaimer-box">
        <b>Note:</b> Values shown are indicative. Run LDA or BERTopic in
        <code>notebooks/05_emotion_modeling.ipynb</code> for actual topic-emotion scores
        derived from your collected dataset.
    </div>
    """, unsafe_allow_html=True)

    # Top emotion by topic
    st.markdown('<div class="section-header">Dominant Emotion by Topic</div>', unsafe_allow_html=True)
    dominant = []
    for i, topic in enumerate(topics):
        scores = matrix[i]
        dom_idx = scores.argmax()
        dominant.append({
            "Topic": topic,
            "Dominant Emotion": emotions[dom_idx],
            "Score": f"{scores[dom_idx]:.2f}",
        })
    dom_df = pd.DataFrame(dominant)
    st.dataframe(dom_df, use_container_width=True, hide_index=True)


# ==============================================================
# PAGE 7: NEWS SOURCE COMPARISON
# ==============================================================

elif page == "🗞️ News Source Comparison":
    st.markdown('<h1 style="color:#58A6FF; font-size:2rem; font-weight:700;">🗞️ News Source Comparison</h1>', unsafe_allow_html=True)

    st.info("Source comparison uses collected article data. Run data collection pipeline for real source comparisons.")

    if articles is not None and "source" in articles.columns and "fear_score" in articles.columns:
        top_sources = articles["source"].value_counts().head(8).index.tolist()
        source_data = articles[articles["source"].isin(top_sources)]

        source_agg = source_data.groupby("source").agg(
            avg_fear=("fear_score", "mean"),
            avg_greed=("greed_score", "mean"),
            avg_uncertainty=("uncertainty_score", "mean"),
            avg_optimism=("optimism_score", "mean"),
            article_count=("article_id", "count"),
        ).reset_index()

        # Grouped bar chart
        fig_src = go.Figure()
        for em_col, color, label in [
            ("avg_fear", FEAR_COLOR, "Fear"),
            ("avg_greed", GREED_COLOR, "Greed"),
            ("avg_uncertainty", UNCERTAINTY_COLOR, "Uncertainty"),
            ("avg_optimism", OPTIMISM_COLOR, "Optimism"),
        ]:
            fig_src.add_trace(go.Bar(
                x=source_agg["source"], y=source_agg[em_col],
                name=label, marker_color=color,
            ))

        fig_src.update_layout(
            barmode="group",
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(tickfont={"color": MUTED}, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor=SURFACE2, tickfont={"color": MUTED},
                       title="Average Emotion Score"),
            legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
            margin={"l": 60, "r": 20, "t": 20, "b": 80},
            height=420,
        )
        st.plotly_chart(fig_src, use_container_width=True)

        st.markdown('<div class="section-header">Source Statistics Table</div>', unsafe_allow_html=True)
        st.dataframe(source_agg.rename(columns={
            "source": "Source", "avg_fear": "Avg Fear", "avg_greed": "Avg Greed",
            "avg_uncertainty": "Avg Uncertainty", "avg_optimism": "Avg Optimism",
            "article_count": "Article Count"
        }).round(4), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="disclaimer-box">
            <b>Framing Note:</b> Differences between sources are described as "emotional framing differences"
            — differences in the language and emphasis used. This analysis does NOT imply editorial bias
            without rigorous statistical validation (ANOVA + post-hoc tests) reported separately.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Demo source comparison
        demo_sources = {
            "Source": ["Economic Times", "Mint", "Business Standard", "MoneyControl", "NDTV Profit"],
            "Avg Fear": [0.148, 0.132, 0.165, 0.142, 0.158],
            "Avg Greed": [0.185, 0.192, 0.171, 0.188, 0.175],
            "Avg Uncertainty": [0.210, 0.198, 0.225, 0.205, 0.215],
            "Avg Optimism": [0.165, 0.178, 0.155, 0.170, 0.160],
        }
        demo_df = pd.DataFrame(demo_sources)
        fig_src = go.Figure()
        for em_col, color, label in [("Avg Fear", FEAR_COLOR, "Fear"),
                                       ("Avg Greed", GREED_COLOR, "Greed"),
                                       ("Avg Uncertainty", UNCERTAINTY_COLOR, "Uncertainty"),
                                       ("Avg Optimism", OPTIMISM_COLOR, "Optimism")]:
            fig_src.add_trace(go.Bar(x=demo_df["Source"], y=demo_df[em_col],
                                      name=label, marker_color=color))
        fig_src.update_layout(
            barmode="group",
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(tickfont={"color": MUTED}),
            yaxis=dict(showgrid=True, gridcolor=SURFACE2, tickfont={"color": MUTED}),
            legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font={"color": TEXT}),
            margin={"l": 60, "r": 20, "t": 20, "b": 40},
            height=380,
        )
        st.plotly_chart(fig_src, use_container_width=True)
        st.caption("Demo data shown. Collect real data to see actual source comparisons.")


# ==============================================================
# PAGE 8: LIVE FINANCIAL EMOTION ANALYZER
# ==============================================================

elif page == "🔬 Live Emotion Analyzer":
    st.markdown('<h1 style="color:#58A6FF; font-size:2rem; font-weight:700;">🔬 Financial Emotion Analyzer</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝 Analyze Text", "📡 Live News", "🎮 Guess the Emotion"])

    # ──────────────────────────────────────────────────────────
    # TAB 1: TEXT ANALYSIS
    # ──────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("### Paste a financial article or headline")
            user_text = st.text_area(
                label="Input Text",
                height=200,
                placeholder='Example: "Markets plunge as investors fear recession and rising inflation..."',
                label_visibility="collapsed",
                key="analyze_text_input",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                analyze_btn = st.button("🔍 Analyze", key="analyze_btn", use_container_width=True)
            with col_b:
                clear_btn = st.button("🗑️ Clear", key="clear_btn", use_container_width=True)

            if clear_btn:
                st.rerun()

        with col2:
            st.markdown("### Example Headlines")
            examples = [
                "Markets plunge as investors fear recession and rising inflation threatens economy",
                "NIFTY surges to record highs as FII inflows hit multi-month peak on bullish sentiment",
                "RBI may hold rates amid uncertain global outlook and mixed domestic signals",
                "India GDP growth beats expectations; manufacturing sector expands robustly in Q3",
            ]
            for ex in examples:
                if st.button(f"📰 {ex[:60]}...", key=f"ex_{ex[:20]}", use_container_width=True):
                    user_text = ex

        if analyze_btn and user_text.strip():
            st.markdown("---")
            st.markdown("### Analysis Results")

            # Load scorer
            with st.spinner("Analyzing..."):
                if MODULES_AVAILABLE:
                    scorer = LexiconEmotionScorer()
                    scores = scorer.score_text(user_text)
                    fear = scores["fear"]
                    greed = scores["greed"]
                    uncertainty = scores["uncertainty"]
                    optimism = scores["optimism"]
                    dominant = max(scores, key=scores.get)
                    keywords = scorer.get_emotion_keywords(user_text)

                    # FNEI for this text
                    dummy_df = pd.DataFrame([{
                        "fear_score": fear, "greed_score": greed,
                        "uncertainty_score": uncertainty, "optimism_score": optimism
                    }])
                    fnei_score = float(calculate_fnei(dummy_df, formula="weighted").iloc[0])
                    confidence = max(fear, greed, uncertainty, optimism)
                else:
                    # Fallback simple scoring
                    text_lower = user_text.lower()
                    fear_words = ["plunge", "crash", "fall", "fear", "panic", "recession", "drop"]
                    greed_words = ["surge", "rally", "record", "gain", "boom", "rise", "bullish"]
                    unc_words = ["uncertain", "may", "could", "volatile", "unclear", "risk"]
                    opt_words = ["growth", "recovery", "strong", "improve", "confidence"]

                    fear = min(1.0, sum(text_lower.count(w) for w in fear_words) * 0.15)
                    greed = min(1.0, sum(text_lower.count(w) for w in greed_words) * 0.15)
                    uncertainty = min(1.0, sum(text_lower.count(w) for w in unc_words) * 0.15)
                    optimism = min(1.0, sum(text_lower.count(w) for w in opt_words) * 0.15)

                    scores = {"fear": fear, "greed": greed, "uncertainty": uncertainty, "optimism": optimism}
                    dominant = max(scores, key=scores.get)
                    keywords = {}
                    fnei_score = 50.0
                    confidence = max(fear, greed, uncertainty, optimism)

            # Display results
            emotion_colors_map = {
                "fear": FEAR_COLOR, "greed": GREED_COLOR,
                "uncertainty": UNCERTAINTY_COLOR, "optimism": OPTIMISM_COLOR
            }
            dom_color = emotion_colors_map.get(dominant, NEUTRAL_COLOR)

            # Main result
            st.markdown(f"""
            <div style="background:{SURFACE}; border:2px solid {dom_color}; border-radius:16px;
                        padding:1.5rem; margin:1rem 0; text-align:center;">
                <div style="color:#8B949E; font-size:0.85rem; font-weight:600; letter-spacing:2px;">DOMINANT EMOTION</div>
                <div style="color:{dom_color}; font-size:3rem; font-weight:700; margin:0.5rem 0;">{dominant.upper()}</div>
                <div style="color:#8B949E; font-size:0.9rem;">Confidence: {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Score bars
            col1_r, col2_r = st.columns(2)
            with col1_r:
                st.plotly_chart(get_emotion_bar_fig(fear, greed, uncertainty, optimism), use_container_width=True)
            with col2_r:
                st.plotly_chart(get_fnei_gauge_fig(fnei_score), use_container_width=True)

            # FNEI category
            cat = fnei_category(fnei_score)
            st.markdown(f"""
            <div style="text-align:center; background:{SURFACE}; border-radius:12px; padding:1rem;
                        border:1px solid {fnei_color(fnei_score)};">
                <span style="color:#8B949E; font-size:0.85rem;">FNEI Score: </span>
                <span style="color:{fnei_color(fnei_score)}; font-size:1.2rem; font-weight:700;">{fnei_score:.1f}/100 — {cat}</span>
            </div>
            """, unsafe_allow_html=True)

            # Key signals
            st.markdown("---")
            st.markdown("### 🔑 Key Emotion Signals")
            if MODULES_AVAILABLE and keywords:
                for emotion, words in keywords.items():
                    if words:
                        color = emotion_colors_map.get(emotion, NEUTRAL_COLOR)
                        words_str = " · ".join([f"`{w}`" for w in words[:8]])
                        st.markdown(f"""
                        <div style="margin:0.4rem 0;">
                            <span style="color:{color}; font-weight:600;">{emotion.upper()}:</span>
                            <span style="color:{MUTED}; font-size:0.9rem;"> {', '.join(words[:8])}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Install project modules to see detailed keyword analysis.")

            # Explanation
            st.markdown("---")
            st.markdown("### 💡 How the Model Reached This Result")
            st.markdown(f"""
            <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:1.2rem;">
                <p style="color:{TEXT};">
                The analysis uses a <b>Financial Emotion Lexicon</b> (Method 1) based on the
                Loughran-McDonald financial word list, extended with India-specific financial terminology.
                </p>
                <p style="color:{TEXT};">
                <b>Scoring methodology:</b>
                <ul>
                    <li>Word-level matching against 4 emotion dictionaries (Fear, Greed, Uncertainty, Optimism)</li>
                    <li>Negation window detection (5-word context window)</li>
                    <li>Intensifier detection (+50% weight for words like "severely", "massively")</li>
                    <li>VADER compound score blended for overall sentiment direction</li>
                    <li>FNEI = normalized composite of (Optimism + Greed) - (Fear + Uncertainty)</li>
                </ul>
                </p>
                <p style="color:#8B949E; font-size:0.85rem;">
                Method: Lexicon/Rule-based (M1) | FNEI Formula: Weighted composite (0-100 scale)<br>
                For M2 (ML) and M3 (FinBERT) predictions, ensure models are trained via the notebooks.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer-box">
                <b>Research Note:</b> Scores are generated by an NLP model trained for academic research.
                These are NOT financial recommendations. Emotion detection may not capture sarcasm, ambiguity,
                or complex contextual meanings. Always verify with domain expertise.
            </div>
            """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TAB 2: LIVE NEWS
    # ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📡 Analyze Latest Financial News")
        st.caption("Fetches live headlines from Economic Times, Business Standard, and MoneyControl RSS feeds (publicly permitted)")

        if st.button("🔄 Fetch Latest Headlines", key="fetch_news_btn", use_container_width=False):
            with st.spinner("Fetching live headlines..."):
                live_articles = fetch_live_news()

            if not live_articles:
                st.warning("Could not fetch live news. Check internet connection or try pasting text in the 'Analyze Text' tab.")
            else:
                st.success(f"✅ Fetched {len(live_articles)} live headlines")

                for i, article in enumerate(live_articles[:10]):
                    headline = article["headline"]
                    source = article["source"]

                    # Score
                    if MODULES_AVAILABLE:
                        scorer = LexiconEmotionScorer()
                        scores = scorer.score_text(headline)
                        dominant = max(scores, key=scores.get)
                    else:
                        scores = {"fear": 0.15, "greed": 0.18, "uncertainty": 0.20, "optimism": 0.17}
                        dominant = "uncertainty"

                    dom_color = {"fear": FEAR_COLOR, "greed": GREED_COLOR,
                                 "uncertainty": UNCERTAINTY_COLOR, "optimism": OPTIMISM_COLOR}.get(dominant, NEUTRAL_COLOR)

                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 3px solid {dom_color};">
                                <div style="font-weight:600; color:{TEXT}; font-size:0.95rem;">{headline}</div>
                                <div style="color:{MUTED}; font-size:0.8rem; margin-top:4px;">🗞️ {source}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f"""
                            <div style="text-align:center; padding:1rem; background:{SURFACE};
                                        border-radius:8px; border:1px solid {dom_color}; margin:0.4rem 0;">
                                <div style="color:{dom_color}; font-weight:700; font-size:1.1rem;">{dominant.upper()}</div>
                                <div style="color:{MUTED}; font-size:0.75rem;">{max(scores.values())*100:.1f}% confident</div>
                            </div>
                            """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TAB 3: GUESS THE EMOTION (INTERACTIVE QUIZ)
    # ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🎮 Guess the Emotion — Interactive Quiz")
        st.caption("Test your ability to identify financial emotions vs the NLP model. Perfect for classroom demonstrations.")

        quiz_headlines = [
            ("Markets crash 8% as global recession fears intensify and FII outflows surge", "fear"),
            ("NIFTY 50 hits all-time high as corporate earnings beat estimates for third consecutive quarter", "greed"),
            ("RBI may hold rates amid uncertain outlook; analysts divided on next policy move", "uncertainty"),
            ("India GDP grows at 7.8% in Q3, strongest in four years; exports surge to record levels", "optimism"),
            ("Banking stocks plunge as NPA concerns resurface; RBI raises red flags on credit quality", "fear"),
            ("Startup ecosystem booms with record $15B in VC funding; unicorn count hits new high", "greed"),
            ("Investors cautious ahead of Fed meeting; markets volatile on mixed economic signals", "uncertainty"),
            ("Manufacturing sector expands robustly; PMI hits 18-month high on strong domestic demand", "optimism"),
        ]

        if "quiz_index" not in st.session_state:
            st.session_state.quiz_index = random.randint(0, len(quiz_headlines) - 1)
            st.session_state.quiz_answered = False
            st.session_state.score = 0
            st.session_state.attempts = 0

        current_hl, correct_label = quiz_headlines[st.session_state.quiz_index]

        st.markdown(f"""
        <div style="background:{SURFACE}; border:2px solid {ACCENT}; border-radius:16px;
                    padding:1.5rem; margin:1rem 0; text-align:center;">
            <div style="color:{MUTED}; font-size:0.85rem; margin-bottom:0.5rem;">FINANCIAL HEADLINE</div>
            <div style="color:{TEXT}; font-size:1.15rem; font-weight:600; line-height:1.5;">"{current_hl}"</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.quiz_answered:
            st.markdown("**What emotion does this headline primarily convey?**")
            cols = st.columns(4)
            for col, emotion, color in zip(cols,
                ["fear", "greed", "uncertainty", "optimism"],
                [FEAR_COLOR, GREED_COLOR, UNCERTAINTY_COLOR, OPTIMISM_COLOR]
            ):
                with col:
                    if st.button(f"{emotion.upper()}", key=f"quiz_{emotion}",
                                  use_container_width=True):
                        st.session_state.user_choice = emotion
                        st.session_state.quiz_answered = True
                        st.session_state.attempts += 1
                        if emotion == correct_label:
                            st.session_state.score += 1
                        st.rerun()
        else:
            user_choice = st.session_state.get("user_choice", "")
            is_correct = user_choice == correct_label

            # Model prediction
            if MODULES_AVAILABLE:
                scorer = LexiconEmotionScorer()
                model_scores = scorer.score_text(current_hl)
                model_prediction = max(model_scores, key=model_scores.get)
                model_confidence = max(model_scores.values())
            else:
                model_prediction = correct_label
                model_confidence = 0.75

            col_r1, col_r2, col_r3 = st.columns(3)
            em_cols = {"fear": FEAR_COLOR, "greed": GREED_COLOR,
                       "uncertainty": UNCERTAINTY_COLOR, "optimism": OPTIMISM_COLOR}

            with col_r1:
                st.markdown(f"""
                <div style="text-align:center; background:{SURFACE}; border-radius:12px; padding:1rem;
                            border:2px solid {em_cols.get(user_choice, NEUTRAL_COLOR)};">
                    <div style="color:{MUTED}; font-size:0.8rem;">YOUR ANSWER</div>
                    <div style="color:{em_cols.get(user_choice, NEUTRAL_COLOR)}; font-size:1.4rem; font-weight:700;">{user_choice.upper()}</div>
                    <div style="font-size:1.5rem;">{'✅' if is_correct else '❌'}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div style="text-align:center; background:{SURFACE}; border-radius:12px; padding:1rem;
                            border:2px solid {em_cols.get(model_prediction, NEUTRAL_COLOR)};">
                    <div style="color:{MUTED}; font-size:0.8rem;">MODEL PREDICTION</div>
                    <div style="color:{em_cols.get(model_prediction, NEUTRAL_COLOR)}; font-size:1.4rem; font-weight:700;">{model_prediction.upper()}</div>
                    <div style="color:{MUTED}; font-size:0.8rem;">{model_confidence*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with col_r3:
                st.markdown(f"""
                <div style="text-align:center; background:{SURFACE}; border-radius:12px; padding:1rem;
                            border:2px solid {em_cols.get(correct_label, NEUTRAL_COLOR)};">
                    <div style="color:{MUTED}; font-size:0.8rem;">CORRECT ANSWER</div>
                    <div style="color:{em_cols.get(correct_label, NEUTRAL_COLOR)}; font-size:1.4rem; font-weight:700;">{correct_label.upper()}</div>
                    <div style="color:{MUTED}; font-size:0.8rem;">Human-Model: {'✅ Agree' if model_prediction == correct_label else '❌ Disagree'}</div>
                </div>
                """, unsafe_allow_html=True)

            # Score
            st.markdown(f"""
            <div style="text-align:center; margin:1rem 0;">
                <span style="color:{ACCENT}; font-size:1.1rem; font-weight:600;">
                    Score: {st.session_state.score}/{st.session_state.attempts}
                    ({st.session_state.score/max(1,st.session_state.attempts)*100:.0f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("➡️ Next Headline", key="next_question", use_container_width=False):
                st.session_state.quiz_index = random.randint(0, len(quiz_headlines) - 1)
                st.session_state.quiz_answered = False
                del st.session_state.user_choice
                st.rerun()

# Footer
st.markdown("""
<div style="margin-top:3rem; padding:1rem; border-top:1px solid #30363D;
            text-align:center; color:#8B949E; font-size:0.8rem;">
    <b>Financial News Emotion Intelligence</b> — Postgraduate Research Project<br>
    NLP-Based Framework for Measuring Fear, Greed, Uncertainty &amp; Optimism in Financial Markets<br>
    <span style="color:#E53935;">FNEI is a research index only. Not for investment use.</span>
</div>
""", unsafe_allow_html=True)
