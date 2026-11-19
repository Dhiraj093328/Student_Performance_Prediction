import math

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EduPredict | Student Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")


model = load_model()

# ---------------------------------------------------------------
# Styling (dark dashboard look; widget colours come from .streamlit/config.toml)
# ---------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{
 --bg:#0a0f1d; --panel:#111a2e; --panel2:#16213a; --border:#22304d;
 --text:#e8edf7; --muted:#8fa0bf; --indigo:#6366f1; --cyan:#22d3ee;
 --green:#34d399; --amber:#fbbf24; --red:#f87171;
}
html, body, [class*="css"], .stApp { font-family:'Plus Jakarta Sans',sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 70% -10%, #1b2a5e55, transparent), var(--bg); }
header[data-testid="stHeader"] { background:transparent; }
footer { visibility:hidden; }
.block-container { max-width:1180px; padding-top:1rem; padding-bottom:3rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--border); }
section[data-testid="stSidebar"] .stButton > button {
 width:100%; justify-content:flex-start; background:var(--panel2); color:var(--text);
 border:1px solid var(--border); border-radius:12px; padding:.55rem .9rem; font-weight:600;
}
section[data-testid="stSidebar"] .stButton > button:hover { border-color:var(--indigo); color:#fff; }
.brand { display:flex; align-items:center; gap:.7rem; margin-bottom:1.4rem; }
.brand .logo { width:38px; height:38px; border-radius:11px; display:grid; place-items:center;
 background:linear-gradient(135deg,var(--indigo),var(--cyan)); font-size:1.2rem; }
.brand b { font-size:1.25rem; color:#fff; }
.side-title { color:var(--muted); font-size:.82rem; font-weight:600; margin:1.2rem 0 .5rem; }
.side-card { background:var(--panel2); border:1px solid var(--border); border-radius:14px;
 padding:.9rem 1rem; color:var(--muted); font-size:.85rem; line-height:1.5; }
.side-card b { color:var(--text); }

/* Top banner */
.banner { display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;
 background:linear-gradient(110deg,#1e3a8a,#4f46e5 55%,#0ea5e9); border-radius:18px; padding:1.1rem 1.5rem; }
.banner h2 { margin:0; padding:0; color:#fff; font-size:1.35rem; font-weight:800; }
.banner p { margin:.15rem 0 0; color:#d6defa; font-size:.88rem; }
.banner .session { background:#ffffff1f; border:1px solid #ffffff33; border-radius:12px;
 padding:.45rem .9rem; color:#fff; font-size:.85rem; font-weight:600; }

/* Hero */
.hero { padding:2.2rem 0 0 0; }
.pill { display:inline-block; background:#6366f126; color:#a5b4fc; border:1px solid #6366f155;
 border-radius:999px; padding:.3rem .85rem; font-size:.82rem; font-weight:600; }
.hero h1 { color:#fff; font-size:2.9rem; line-height:1.1; font-weight:800; margin:1rem 0 1rem; padding:0; letter-spacing:-.02em; }
.hero h1 span { background:linear-gradient(90deg,var(--indigo),var(--cyan)); -webkit-background-clip:text;
 background-clip:text; color:transparent; }
.hero p.lead { color:var(--muted); font-size:1.02rem; line-height:1.65; max-width:30rem; margin:0 0 1.6rem; }
.feats { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; max-width:34rem; }
.feat { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:.85rem; display:flex; gap:.7rem; align-items:center; }
.feat i { font-style:normal; width:38px; height:38px; flex:none; border-radius:50%; display:grid; place-items:center;
 background:linear-gradient(135deg,#4f46e5,#0ea5e9); }
.feat b { color:#fff; font-size:.9rem; display:block; }
.feat small { color:var(--muted); font-size:.78rem; }

/* Form card */
[data-testid="stForm"] { background:var(--panel); border:1px solid var(--border); border-radius:20px;
 padding:1.6rem 1.6rem 1.2rem; box-shadow:0 20px 60px #00000055; }
.form-head { text-align:center; margin-bottom:.6rem; }
.form-head .ico { width:54px; height:54px; border-radius:50%; margin:0 auto .6rem; display:grid; place-items:center;
 font-size:1.5rem; background:linear-gradient(135deg,var(--indigo),var(--cyan)); }
.form-head h3 { margin:0; padding:0; color:#fff; font-size:1.3rem; font-weight:800; }
.form-head p { margin:.2rem 0 0; color:var(--muted); font-size:.88rem; }
[data-testid="stWidgetLabel"] p { color:var(--text); font-weight:600; font-size:.88rem; }
.stFormSubmitButton > button { width:100%; border:none; border-radius:12px; padding:.8rem 1rem; color:#fff;
 font-weight:700; font-size:1rem; background:linear-gradient(90deg,#4f46e5,#0ea5e9); transition:filter .15s ease; }
.stFormSubmitButton > button:hover { filter:brightness(1.12); color:#fff; }
.stFormSubmitButton > button:focus-visible { outline:3px solid var(--amber); outline-offset:2px; }

/* Empty state */
.empty { margin-top:1.5rem; border:1.5px dashed var(--border); border-radius:18px; padding:2rem; text-align:center; color:var(--muted); }
.empty b { color:var(--text); display:block; font-size:1.05rem; margin-bottom:.3rem; }

/* Dashboard */
.sec { color:#fff; font-size:1.25rem; font-weight:800; margin:2.2rem 0 1rem; }
.grid4 { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:1rem; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1rem; margin-top:1rem; }
.kpi { border-radius:16px; padding:1.1rem 1.2rem; color:#fff; display:flex; gap:.9rem; align-items:center; }
.kpi .ic { width:46px; height:46px; flex:none; border-radius:13px; background:#ffffff26; display:grid; place-items:center; font-size:1.4rem; }
.kpi small { display:block; color:#ffffffcc; font-size:.8rem; font-weight:500; }
.kpi strong { font-size:1.7rem; font-weight:800; line-height:1.15; }
.kpi em { font-style:normal; font-size:.78rem; color:#ffffffd9; display:block; }
.k1 { background:linear-gradient(135deg,#4f46e5,#2563eb); }
.k2 { background:linear-gradient(135deg,#0f766e,#10b981); }
.k3 { background:linear-gradient(135deg,#7c3aed,#4f46e5); }
.k4 { background:linear-gradient(135deg,#0284c7,#14b8a6); }
.mini-bar { height:6px; background:#ffffff33; border-radius:99px; margin-top:.4rem; overflow:hidden; }
.mini-bar div { height:100%; background:#fff; border-radius:99px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:18px; padding:1.2rem 1.3rem; }
.card h4 { margin:0 0 .2rem; padding:0; color:#fff; font-size:1rem; font-weight:700; }
.card p.sub { margin:0 0 .6rem; color:var(--muted); font-size:.82rem; }
.card svg { width:100%; height:auto; display:block; }
.gauge-wrap { max-width:300px; margin:0 auto; }
.insights { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1rem; }
.ins { background:var(--panel2); border:1px solid var(--border); border-left:4px solid var(--green);
 border-radius:14px; padding:1rem 1.1rem; }
.ins b { color:#fff; display:block; font-size:.95rem; }
.ins span { color:var(--green); font-size:1.5rem; font-weight:800; }
.ins small { color:var(--muted); display:block; margin-top:.15rem; }
.note { color:var(--muted); font-size:.82rem; text-align:center; margin-top:1.8rem; }
</style>
"""


def html(markup: str):
    """Render HTML; whitespace is collapsed so Markdown never treats it as code."""
    st.markdown(" ".join(markup.split()), unsafe_allow_html=True)


html(CSS)

# ---------------------------------------------------------------
# State + presets
# ---------------------------------------------------------------
DEFAULTS = dict(hours=5, prev=70, extra="Yes", sleep=7, papers=5)
PRESETS = {
    "🏆 Top performer": dict(hours=8, prev=92, extra="Yes", sleep=8, papers=8),
    "⚖️ Balanced": DEFAULTS,
    "⚠️ Needs support": dict(hours=2, prev=48, extra="No", sleep=5, papers=1),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    st.session_state.update(PRESETS[name])


with st.sidebar:
    html('<div class="brand"><div class="logo">🎓</div><b>EduPredict</b></div>')
    html('<div class="side-title">Quick profiles</div>')
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), key=f"preset_{name}")
    html(
        """<div class="side-title">About the model</div>
        <div class="side-card"><b>5 inputs</b> → <b>1 performance index</b> (0 to 100).<br>
        Charts use the same trained model to test how changing one habit moves the score.</div>"""
    )

# ---------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------
def predict(rows):
    preds = model.predict(pd.DataFrame(rows))
    return [max(0.0, min(100.0, float(p))) for p in preds]


def tier(score):
    if score >= 80:
        return "Excellent", "Outstanding outlook", "#34d399"
    if score >= 60:
        return "Good", "On a solid track", "#22d3ee"
    if score >= 40:
        return "Average", "Room to improve", "#fbbf24"
    return "Needs support", "Needs attention", "#f87171"


def gauge_svg(score, color):
    arc = math.pi * 80
    fill = arc * score / 100
    return f"""<svg viewBox="0 0 200 125" role="img" aria-label="Score {score:.1f} out of 100">
    <path d="M20 105 A80 80 0 0 1 180 105" fill="none" stroke="#22304d" stroke-width="16" stroke-linecap="round"/>
    <path d="M20 105 A80 80 0 0 1 180 105" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"
     stroke-dasharray="{fill:.1f} {arc:.1f}"/>
    <text x="100" y="92" text-anchor="middle" font-size="34" font-weight="800" fill="#fff">{score:.1f}</text>
    <text x="100" y="112" text-anchor="middle" font-size="10" fill="#8fa0bf">out of 100</text></svg>"""


def radar_svg(items):
    cx, cy, r, n = 110, 110, 74, len(items)

    def pt(i, k):
        a = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * k * math.cos(a), cy + r * k * math.sin(a)

    grid = ""
    for k in (0.25, 0.5, 0.75, 1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, k) for i in range(n)))
        grid += f'<polygon points="{pts}" fill="none" stroke="#22304d"/>'
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{pt(i, 1)[0]:.1f}" y2="{pt(i, 1)[1]:.1f}" stroke="#22304d"/>' for i in range(n)
    )
    vals = [pt(i, max(v, 0.04)) for i, (_, v) in enumerate(items)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in vals)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#22d3ee"/>' for x, y in vals)
    labels = ""
    for i, (name, v) in enumerate(items):
        x, y = pt(i, 1.28)
        anchor = "middle" if abs(x - cx) < 6 else ("start" if x > cx else "end")
        labels += (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="9.5" fill="#8fa0bf">{name} '
            f'<tspan fill="#fff" font-weight="700">{round(v * 100)}</tspan></text>'
        )
    return f"""<svg viewBox="-20 -5 260 230" role="img" aria-label="Habit profile radar chart">{grid}{axes}
    <polygon points="{poly}" fill="#6366f155" stroke="#818cf8" stroke-width="2"/>{dots}{labels}</svg>"""


def line_svg(xs, ys, cur, uid):
    W, H, L, B, T, R = 400, 215, 40, 34, 14, 14
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.25, 1.0)
    lo, hi = lo - pad, hi + pad
    sx = lambda x: L + (x - xs[0]) / (xs[-1] - xs[0]) * (W - L - R)
    sy = lambda y: T + (hi - y) / (hi - lo) * (H - T - B)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    area = f"{sx(xs[0]):.1f},{H - B} {pts} {sx(xs[-1]):.1f},{H - B}"
    yt = "".join(
        f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#22304d" stroke-dasharray="3 4"/>'
        f'<text x="{L - 8}" y="{sy(v) + 3:.1f}" text-anchor="end" font-size="10" fill="#8fa0bf">{v:.0f}</text>'
        for v in (lo, (lo + hi) / 2, hi)
    )
    xt = "".join(
        f'<text x="{sx(x):.1f}" y="{H - B + 18}" text-anchor="middle" font-size="10" fill="#8fa0bf">{x}</text>' for x in xs
    )
    cy_ = ys[xs.index(cur)]
    return f"""<svg viewBox="0 0 {W} {H}" role="img" aria-label="What-if line chart">
    <defs><linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#6366f1" stop-opacity=".45"/><stop offset="1" stop-color="#6366f1" stop-opacity="0"/>
    </linearGradient></defs>{yt}{xt}
    <polygon points="{area}" fill="url(#g{uid})"/>
    <polyline points="{pts}" fill="none" stroke="#818cf8" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    <line x1="{sx(cur):.1f}" x2="{sx(cur):.1f}" y1="{T}" y2="{H - B}" stroke="#22d3ee" stroke-dasharray="4 4"/>
    <circle cx="{sx(cur):.1f}" cy="{sy(cy_):.1f}" r="6" fill="#22d3ee" stroke="#0a0f1d" stroke-width="2"/></svg>"""


# ---------------------------------------------------------------
# Banner
# ---------------------------------------------------------------
html(
    """<div class="banner"><div><h2>Student Performance Predictor</h2>
    <p>Insights for better learning. Data-driven decisions.</p></div>
    <div class="session">Academic session 2026 – 2027</div></div>"""
)

# ---------------------------------------------------------------
# Hero + form
# ---------------------------------------------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    html(
        """<div class="hero"><span class="pill">⚡ AI-powered predictions</span>
        <h1>Get a student's<br><span>performance forecast</span></h1>
        <p class="lead">Enter study habits and past results. Our trained model estimates the performance index
        and shows which habits would move it the most.</p>
        <div class="feats">
        <div class="feat"><i>⚡</i><div><b>Instant results</b><small>Predictions in seconds</small></div></div>
        <div class="feat"><i>🔒</i><div><b>Private by design</b><small>Inputs stay in your session</small></div></div>
        <div class="feat"><i>📊</i><div><b>What-if analysis</b><small>See how habits shift the score</small></div></div>
        <div class="feat"><i>🎯</i><div><b>Clear next steps</b><small>Tips ranked by impact</small></div></div>
        </div></div>"""
    )

with right:
    with st.form("prediction_form"):
        html(
            """<div class="form-head"><div class="ico">🎓</div><h3>Student profile</h3>
            <p>Enter the details to get a prediction</p></div>"""
        )
        c1, c2 = st.columns(2)
        with c1:
            hours = st.slider("📚 Hours studied per day", 1, 9, key="hours")
            papers = st.slider("🗂️ Sample papers practiced", 0, 9, key="papers")
        with c2:
            sleep_hours = st.slider("😴 Sleep hours per night", 4, 9, key="sleep")
            previous_scores = st.slider("📝 Previous scores", 40, 99, key="prev")
        extracurricular = st.radio("⚽ Extracurricular activities", ["Yes", "No"], horizontal=True, key="extra")
        submit = st.form_submit_button("Predict performance")

# ---------------------------------------------------------------
# Run the model
# ---------------------------------------------------------------
if submit:
    profile = {
        "Hours Studied": hours,
        "Previous Scores": previous_scores,
        "Extracurricular Activities": extracurricular,
        "Sleep Hours": sleep_hours,
        "Sample Question Papers Practiced": papers,
    }
    with st.spinner("Calculating..."):
        score = predict([profile])[0]
        xs_h, xs_p = list(range(1, 10)), list(range(0, 10))
        ys_h = predict([{**profile, "Hours Studied": h} for h in xs_h])
        ys_p = predict([{**profile, "Sample Question Papers Practiced": p} for p in xs_p])

        levers = []
        if hours < 9:
            levers.append(("Study 1 more hour a day", {"Hours Studied": hours + 1}))
        if papers < 9:
            n = min(9, papers + 2)
            levers.append((f"Practice {n - papers} more sample papers", {"Sample Question Papers Practiced": n}))
        if sleep_hours < 9:
            levers.append(("Sleep 1 more hour", {"Sleep Hours": sleep_hours + 1}))
        if extracurricular == "No":
            levers.append(("Join an extracurricular activity", {"Extracurricular Activities": "Yes"}))
        gains = []
        if levers:
            for (label, change), p in zip(levers, predict([{**profile, **c} for _, c in levers])):
                gains.append((label, p - score))
        gains = sorted([g for g in gains if g[1] > 0.05], key=lambda g: -g[1])[:3]

    st.session_state["result"] = dict(
        profile=profile, score=score, xs_h=xs_h, ys_h=ys_h, xs_p=xs_p, ys_p=ys_p, gains=gains
    )

# ---------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------
res = st.session_state.get("result")

if not res:
    html(
        """<div class="empty"><b>Your results will appear here</b>
        Fill in the student profile and select Predict performance, or start from a quick profile in the sidebar.</div>"""
    )
else:
    p, score = res["profile"], res["score"]
    label, blurb, color = tier(score)
    h, sl, pa = p["Hours Studied"], p["Sleep Hours"], p["Sample Question Papers Practiced"]
    prev, extra = p["Previous Scores"], p["Extracurricular Activities"]

    radar_items = [
        ("Study", h / 9),
        ("Sleep", (sl - 4) / 5),
        ("Practice", pa / 9),
        ("Previous", (prev - 40) / 59),
        ("Activities", 1.0 if extra == "Yes" else 0.4),
    ]
    habit = round(sum(v for n, v in radar_items if n != "Previous") / 4 * 100)
    delta = score - prev
    arrow = "▲" if delta >= 0 else "▼"

    html('<div class="sec">Prediction overview</div>')
    html(
        f"""<div class="grid4">
        <div class="kpi k1"><div class="ic">🎯</div><div><small>Predicted index</small><strong>{score:.1f}</strong></div></div>
        <div class="kpi k2"><div class="ic">🏅</div><div><small>Performance level</small><strong>{label}</strong><em>{blurb}</em></div></div>
        <div class="kpi k3"><div class="ic">📝</div><div><small>Previous score</small><strong>{prev}</strong>
        <em>{arrow} {abs(delta):.1f} points vs prediction</em></div></div>
        <div class="kpi k4"><div class="ic">💪</div><div style="flex:1"><small>Habit score</small><strong>{habit}%</strong>
        <div class="mini-bar"><div style="width:{habit}%"></div></div></div></div></div>"""
    )

    html(
        f"""<div class="grid2">
        <div class="card"><h4>Performance gauge</h4><p class="sub">Predicted index on a 0 to 100 scale</p>
        <div class="gauge-wrap">{gauge_svg(score, color)}</div></div>
        <div class="card"><h4>Habit profile</h4><p class="sub">Each axis is scaled 0 to 100 within the allowed range</p>
        {radar_svg(radar_items)}</div></div>"""
    )

    html('<div class="sec">What-if analysis</div>')
    html(
        f"""<div class="grid2" style="margin-top:0">
        <div class="card"><h4>Hours studied vs. predicted index</h4><p class="sub">Other inputs held constant. Marker shows the current value.</p>
        {line_svg(res["xs_h"], res["ys_h"], h, "h")}</div>
        <div class="card"><h4>Sample papers vs. predicted index</h4><p class="sub">Other inputs held constant. Marker shows the current value.</p>
        {line_svg(res["xs_p"], res["ys_p"], pa, "p")}</div></div>"""
    )

    html('<div class="sec">Biggest opportunities</div>')
    if res["gains"]:
        cards = "".join(
            f'<div class="ins"><b>{lab}</b><span>+{g:.1f}</span><small>estimated points on the index</small></div>'
            for lab, g in res["gains"]
        )
    else:
        cards = (
            '<div class="ins"><b>These habits are already near the model\'s best</b>'
            "<small>No single change is predicted to raise the index.</small></div>"
        )
    html(f'<div class="insights">{cards}</div>')
    html('<div class="note">Estimates come from your trained machine learning model and are not guaranteed results.</div>')