"""
Stress Detection App — Streamlit UI (Redesigned)
Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import stress_model_pipeline as smp


# ════════════════════════════════════════════════════════════════════
# Suggestions Engine
# ════════════════════════════════════════════════════════════════════
# Each suggestion is (icon, title, tip) — icon is kept separate to avoid emoji-split bugs
SUGGESTIONS = {
    "sleep":      ("😴", "Improve Sleep",    "Try sleeping at the same time every day."),
    "exhausted":  ("🛌", "Rest",             "Take proper rest and avoid screens before sleep."),
    "overwhelmed":("🧘", "Calm Down",        "Try breathing exercises for 5 minutes."),
    "anxious":    ("🧘", "Relax",            "Practice meditation or grounding techniques."),
    "hopeless":   ("💬", "Talk to Someone",  "Consult a psychologist or a trusted person."),
    "stress":     ("🧘", "Try Yoga",         "Yoga can help reduce cortisol levels."),
    "deadline":   ("📋", "Plan It Out",      "Break tasks into smaller, manageable steps."),
}

DEFAULT_HIGH = [
    ("💬", "Consult a Psychologist", "Talking to a professional can really help."),
    ("🧘", "Try Yoga",               "Yoga helps calm your mind and release tension."),
    ("🌬", "Deep Breathing",         "Inhale 4s, hold 4s, exhale 6s. Repeat 4 times."),
]

DEFAULT_LOW = [
    ("🌟", "Keep Going",    "You're doing well — keep it up!"),
    ("🎵", "Listen to Music", "Put on something you enjoy and unwind."),
    ("🚶", "Stay Active",   "A short walk can boost your mood further."),
]

HELPLINE = "iCall India: 9152987821"


def get_suggestions(label, top_words):
    suggestions = []
    stress_words = [w['word'] for w in top_words if w['direction'] == 'increases stress']
    for word in stress_words:
        for key in SUGGESTIONS:
            if key in word:
                suggestions.append(SUGGESTIONS[key])
    if not suggestions:
        suggestions = DEFAULT_HIGH if label == "HIGH STRESS" else DEFAULT_LOW
    return suggestions[:3]


# ════════════════════════════════════════════════════════════════════
# Load Model
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline():
    if not os.path.exists("stress_model.pkl"):
        st.error("❌ Model not found. Please train and save the model first.")
        st.stop()
    return smp.load_model("stress_model.pkl")

model = load_pipeline()


# ════════════════════════════════════════════════════════════════════
# Page Config
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MindScan — Stress Detector",
    page_icon="🧠",
    layout="centered",
)


# ════════════════════════════════════════════════════════════════════
# Global CSS — Dark Theme
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:           #0a0a0f;
    --surface:      #11111a;
    --surface2:     #181824;
    --border:       #2a2a3d;
    --accent-hi:    #ff4f6d;
    --accent-lo:    #3ecf8e;
    --accent-blue:  #6c8dfa;
    --accent-amber: #f5a623;
    --text-primary: #f0f0f8;
    --text-muted:   #7b7b9a;
    --text-dim:     #44445a;
    --radius:       14px;
    --radius-sm:    8px;
}

/* ── Force black background everywhere ── */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="block-container"],
section[data-testid="stSidebar"] {
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Remove default padding ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 760px !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Headings ── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

/* ══════════════════════════════════════════
   HEADER BANNER
══════════════════════════════════════════ */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-header .badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
    border-radius: 100px;
    padding: 3px 12px;
    margin-bottom: 1rem;
}
.app-header h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.4rem !important;
    background: linear-gradient(135deg, #f0f0f8 30%, var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-header p {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin: 0;
}

/* ══════════════════════════════════════════
   INPUT CARD
══════════════════════════════════════════ */
.input-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

/* Textarea override */
.stTextArea textarea {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    line-height: 1.6 !important;
    resize: none !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(108,141,250,0.12) !important;
}
.stTextArea textarea::placeholder { color: var(--text-dim) !important; }
.stTextArea label { display: none !important; }

/* ── Analyze button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent-blue), #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease, transform 0.1s ease !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ══════════════════════════════════════════
   RESULT CARDS
══════════════════════════════════════════ */
.result-card {
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 0;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    min-height: 110px;
}
.result-card.high {
    background: rgba(255,79,109,0.08);
    border: 1px solid rgba(255,79,109,0.3);
}
.result-card.low {
    background: rgba(62,207,142,0.08);
    border: 1px solid rgba(62,207,142,0.3);
}
.result-card .verdict {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.1;
}
.result-card.high  .verdict { color: var(--accent-hi); }
.result-card.low   .verdict { color: var(--accent-lo); }
.result-card .dot { font-size: 1.4rem; flex-shrink: 0; }

/* Confidence strip */
.confidence-strip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 0;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.confidence-strip .conf-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
.confidence-strip .conf-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.8rem;
}
.conf-bar-bg {
    background: var(--surface2);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}
.conf-bar-fill.high { background: linear-gradient(90deg, #ff4f6d, #ff8fa0); }
.conf-bar-fill.low  { background: linear-gradient(90deg, #3ecf8e, #7af0b8); }

/* ══════════════════════════════════════════
   SECTION BLOCKS
══════════════════════════════════════════ */
.section-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 0;
}
.section-block .sec-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.9rem;
}
.section-block p { color: var(--text-primary); font-size: 0.95rem; margin: 0; line-height: 1.7; }

/* Top words */
.word-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--surface2);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 4px 4px 4px 0;
    border: 1px solid var(--border);
    color: var(--text-primary);
}
.word-pill.stress  { border-color: rgba(255,79,109,0.4);  color: #ff7a90; }
.word-pill.calm    { border-color: rgba(62,207,142,0.4);  color: #3ecf8e; }

/* Suggestions */
.suggestion-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
}
.suggestion-row:last-child { border-bottom: none; padding-bottom: 0; }
.suggestion-row:first-of-type { padding-top: 0; }
.suggestion-row .sug-icon { font-size: 1.35rem; flex-shrink: 0; margin-top: 1px; line-height: 1; }
.suggestion-row .sug-body {}
.suggestion-row .sug-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--text-primary);
    margin-bottom: 3px;
}
.suggestion-row .sug-tip {
    font-size: 0.83rem;
    color: var(--text-muted);
    line-height: 1.5;
}

/* Helpline banner */
.helpline-banner {
    background: rgba(255,79,109,0.06);
    border: 1px solid rgba(255,79,109,0.25);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
}
.helpline-banner .hl-icon { font-size: 1.3rem; flex-shrink: 0; }
.helpline-banner .hl-text { font-size: 0.88rem; color: #ff9aaa; }
.helpline-banner .hl-num  {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--accent-hi);
}

/* Disclaimer */
.disclaimer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-dim);
    padding-top: 0.5rem;
}

/* Override Streamlit info/error boxes */
.stAlert { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="badge">AI · Mental Health</div>
    <h1>MindScan</h1>
    <p>Explainable stress detection — understand what your words reveal</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# INPUT
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="input-label">How are you feeling?</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="",
    placeholder="e.g. I feel overwhelmed with work and can't sleep at all lately...",
    height=130,
    key="user_input",
)

analyze_clicked = st.button("Analyze →", use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════════
if analyze_clicked:
    if not user_input.strip():
        st.markdown("""
        <div class="section-block" style="border-color:rgba(245,166,35,0.3); background:rgba(245,166,35,0.05);">
            <p style="color:#f5a623;">⚠️ Please enter some text before analyzing.</p>
        </div>""", unsafe_allow_html=True)
    else:
        result      = smp.explain_prediction(model, user_input)
        label       = result['label']
        confidence  = result['confidence']
        top_words   = result['top_words']
        explanation = result['explanation']

        is_high    = label == "HIGH STRESS"
        card_cls   = "high" if is_high else "low"
        dot        = "🔴" if is_high else "🟢"
        label_disp = label

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

        # ── Verdict + Confidence side by side ──────────────────────────
        col_verdict, col_conf = st.columns([1, 1], gap="medium")

        with col_verdict:
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <span class="dot">{dot}</span>
                <div>
                    <div style="font-size:0.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--text-muted);margin-bottom:6px;">Result</div>
                    <div class="verdict">{label_disp}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_conf:
            st.markdown(f"""
            <div class="confidence-strip">
                <div class="conf-label">Confidence</div>
                <div class="conf-value">{confidence}%</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill {card_cls}" style="width:{confidence}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # ── Why this result ─────────────────────────────────────────────
        st.markdown(f"""
        <div class="section-block">
            <div class="sec-title">🔍 Why this result?</div>
            <p>{explanation}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        # ── Key influencing words ───────────────────────────────────────
        if top_words:
            pills_html = ""
            for w in top_words:
                cls  = "stress" if w['direction'] == "increases stress" else "calm"
                icon = "↑" if cls == "stress" else "↓"
                pills_html += f'<span class="word-pill {cls}">{icon} {w["word"]}</span>'

            st.markdown(f"""
            <div class="section-block">
                <div class="sec-title">💡 Key Influencing Words</div>
                <div style="margin-top:0.4rem">{pills_html}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        # ── Suggestions ─────────────────────────────────────────────────
        # Build each row individually and join — avoids Streamlit sanitizing
        # nested HTML when emojis and tags are mixed in one big f-string
        suggestions = get_suggestions(label, top_words)
        row_parts = []
        for icon, name, tip in suggestions:
            row_parts.append(
                f'<div class="suggestion-row">'
                f'<div class="sug-icon">{icon}</div>'
                f'<div class="sug-body">'
                f'<div class="sug-title">{name}</div>'
                f'<div class="sug-tip">{tip}</div>'
                f'</div></div>'
            )
        suggestions_html = "\n".join(row_parts)

        st.markdown(
            f'<div class="section-block">'
            f'<div class="sec-title">&#x2705; What you can do</div>'
            f'{suggestions_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        # ── Helpline (only for high stress) ────────────────────────────
        if is_high:
            st.markdown(
                f'<div class="helpline-banner">'
                f'<div class="hl-icon">&#x260E;</div>'
                f'<div>'
                f'<div class="hl-text">If you\'re feeling overwhelmed, please reach out</div>'
                f'<div class="hl-num">iCall India: {HELPLINE.split(": ")[1]}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        # ── Disclaimer ──────────────────────────────────────────────────
        st.markdown(
            '<div class="disclaimer">&#x26A0;&#xFE0F; This is not a medical diagnosis. '
            'Please consult a professional if needed.</div>',
            unsafe_allow_html=True
        )