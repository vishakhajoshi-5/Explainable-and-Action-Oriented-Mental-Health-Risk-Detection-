"""
Stress Detection App — Streamlit UI
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
SUGGESTIONS = {
    "sleep": ("😴 Improve Sleep", "Try sleeping at the same time every day."),
    "exhausted": ("🛌 Rest", "Take proper rest and avoid screen before sleep."),
    "overwhelmed": ("🧘 Calm Down", "Try breathing exercises for 5 minutes."),
    "anxious": ("🧘 Relax", "Practice meditation or grounding techniques."),
    "hopeless": ("💬 Talk to Someone", "Consult a psychologist or trusted person."),
    "stress": ("🧘 Yoga", "Yoga can help reduce cortisol levels."),
    "deadline": ("📋 Plan", "Break tasks into smaller steps."),
}

DEFAULT_HIGH = [
    ("💬 Consult a Psychologist", "Talking to a professional can really help."),
    ("🧘 Try Yoga", "Yoga helps calm your mind."),
    ("🌬️ Deep Breathing", "Inhale 4s, hold 4s, exhale 6s."),
]

DEFAULT_LOW = [
    ("🌟 Keep Going", "You’re doing well, keep it up!"),
    ("🎵 Music", "Listen to something you enjoy."),
]

HELPLINE = "☎️ iCall India: 9152987821"


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
    st.write("Loading model...")   # DEBUG

    if not os.path.exists("stress_model.pkl"):
        st.error("❌ Model not found")
        st.stop()

    model = smp.load_model("stress_model.pkl")
    st.write("Model loaded successfully!")  # DEBUG

    return model


model = load_pipeline()


# ════════════════════════════════════════════════════════════════════
# UI Design
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Stress Detector")

st.title("Mental Health Risk Detector")
st.caption("Explainable + Action-Oriented Stress Detection System")

user_input = st.text_area(
    "How are you feeling?",
    placeholder="e.g. I feel overwhelmed with work and can't sleep..."
)

if st.button("Analyze"):

    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        result = smp.explain_prediction(model, user_input)

        label = result['label']
        confidence = result['confidence']
        top_words = result['top_words']
        explanation = result['explanation']

        # ── Result ─────────────────────────────────────
        if label == "HIGH STRESS":
            st.error(f"🔴 {label} ({confidence}%)")
        else:
            st.success(f"🟢 {label} ({confidence}%)")

        # ── Explanation ────────────────────────────────
        st.subheader("🔍 Why this result?")
        st.info(explanation)

        # ── Top Words ─────────────────────────────────
        if top_words:
            st.subheader("Key Influencing Words")
            for w in top_words:
                emoji = "🔴" if w['direction'] == 'increases stress' else "🟢"
                st.write(f"{emoji} {w['word']} → {w['direction']}")

        # ── Suggestions ───────────────────────────────
        st.subheader("What you can do")

        suggestions = get_suggestions(label, top_words)

        for title, tip in suggestions:
            st.markdown(f"**{title}** — {tip}")

        # ── Helpline ─────────────────────────────────
        if label == "HIGH STRESS":
            st.error(f"If you're feeling overwhelmed, please reach out: {HELPLINE}")

        # ── Disclaimer ───────────────────────────────
        st.caption("⚠️ This is not a medical diagnosis. Please consult a professional if needed.")

