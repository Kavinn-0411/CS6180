# ============================================================
# Part 8 - Streamlit game: Human vs. AI Detector  (app.py)
# ============================================================
import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = Path(__file__).parent
IMG_DIR = APP_DIR / "test_images"
LABELS = {1: "Real", 0: "AI-Generated"}

st.set_page_config(page_title="Real or AI?", page_icon="🕵️", layout="centered")


@st.cache_resource
def load_model():
    cfg = json.loads((APP_DIR / "model_config.json").read_text())
    model = tf.keras.models.load_model(APP_DIR / "best_model.keras")
    return model, cfg


@st.cache_data
def load_image_list():
    """(path, label) for every test image; label 1 = Real, 0 = AI (same convention as training)."""
    items = []
    for label, sub in ((1, "real"), (0, "ai")):
        for p in sorted((IMG_DIR / sub).iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((str(p), label))
    return items


def model_predict(path):
    """P(Real) for one image, preprocessed exactly like the notebook (resize to 128x128, scale)."""
    model, cfg = load_model()
    img = Image.open(path).convert("RGB").resize((128, 128))
    x = np.asarray(img, dtype=np.float32) / 255.0 * cfg["input_scale"]
    return float(model.predict(x[None, ...], verbose=0)[0, 0])


model, cfg = load_model()
images = load_image_list()
N_ROUNDS = min(int(cfg.get("n_rounds", 10)), len(images))


# ---- Game state ----
def new_game():
    ss = st.session_state
    ss.deck = random.sample(range(len(load_image_list())), N_ROUNDS)
    ss.round = 0          # index of the current round
    ss.human = 0          # human score
    ss.model = 0          # model score
    ss.result = None      # feedback for the current round once the human has guessed
    ss.log = []
    ss.celebrated = False


def submit_guess(guess):
    ss = st.session_state
    path, label = load_image_list()[ss.deck[ss.round]]
    prob = model_predict(path)
    model_guess = int(prob >= 0.5)
    ss.human += int(guess == label)
    ss.model += int(model_guess == label)
    ss.result = {"label": label, "guess": guess, "prob": prob, "model_guess": model_guess}
    ss.log.append({
        "Round": ss.round + 1,
        "Answer": LABELS[label],
        "You": ("✅ " if guess == label else "❌ ") + LABELS[guess],
        "Model": ("✅ " if model_guess == label else "❌ ") + f"{LABELS[model_guess]} (P(Real)={prob:.2f})",
    })


def next_round():
    st.session_state.round += 1
    st.session_state.result = None


if "deck" not in st.session_state:
    new_game()
ss = st.session_state

# ---- Header ----
st.title("🕵️ Real or AI? Human vs. Model")
st.caption("Made by **Kavinn Premmesh Tamilarasu** · CS6180 Foundations of Generative AI, HW1")
st.write(
    f"Each round shows a face. Decide whether it's a **real photo** or **AI-generated**. "
    f"Our best classifier (**{cfg['model_name']}**, {cfg['test_acc']:.0%} test accuracy) "
    f"plays the same images. After **{N_ROUNDS} rounds**, whoever has more correct answers wins."
)

# ---- Scoreboard ----
rounds_played = len(ss.log)
c1, c2, c3 = st.columns(3)
c1.metric("Round", f"{min(ss.round + 1, N_ROUNDS)} / {N_ROUNDS}")
c2.metric("🧑 You", ss.human, f"{ss.human / rounds_played:.0%} accuracy" if rounds_played else None, delta_color="off")
c3.metric("🤖 Model", ss.model, f"{ss.model / rounds_played:.0%} accuracy" if rounds_played else None, delta_color="off")
st.progress(rounds_played / N_ROUNDS)

# ---- Final summary ----
if ss.round >= N_ROUNDS:
    st.header("🏁 Final results")
    if ss.human > ss.model:
        st.success(f"🎉 You win, {ss.human} to {ss.model}! Humans 1, machines 0.")
        if not ss.celebrated:
            st.balloons()
            ss.celebrated = True
    elif ss.human < ss.model:
        st.error(f"🤖 The model wins, {ss.model} to {ss.human}. Spotting AI faces is hard!")
    else:
        st.info(f"🤝 It's a tie at {ss.human} each!")
    st.dataframe(ss.log, hide_index=True)
    st.button("🔁 Play again", on_click=new_game, type="primary")
    st.stop()

# ---- Current round ----
path, label = images[ss.deck[ss.round]]
st.image(path, width=380)

if ss.result is None:
    b1, b2 = st.columns(2)
    b1.button("📸 Real", on_click=submit_guess, args=(1,), type="primary")
    b2.button("🤖 AI-Generated", on_click=submit_guess, args=(0,), type="primary")
else:
    r = ss.result
    truth = LABELS[r["label"]]
    if r["guess"] == r["label"]:
        st.success(f"✅ Correct! This face is **{truth}**.")
    else:
        st.error(f"❌ Not quite. This face is **{truth}**.")
    model_ok = r["model_guess"] == r["label"]
    confidence = r["prob"] if r["model_guess"] == 1 else 1 - r["prob"]
    model_msg = (f"🤖 The model said **{LABELS[r['model_guess']]}** with {confidence:.0%} confidence "
                 f"(P(Real) = {r['prob']:.2f}), which is {'correct ✅' if model_ok else 'wrong ❌'}.")
    (st.info if model_ok else st.warning)(model_msg)
    last_round = ss.round + 1 >= N_ROUNDS
    st.button("🏁 See final results" if last_round else "➡️ Next image", on_click=next_round, type="primary")
