---
title: Real or AI Face Game
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: false
---

# Real or AI? Human vs. Model

A small game from CS6180 HW1 (Foundations of Generative AI). You see faces from the course's fixed
test set and decide whether each one is a real photo or AI-generated. A fine-tuned MobileNetV3Small
(77.0% test accuracy) plays the same 10 rounds, and the game shows who did better at the end.

| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `best_model.keras` | Trained model (MobileNetV3Small, fine-tuned) |
| `model_config.json` | Model name, input scaling and number of rounds |
| `test_images/real`, `test_images/ai` | The 300 test images (150 per class) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Only used for Hugging Face Spaces (Docker SDK) |

The block at the top of this README is the configuration for a Hugging Face Space. GitHub and
Streamlit Cloud ignore it.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo (it can be the repo root or a subfolder).
2. Go to https://share.streamlit.io, click **Create app**, then choose the repo and branch.
3. Set **Main file path** to `app.py`, or `streamlit_app/app.py` if the folder isn't the repo root.
4. Under **Advanced settings**, set the Python version to **3.12**.
5. Click **Deploy**. The first build takes a few minutes because TensorFlow is large.

## Deploy on Hugging Face Spaces

1. Create a new Space and choose the **Docker** SDK (blank template).
2. Upload the contents of this folder to the Space, so that `app.py`, `Dockerfile` and `README.md` sit at its root.
   - With the web uploader or `huggingface_hub` this works as is.
   - With `git push`, `best_model.keras` is just over 10 MB and Hugging Face rejects files that size unless they are tracked with Git LFS. Run `git lfs install` and then `git lfs track "*.keras"` before committing.
3. The Space builds from the `Dockerfile` and serves the app on port 8501.
