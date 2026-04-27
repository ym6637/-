# Streamlit Deployment Guide

## 1. Prepare the repository

Keep the following files in the repo root:

- `dashboard.py`
- `requirements.txt`
- `.gitattributes`
- `.gitignore`

Do not commit real secrets. Local development secrets should live in:

- `.streamlit/secrets.toml`

Example:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

## 2. Track large assets with Git LFS

This project includes model and image assets that are better managed with Git LFS.

Recommended commands:

```powershell
git lfs install
git lfs track "*.pt"
git lfs track "*.h5"
git lfs track "*.png"
git lfs track "*.avif"
git add .gitattributes
git add best.pt hybrid_model.h5 image.png converted_avif_lossless
git commit -m "Configure Git LFS for deployment assets"
git push
```

## 3. Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the entrypoint file to `dashboard.py`.
5. Open `Advanced settings`.
6. Select Python `3.11`.
7. Add the app secret:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

8. Deploy.

## 4. Notes for this project

- `dashboard.py` uses `openai_service.py`, which now reads `OPENAI_API_KEY` from `st.secrets` first and environment variables second.
- `pages/chromate.py` can keep importing `openai_service2.py`, because that file now delegates to the same shared implementation.
- If Community Cloud struggles with `tensorflow` or model memory usage, move deployment to a container-based platform such as Render, Railway, or a VM.
