# Deployment Guide: Hugging Face Spaces 🐳

This guide will walk you through deploying the **Clinical Trial Inspector Agent** to **Hugging Face Spaces** using Docker.

## Prerequisites

1.  **Hugging Face Account**: [Sign up here](https://huggingface.co/join).
2.  **Git LFS (Large File Storage)**: Required to upload the database (~700MB).
    *   **Mac**: `brew install git-lfs`
    *   **Windows**: Download from [git-lfs.com](https://git-lfs.com/)
    *   **Linux**: `sudo apt-get install git-lfs`

## Step 1.5: Authentication (Crucial!) 🔑

Hugging Face requires an **Access Token** for Git operations (passwords don't work).

1.  Go to **[Settings > Access Tokens](https://huggingface.co/settings/tokens)**.
2.  Click **Create new token**.
3.  **Type**: Select **Write** (important!).
4.  Copy the token (starts with `hf_...`).
5.  **Usage**: When `git push` asks for a password, **paste this token**.

## Step 2: Create a New Space

1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Space Name**: e.g., `clinical-trial-agent`.
3.  **License**: `MIT` (or your choice).
4.  **SDK**: Select **Docker**.
5.  **Visibility**: Public or Private.
6.  Click **Create Space**.

## Step 2: Prepare Your Local Repo

You need to initialize Git LFS to track the large LanceDB files.

```bash
# Initialize LFS
git lfs install

# Track the LanceDB files
git lfs track "ct_gov_lancedb/**/*"
git add .gitattributes
```

## Step 3: Push to Hugging Face

You can either push your existing repo or clone the Space and copy files. Pushing existing is easier:

```bash
# Add the Space as a remote (replace YOUR_USERNAME and SPACE_NAME)
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME

# Push the main branch
git push space main
# OR if you are on a feature branch:
git push space feature/deploy_app:main
```

> **Note**: The first push will take time as it uploads the 700MB database.

## Step 4: Configure Secrets (Optional but Recommended)

To run in **Admin Mode** (no user prompt for API key):

1.  Go to your Space's **Settings** tab.
2.  Scroll to **Variables and secrets**.
3.  Click **New secret**.
4.  **Name**: `GOOGLE_API_KEY`
5.  **Value**: Your Google API Key (starts with `AIza...`).

## Step 5: Verify Deployment

1.  Go to the **App** tab in your Space.
2.  You should see "Building..." in the logs.
3.  Once built, the app will launch! 🚀

---

## Troubleshooting

*   **"LFS upload failed"**: Ensure you ran `git lfs install` and `git lfs track`.
*   **"Runtime Error"**: Check the **Logs** tab. If it says "API Key Missing", ensure you set the Secret or enter it in the UI.
