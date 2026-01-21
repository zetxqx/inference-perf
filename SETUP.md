# Development Environment Setup

This guide explains how to set up the environment for this project and configure VSCode for full IDE features like **Go to Definition** and **Find References**.

## 1. Prerequisites

This project uses **PDM** as its package manager (detected via `pdm.lock`).

- **Install PDM**:
  ```bash
  curl -sSL https://pdm-project.org/install-pdm.py | python3 -
  ```

## 2. Environment Setup

To ensure VSCode indexes everything correctly, it is best to use a local virtual environment (`.venv`).

1. **Configure PDM to use virtualenvs**:
   ```bash
   pdm config python.use_venv true
   ```

2. **Install dependencies**:
   ```bash
   pdm install
   ```
   This will create a `.venv` directory in your project root containing all libraries.

## 3. VSCode Configuration

For "Go to Definition" and "Find References" to work, VSCode must be linked to the project's interpreter.

1. **Install Extensions**:
   Ensure you have the **Python** and **Pylance** extensions from Microsoft installed.

2. **Select the Interpreter**:
   - Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS).
   - Type `Python: Select Interpreter`.
   - Select the path that points to `./.venv/bin/python`.

3. **Verify Indexing**:
   - Open a file (e.g., `analyze.py`).
   - Wait a few seconds for Pylance to finish "Loading..." in the status bar.
   - Right-click any import or function and select **Go to Definition**.

## 4. Troubleshooting Indexing

If "Go to Definition" is not working:
- **Check `.vscode/settings.json`**: Ensure `python.analysis.extraPaths` is not pointing to old directories. Usually, with a `.venv`, you don't need extra paths.
- **Restart Language Server**: Run `Developer: Reload Window` from the Command Palette.

---

### Pro-Tip: Using `uv` (Fastest Alternative)
If you prefer a faster experience, you can use `uv` to manage the environment:
```bash
# Install uv
curl -sSL https://astral.sh/uv/install.sh | sh

# Create venv and sync
uv venv
uv pip install -e .
```
VSCode will detect the `.venv` created by `uv` automatically.
