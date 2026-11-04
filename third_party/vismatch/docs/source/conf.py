import os
import sys
import subprocess
import importlib
import inspect

# sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

project = "vismatch"
copyright = "2024, Alex Stoken and Gabriele Berton"
author = "Alex Stoken and Gabriele Berton"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.linkcode",  # [source] buttons linking to GitHub
    "sphinx.ext.viewcode",  # fallback inline source viewer
    "myst_parser",  # Markdown support
]

autodoc_mock_imports = [
    "src",
    "model",
    "tools",
    "utils.featurebooster",
    "tensorflow",
    "larq",
    "torch_geometric",
    "utils",
    "third_party",
    "models",
]

# -- autodoc config ----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True  # auto-generate stub .rst files

# -- napoleon config ---------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- HTML theme --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]


# -- linkcode ----------------------------------------------------------------
def _get_git_revision():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).strip().decode()
        return rev if rev else "main"
    except Exception:
        return "main"


_commit = _get_git_revision()
_GITHUB_ROOT = f"https://github.com/gmberton/vismatch/blob/{_commit}"


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    try:
        mod = importlib.import_module(info["module"])
        obj = mod
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        src_file = inspect.getfile(obj)
        lines, start = inspect.getsourcelines(obj)
        # Path relative to the directory containing the package, for both editable and regular
        # installs. Searching for the first "/vismatch/" breaks when a parent directory is also
        # named vismatch (e.g. the repo checkout itself, or Read the Docs' user_builds/vismatch/).
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(importlib.import_module("vismatch").__file__)))
        rel = os.path.relpath(os.path.abspath(src_file), pkg_parent).replace(os.sep, "/")
        if not rel.startswith("vismatch/"):
            return None
        end = start + len(lines) - 1
        return f"{_GITHUB_ROOT}/{rel}#L{start}-L{end}"
    except Exception:
        return None
