"""Upload one folder to a private Hugging Face repository, with its card as README.md.

Usage: scripts/publish.py REPO_ID FOLDER CARD model|dataset [PATTERN...]

The repository type is named before any file pattern: the first publishing run took a
pattern for the type and refused both GGUF repositories (2026-09-27).
The token comes from HF_TOKEN in the environment and is never printed.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, folder, card = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
repo_type = sys.argv[4] if len(sys.argv) > 4 else "model"
patterns = sys.argv[5:] or None

api = HfApi()
api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=str(folder),
    allow_patterns=patterns,
    ignore_patterns=["README.md"],
    commit_message=f"Upload from {folder.name}",
)
api.upload_file(
    repo_id=repo_id,
    repo_type=repo_type,
    path_or_fileobj=str(card),
    path_in_repo="README.md",
    commit_message="The card, generated from the records",
)
files = api.list_repo_files(repo_id, repo_type=repo_type)
print(f"=== {repo_id}: {len(files)} files, private")
