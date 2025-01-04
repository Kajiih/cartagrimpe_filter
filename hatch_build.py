"""Hatch build hook."""

import importlib.util
import sys
from pathlib import Path
from typing import Any, override

from hatchling.metadata.plugin.interface import MetadataHookInterface


class AboutMetadataHookOld(MetadataHookInterface):
    """Hatchling metadata hook that loads dynamic metadata for authors, and URLs from the `__about__.py` file within the project."""

    @override
    def update(self, metadata: dict[str, Any]) -> None:
        """
        Update the metadata dictionary with values from the `__about__.py` file.

        Args:
            metadata: The dictionary containing the project metadata.
        """
        # Dynamically load __about__.py
        about_path = Path(self.root) / "src/cartagrimpe_filter/__about__.py"
        spec = importlib.util.spec_from_file_location("__about__", about_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load metadata from {about_path}")  # noqa: TRY003

        about = importlib.util.module_from_spec(spec)
        sys.modules["__about__"] = about
        spec.loader.exec_module(about)

        # Map __about__ attributes to metadata fields
        authors: list[str] = about.__authors__
        emails: list[str] = about.__author_emails__
        metadata["authors"] = [
            {"name": name, "email": email} for name, email in zip(authors, emails, strict=True)
        ]
        metadata["urls"] = {
            "Repository": about.__repo_url__,
            "Issues": about.__issues_url__,
        }
