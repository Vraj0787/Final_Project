from __future__ import annotations

from pathlib import Path

from final_project_analysis import main as run_full_analysis


class EmploymentAnalysis:
    """
    Compatibility wrapper for the team's original starter file.

    This keeps the existing entry point name but redirects execution to the
    finished multi-dataset analysis pipeline used for the final project.
    """

    def __init__(self, project_root: str | None = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent

    def run(self) -> None:
        run_full_analysis()


if __name__ == "__main__":
    EmploymentAnalysis().run()
