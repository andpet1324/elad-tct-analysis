"""
Progress display: a top bar over the steps and a second bar for the work in the current step.

    with Steps(["Load data", "Analyze", "Plot"]) as steps:
        steps.start("Load data", total=100)   # total=None gives a pulsing bar
        steps.advance()                       # or steps.advance(10)
        steps.start("Analyze", total=None)    # finishes the previous step

print() works while it is running, rich draws the bars below the text.
"""

from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn)


class Steps:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description:<22}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.top = self.progress.add_task("TCT analysis", total=len(names))
        self.sub: int | None = None

    def __enter__(self) -> "Steps":
        self.progress.start()
        return self

    def __exit__(self, exc_type, *exc) -> None:
        if exc_type is None:
            self._finish_sub()
        self.progress.update(self.top, description="TCT analysis, " + ("done" if exc_type is None else "stopped"))
        self.progress.stop()

    def _finish_sub(self) -> None:
        if self.sub is not None:
            self.progress.remove_task(self.sub)
            self.progress.advance(self.top)
            self.sub = None

    def start(self, name: str, total: int | None = None) -> None:
        self._finish_sub()
        step = self.names.index(name) + 1
        self.progress.update(self.top, description=f"Step {step}/{len(self.names)}")
        self.sub = self.progress.add_task(name, total=total)

    def set_total(self, total: int | None) -> None:
        """Set the total of the current step once it is known"""
        if self.sub is not None:
            self.progress.update(self.sub, total=total)

    def advance(self, n: int = 1) -> None:
        if self.sub is not None:
            self.progress.advance(self.sub, n)
