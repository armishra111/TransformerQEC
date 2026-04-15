from pathlib import Path
from typing import Annotated

import typer

from transformerqec.config.io import load_run_config

app = typer.Typer(name="transformerqec", help="TransformerQEC decoder library CLI")


@app.callback()
def main() -> None:
    return None


@app.command("generate")
def generate(
    config: Annotated[Path, typer.Option("--config", help="Path to a run config YAML.")],
) -> None:
    load_run_config(config)
    typer.echo(f"Loaded data generation config: {config}")


@app.command("train")
def train(
    config: Annotated[Path, typer.Option("--config", help="Path to a run config YAML.")],
) -> None:
    load_run_config(config)
    typer.echo(f"Loaded training config: {config}")


@app.command("eval")
def evaluate(
    config: Annotated[Path, typer.Option("--config", help="Path to a run config YAML.")],
) -> None:
    load_run_config(config)
    typer.echo(f"Loaded evaluation config: {config}")


@app.command("reproduce-baseline")
def reproduce_baseline(
    config: Annotated[Path, typer.Option("--config", help="Path to a run config YAML.")],
) -> None:
    loaded = load_run_config(config)
    typer.echo(f"Baseline reproduction ready for {loaded.experiment_name}")


@app.command("benchmark")
def benchmark(
    config: Annotated[Path, typer.Option("--config", help="Path to a run config YAML.")],
) -> None:
    load_run_config(config)
    typer.echo(f"Benchmark config accepted: {config}")


@app.command("infer")
def infer(
    config: Annotated[Path, typer.Option("--config", help="Path to a run config YAML.")],
) -> None:
    load_run_config(config)
    typer.echo(f"Inference config accepted: {config}")
