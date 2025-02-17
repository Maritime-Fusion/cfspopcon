#!.venv/bin/python
# Run this script from the repository directory.
"""CLI for cfspopcon."""

import warnings
from pathlib import Path
import sys

import click
import matplotlib.pyplot as plt
import xarray as xr

from cfspopcon import file_io
from cfspopcon.input_file_handling import read_case
from cfspopcon.plotting import make_plot, read_plot_style
from cfspopcon.unit_handling import UnitStrippedWarning



@click.command()
@click.argument("case", type=click.Path(exists=True))
@click.option("--dict", "-d", "kwargs", type=(str, str), multiple=True, help="Command-line arguments, takes precedence over config.")
@click.option("--show", is_flag=True, help="Display an interactive figure of the result.")
@click.option("--debug", is_flag=True, help="Enable the ipdb exception catcher. (Development helper)", hidden=True)
def run_popcon_cli(case: str, show: bool, debug: bool, kwargs: tuple[tuple[str, str]]) -> None:
    """Run POPCON from the command line.

    This function uses "Click" to develop the command line interface. You can execute it using
    poetry run python cfspopcon/cli.py --help
    """
    cli_args: dict[str, str] = dict(kwargs)

    if debug:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UnitStrippedWarning)
            try:
                # if ipdb is installed we use it to catch exceptions during development
                from ipdb import launch_ipdb_on_exception  # type:ignore[import-untyped]

                with launch_ipdb_on_exception():
                    run_popcon(case, show, cli_args)
            except ModuleNotFoundError:
                run_popcon(case, show, cli_args)
    else:
        run_popcon(case, show, cli_args)


@click.command()
@click.option("-o", "--output", default="./popcon_algorithms.yaml", type=click.Path(exists=False))
def write_algorithms_yaml(output: str) -> None:
    """Write all available algorithms to a yaml helper file."""
    from cfspopcon import Algorithm

    Algorithm.write_yaml(Path(output))


def run_popcon(case: str, show: bool, cli_args: dict[str, str]) -> None:
    """Run popcon case.

    Args:
        case: specify case to run (corresponding to a case in cases)
        show: show the resulting plots
        cli_args: command-line arguments, takes precedence over config.
    """
    input_parameters, algorithm, points, plots = read_case(case, cli_args)

    dataset = xr.Dataset(input_parameters)

    algorithm.validate_inputs(dataset)
    dataset = algorithm.update_dataset(dataset)

    output_dir = Path(case) / "output" if Path(case).is_dir() else Path(case).parent / "output"
    output_dir.mkdir(exist_ok=True)

    file_io.write_dataset_to_netcdf(dataset, filepath=output_dir / "dataset.nc")

    if points is not None:
        for point, point_params in points.items():
            file_io.write_point_to_file(dataset, point, point_params, output_dir=output_dir)

    # Plot the results
    if plots is not None:
        for plot_name, plot_style in plots.items():
            # print(f"Plotting {plot_name}")
            make_plot(dataset, read_plot_style(plot_style), points, title=plot_name, output_dir=output_dir, save_name=plot_style.stem)

    # print("Done")
    if show:
        # plt.show()
        pass


if __name__ == "__main__":
    run_popcon_cli()
