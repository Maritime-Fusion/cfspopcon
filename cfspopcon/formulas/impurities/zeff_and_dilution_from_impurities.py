"""Calculate the impact of core impurities on z_effective and dilution."""

import xarray as xr
from pathlib import Path
from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull
from .impurity_charge_state import calc_impurity_charge_state


@Algorithm.register_algorithm(
    return_keys=[
        "impurity_charge_state",
        "change_in_zeff",
        "change_in_dilution",
        "z_effective",
        "dilution",
        "summed_impurity_density",
        "average_ion_density",
    ]
)
def calc_zeff_and_dilution_due_to_impurities(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    impurity_concentration: xr.DataArray,
    atomic_data: xr.DataArray,
) -> tuple[Unitfull, ...]:
    """Calculate the impact of core impurities on z_effective and dilution.

    Args:
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        impurity_concentration: :term:`glossary link<impurity_concentration>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
        :term:`impurity_charge_state`, :term:`change_in_zeff`, :term:`change_in_dilution`, :term:`z_effective`, :term:`dilution`, :term:`summed_impurity_density`, :term:`average_ion_density`
    """
    starting_zeff = 1.0
    starting_dilution = 1.0

    impurity_charge_state = calc_impurity_charge_state(
        average_electron_density, average_electron_temp, impurity_concentration, atomic_data
    )
    change_in_zeff = calc_change_in_zeff(impurity_charge_state, impurity_concentration)
    change_in_dilution = calc_change_in_dilution(impurity_charge_state, impurity_concentration)

    # Sum over species to obtain effective values.
    z_effective = starting_zeff + change_in_zeff.sum(dim="dim_species")
    dilution = starting_dilution - change_in_dilution.sum(dim="dim_species")

    # Prevent negative dilution values.
    dilution = dilution.where(dilution >= 0, 0.0)

    # Create output directory.
    output_dir = Path('fusion_data')
    output_dir.mkdir(exist_ok=True)

    # Define a helper function to save binary data and print array details.
    def save_binary(filename, array):
        # Print the shape and size of the array before saving.
        print(f"Saving {filename}: shape = {array.shape}, total elements = {array.size}")
        with open(filename, 'wb') as f:
            f.write(array.values.tobytes())

    # Save z_effective to a binary file.
    save_binary(output_dir / 'z_effective.bin', z_effective)
    # Save dilution to a binary file.
    save_binary(output_dir / 'dilution.bin', dilution)

    # Calculate summed impurity density.
    summed_impurity_density = impurity_concentration.sum(dim="dim_species") * average_electron_density
    save_binary(output_dir / 'summed_impurity_density.bin', summed_impurity_density)

    # Calculate average ion density.
    average_ion_density = dilution * average_electron_density

    return (
        impurity_charge_state,
        change_in_zeff,
        change_in_dilution,
        z_effective,
        dilution,
        summed_impurity_density,
        average_ion_density,
    )



def calc_change_in_zeff(impurity_charge_state: float, impurity_concentration: xr.DataArray) -> xr.DataArray:
    """Calculate the change in the effective charge due to the specified impurities.

    Args:
        impurity_charge_state: [~] :term:`glossary link<impurity_charge_state>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`

    Returns:
        change in zeff [~]
    """
    return impurity_charge_state * (impurity_charge_state - 1.0) * impurity_concentration


def calc_change_in_dilution(impurity_charge_state: float, impurity_concentration: xr.DataArray) -> xr.DataArray:
    """Calculate the change in n_fuel/n_e due to the specified impurities.

    Args:
        impurity_charge_state: [~] :term:`glossary link<impurity_charge_state>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`

    Returns:
        change in dilution [~]
    """
    return impurity_charge_state * impurity_concentration
