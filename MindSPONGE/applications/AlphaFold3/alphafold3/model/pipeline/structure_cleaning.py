# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Prepare PDB structure for training or inference."""

from typing import Any, Optional
import numpy as np
from absl import logging
from alphafold3 import structure
from alphafold3.constants import chemical_component_sets
from alphafold3.constants import chemical_components
from alphafold3.constants import mmcif_names
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.pipeline import inter_chain_bonds
from alphafold3.model.scoring import covalent_bond_cleaning
from alphafold3.structure import sterics


def _get_leaving_atom_mask(
    struct: structure.Structure,
    polymer_ligand_bonds: Optional[atom_layout.AtomLayout],
    ligand_ligand_bonds: Optional[atom_layout.AtomLayout],
    chain_id: str,
    chain_type: str,
    res_id: int,
    res_name: str,
) -> np.ndarray:
    """Updates a drop_leaving_atoms mask with new leaving atom locations."""
    bonded_atoms = atom_layout.get_bonded_atoms(
        polymer_ligand_bonds,
        ligand_ligand_bonds,
        res_id,
        chain_id,
    )
    # Connect the amino-acids, i.e. remove OXT, HXT and H2.
    drop_atoms = atom_layout.get_link_drop_atoms(
        res_name=res_name,
        chain_type=chain_type,
        is_start_terminus=False,
        is_end_terminus=False,
        bonded_atoms=bonded_atoms,
        drop_ligand_leaving_atoms=True,
    )
    # Default mask where everything is false, which equates to being kept.
    drop_atom_filter_atoms = struct.chain_id != struct.chain_id
    for drop_atom in drop_atoms:
        drop_atom_filter_atom = np.logical_and(
            np.logical_and(
                struct.atom_name == drop_atom,
                struct.chain_id == chain_id,
            ),
            struct.res_id == res_id,
        )
        drop_atom_filter_atoms = np.logical_or(
            drop_atom_filter_atoms, drop_atom_filter_atom
        )
    return drop_atom_filter_atoms


def clean_structure(
    struct: structure.Structure,
    ccd: chemical_components.Ccd,
    *,
    drop_missing_sequence: bool,
    filter_clashes: bool,
    drop_non_standard_atoms: bool,
    filter_crystal_aids: bool,
    filter_waters: bool,
    filter_hydrogens: bool,
    filter_leaving_atoms: bool,
    only_glycan_ligands_for_leaving_atoms: bool,
    covalent_bonds_only: bool,
    remove_polymer_polymer_bonds: bool,
    remove_bad_bonds: bool,
    remove_nonsymmetric_bonds: bool,
) -> tuple[structure.Structure, dict[str, Any]]:
    """Cleans structure.

    Args:
        struct: Structure to clean.
        ccd: The chemical components dictionary.
        drop_missing_sequence: Whether to drop chains without specified sequences.
        filter_clashes: Whether to drop clashing chains.
        drop_non_standard_atoms: Whether to drop non CCD standard atoms.
        filter_crystal_aids: Whether to drop ligands in the crystal aid set.
        filter_waters: Whether to drop water chains.
        filter_hydrogens: Whether to drop hyrdogen atoms.
        filter_leaving_atoms: Whether to drop leaving atoms based on heuristics.
        only_glycan_ligands_for_leaving_atoms: Whether to only include glycan
            ligands when filtering leaving atoms.
        covalent_bonds_only: Only include covalent bonds.
        remove_polymer_polymer_bonds: Remove polymer-polymer bonds.
        remove_bad_bonds: Whether to remove badly bonded ligands.
        remove_nonsymmetric_bonds: Whether to remove nonsymmetric polymer-ligand
            bonds from symmetric polymer chains.

    Returns:
        Tuple of structure and metadata dict. The metadata dict has
        information about what was cleaned from the original.
    """

    metadata = {}
    # Crop crystallization aids.
    if (
        filter_crystal_aids
        and struct.structure_method in mmcif_names.CRYSTALLIZATION_METHODS
    ):
        struct = struct.filter_out(
            res_name=chemical_component_sets.COMMON_CRYSTALLIZATION_AIDS
        )

    # Drop chains without specified sequences.
    if drop_missing_sequence:
        chains_with_unk_sequence = struct.find_chains_with_unknown_sequence()
        num_with_unk_sequence = len(chains_with_unk_sequence)
        if chains_with_unk_sequence:
            struct = struct.filter_out(chain_id=chains_with_unk_sequence)
    else:
        num_with_unk_sequence = 0
    metadata['num_with_unk_sequence'] = num_with_unk_sequence

    # Remove intersecting chains.
    if filter_clashes and struct.num_chains > 1:
        clashing_chains = sterics.find_clashing_chains(struct)
        if clashing_chains:
            struct = struct.filter_out(chain_id=clashing_chains)
    else:
        clashing_chains = []
    metadata['num_clashing_chains_removed'] = len(clashing_chains)
    metadata['chains_removed'] = clashing_chains

    # Drop non-standard atoms
    if drop_non_standard_atoms:
        struct = struct.drop_non_standard_atoms(
            ccd=ccd, drop_unk=False, drop_non_ccd=False
        )

    # Sort chains in "reverse-spreadsheet" order.
    struct = struct.with_sorted_chains

    if filter_hydrogens:
        struct = struct.without_hydrogen()

    if filter_waters:
        struct = struct.filter_out(chain_type=mmcif_names.WATER)

    if filter_leaving_atoms:
        drop_leaving_atoms_all = struct.chain_id != struct.chain_id
        polymer_ligand_bonds = inter_chain_bonds.get_polymer_ligand_bonds(
            struct,
            only_glycan_ligands=only_glycan_ligands_for_leaving_atoms,
        )
        ligand_ligand_bonds = inter_chain_bonds.get_ligand_ligand_bonds(
            struct,
            only_glycan_ligands=only_glycan_ligands_for_leaving_atoms,
        )
        all_glycans = {
            *chemical_component_sets.GLYCAN_OTHER_LIGANDS,
            *chemical_component_sets.GLYCAN_LINKING_LIGANDS,
        }
        # If only glycan ligands and no O1 atoms, we can do parallel drop.
        if (
            only_glycan_ligands_for_leaving_atoms
            and (not (ligand_ligand_bonds.atom_name == 'O1').any())
            and (not (polymer_ligand_bonds.atom_name == 'O1').any())
        ):
            drop_leaving_atoms_all = np.logical_and(
                np.isin(struct.atom_name, 'O1'),
                np.isin(struct.res_name, list(all_glycans)),
            )
        else:
            substruct = struct.group_by_residue
            glycan_mask = np.isin(substruct.res_name, list(all_glycans))
            substruct = substruct.filter(glycan_mask)
            # We need to iterate over all glycan residues for this.
            for res in substruct.iter_residues():
                # Only need to do drop leaving atoms for glycans depending on bonds.
                if (res_name := res['res_name']) in all_glycans:
                    drop_atom_filter = _get_leaving_atom_mask(
                        struct=struct,
                        polymer_ligand_bonds=polymer_ligand_bonds,
                        ligand_ligand_bonds=ligand_ligand_bonds,
                        chain_id=res['chain_id'],
                        chain_type=res['chain_type'],
                        res_id=res['res_id'],
                        res_name=res_name,
                    )
                    drop_leaving_atoms_all = np.logical_or(
                        drop_leaving_atoms_all, drop_atom_filter
                    )

        num_atoms_before = struct.num_atoms
        struct = struct.filter_out(drop_leaving_atoms_all)
        num_atoms_after = struct.num_atoms

        if num_atoms_before > num_atoms_after:
            logging.error(
                'Dropped %s atoms from GT struct: chain_id %s res_id %s res_name %s',
                num_atoms_before - num_atoms_after,
                struct.chain_id,
                struct.res_id,
                struct.res_name,
            )

    # Can filter by bond type without having to iterate over bonds.
    if struct.bonds and covalent_bonds_only:
        is_covalent = np.isin(struct.bonds.type, ['covale'])
        if sum(is_covalent) > 0:
            new_bonds = struct.bonds[is_covalent]
        else:
            new_bonds = structture.Bonds.make_empty()
        struct = struct.copy_and_update(bonds=new_bonds)

    # Other bond filters require iterating over individual bonds.
    if struct.bonds and (remove_bad_bonds or remove_polymer_polymer_bonds):
        include_bond = []
        num_pp_bonds = 0
        num_bad_bonds = 0
        for bond in struct.iter_bonds():
            dest_atom = bond.dest_atom
            from_atom = bond.from_atom
            if remove_polymer_polymer_bonds:
                if (
                    from_atom['chain_type'] in mmcif_names.POLYMER_CHAIN_TYPES
                    and dest_atom['chain_type'] in mmcif_names.POLYMER_CHAIN_TYPES
                ):
                    num_pp_bonds += 1
                    include_bond.append(False)
                    continue
            if remove_bad_bonds:
                dest_coords = np.array(
                    [dest_atom['atom_x'], dest_atom['atom_y'], dest_atom['atom_z']]
                )
                from_coords = np.array(
                    [from_atom['atom_x'], from_atom['atom_y'], from_atom['atom_z']]
                )
                squared_dist = np.sum(np.square(dest_coords - from_coords))
                squared_threshold = 2.4 * 2.4
                if squared_dist > squared_threshold:
                    num_bad_bonds += 1
                    include_bond.append(False)
                    continue
            include_bond.append(True)
        if sum(include_bond) < len(struct.bonds):
            logging.info(
                'Reducing number of bonds for %s from %s to %s, of which %s are'
                ' polymer-polymer bonds and %s are bad bonds.',
                struct.name,
                len(struct.bonds),
                sum(include_bond),
                num_pp_bonds,
                num_bad_bonds,
            )
            if sum(include_bond) > 0:
                # Need to index bonds with bond keys or arrays of bools with same length
                # as num bonds. In this case, we use array of bools (as elsewhere in the
                # cleaning code).
                new_bonds = struct.bonds[np.array(include_bond, dtype=bool)]
            else:
                new_bonds = structure.Bonds.make_empty()
            struct = struct.copy_and_update(bonds=new_bonds)

    if struct.bonds and remove_nonsymmetric_bonds:
        # Check for asymmetric polymer-ligand bonds and remove if these exist.
        polymer_ligand_bonds = inter_chain_bonds.get_polymer_ligand_bonds(
            struct,
            only_glycan_ligands=False,
        )
        if polymer_ligand_bonds:
            if covalent_bond_cleaning.has_nonsymmetric_bonds_on_symmetric_polymer_chains(
                struct, polymer_ligand_bonds
            ):
                from_atom_idxs, dest_atom_idxs = struct.bonds.get_atom_indices(
                    struct.atom_key
                )
                poly_chain_types = list(mmcif_names.POLYMER_CHAIN_TYPES)
                is_polymer_bond = np.logical_or(
                    np.isin(
                        struct.chain_type[from_atom_idxs], poly_chain_types),
                    np.isin(
                        struct.chain_type[dest_atom_idxs], poly_chain_types),
                )
                struct = struct.copy_and_update(
                    bonds=struct.bonds[~is_polymer_bond])

    return struct, metadata


def create_empty_output_struct_and_layout(
    struct: structure.Structure,
    ccd: chemical_components.Ccd,
    *,
    with_hydrogens: bool = False,
    skip_unk: bool = False,
    polymer_ligand_bonds: Optional[atom_layout.AtomLayout] = None,
    ligand_ligand_bonds: Optional[atom_layout.AtomLayout] = None,
    drop_ligand_leaving_atoms: bool = False,
) -> tuple[structure.Structure, atom_layout.AtomLayout]:
    """Make zero-coordinate structure from all physical residues.

    Args:
        struct: Structure object.
        ccd: The chemical components dictionary.
        with_hydrogens: Whether to keep hydrogen atoms in structure.
        skip_unk: Whether to remove unknown residues from structure.
        polymer_ligand_bonds: Bond information for polymer-ligand pairs.
        ligand_ligand_bonds: Bond information for ligand-ligand pairs.
        drop_ligand_leaving_atoms: Flag for handling leaving atoms for ligands.

    Returns:
        Tuple of structure with all bonds, physical residues and coordinates set to
        0 and a flat atom layout of empty structure.
    """
    bonded_atom_pairs = []
    if polymer_ligand_bonds:
        for chain_ids, res_ids, atom_names in zip(
            polymer_ligand_bonds.chain_id,
            polymer_ligand_bonds.res_id,
            polymer_ligand_bonds.atom_name,
            strict=True,
        ):
            bonded_atom_pairs.append((
                (chain_ids[0], res_ids[0], atom_names[0]),
                (chain_ids[1], res_ids[1], atom_names[1]),
            ))
    if ligand_ligand_bonds:
        for chain_ids, res_ids, atom_names in zip(
            ligand_ligand_bonds.chain_id,
            ligand_ligand_bonds.res_id,
            ligand_ligand_bonds.atom_name,
            strict=True,
        ):
            bonded_atom_pairs.append((
                (chain_ids[0], res_ids[0], atom_names[0]),
                (chain_ids[1], res_ids[1], atom_names[1]),
            ))
    residues = atom_layout.residues_from_structure(
        struct, include_missing_residues=True
    )

    flat_output_layout = atom_layout.make_flat_atom_layout(
        residues,
        ccd=ccd,
        with_hydrogens=with_hydrogens,
        skip_unk_residues=skip_unk,
        polymer_ligand_bonds=polymer_ligand_bonds,
        ligand_ligand_bonds=ligand_ligand_bonds,
        drop_ligand_leaving_atoms=drop_ligand_leaving_atoms,
    )

    empty_output_struct = atom_layout.make_structure(
        flat_layout=flat_output_layout,
        atom_coords=np.zeros((flat_output_layout.shape[0], 3)),
        name=struct.name,
        atom_b_factors=None,
        all_physical_residues=residues,
    )
    if bonded_atom_pairs:
        empty_output_struct = empty_output_struct.add_bonds(
            bonded_atom_pairs, bond_type=mmcif_names.COVALENT_BOND
        )

    return empty_output_struct, flat_output_layout
