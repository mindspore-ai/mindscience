# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
# ============================================================================

"""Utilities for structure module testing."""

import dataclasses

from absl.testing import parameterized
from alphafold3 import structure
from alphafold3.common.testing import data
import numpy as np

import os
import contextlib
import datetime
import difflib
import functools
import hashlib
import shutil
import pathlib
from typing import Any
from absl.testing import absltest
import mindspore as ms
from alphafold3.common.testing import data as testing_data
from alphafold3.common import resources
from alphafold3.data import pipeline
from alphafold3.model.atom_layout import atom_layout

_JACKHMMER_BINARY_PATH = shutil.which('jackhmmer')
_NHMMER_BINARY_PATH = shutil.which('nhmmer')
_HMMALIGN_BINARY_PATH = shutil.which('hmmalign')
_HMMSEARCH_BINARY_PATH = shutil.which('hmmsearch')
_HMMBUILD_BINARY_PATH = shutil.which('hmmbuild')

@contextlib.contextmanager
def _output(name: str):
    with open(result_path := f'{absltest.TEST_TMPDIR.value}/{name}', "wb") as f:
        yield result_path, f


@functools.singledispatch
def _hash_data(x: Any, /) -> str:
    if x is None:
        return '<<None>>'
    return _hash_data(json.dumps(x).encode('utf-8'))


@_hash_data.register
def _(x: bytes, /) -> str:
    return hashlib.sha256(x).hexdigest()


@_hash_data.register
def _(x: ms.Tensor) -> str:
    return _hash_data(x.asnumpy())


@_hash_data.register
def _(x: np.ndarray) -> str:
    if x.dtype == object:
        return ';'.join(map(_hash_data, x.ravel().tolist()))
    return _hash_data(x.tobytes())


@_hash_data.register
def _(_: structure.Structure) -> str:
    return '<<structure>>'


@_hash_data.register
def _(_: atom_layout.AtomLayout) -> str:
    return '<<atom-layout>>'


def _generate_diff(actual: str, expected: str) -> str:
    return '\n'.join(
        difflib.unified_diff(
            expected.split('\n'),
            actual.split('\n'),
            fromfile='expected',
            tofile='actual',
            lineterm='',
        )
    )


def tree_map(func, dict_tree):
    if isinstance(dict_tree, dict):
        return {k: tree_map(func, v) for k, v in dict_tree.items()}
    else:
        if func == "asnumpy":
            return dict_tree.asnumpy()
        elif func == "float32":
            return dict_tree.astype(ms.float32)
        elif func == "bfloat16":
            return dict_tree.astype(ms.bfloat16)
        else:
            return func(dict_tree)

class StructureTestCase(parameterized.TestCase):
    """Testing utilities for working with structure.Structure."""

    def set_path(self, use_full_database=False):
        if use_full_database:
            small_bfd_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/bfd-first_non_consensus_sequences.fasta'
            ).path()
            mgnify_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/mgy_clusters_2022_05.fa'
            ).path()
            uniprot_cluster_annot_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/uniprot_all_2021_04.fa'
            ).path()
            uniref90_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/uniref90_2022_05.fa'
            ).path()
            ntrna_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta'
            ).path()
            rfam_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta'
            ).path()
            rna_central_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/rnacentral_active_seq_id_90_cov_80_linclust.fasta'
            ).path()
            pdb_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/mmcif_files'
            ).path()
            seqres_database_path = testing_data.Data(
                '/data/zmmVol2/AF3/public_databases/pdb_seqres_2022_09_28.fasta'
            ).path()
        else:
            small_bfd_database_path = testing_data.Data(
                resources.ROOT
                / 'test_data/miniature_databases/bfd-first_non_consensus_sequences__subsampled_1000.fasta'
            ).path()
            mgnify_database_path = testing_data.Data(
                resources.ROOT
                / 'test_data/miniature_databases/mgy_clusters__subsampled_1000.fa'
            ).path()
            uniprot_cluster_annot_database_path = testing_data.Data(
                resources.ROOT
                / 'test_data/miniature_databases/uniprot_all__subsampled_1000.fasta'
            ).path()
            uniref90_database_path = testing_data.Data(
                resources.ROOT
                / 'test_data/miniature_databases/uniref90__subsampled_1000.fasta'
            ).path()
            ntrna_database_path = testing_data.Data(
                resources.ROOT
                / ('test_data/miniature_databases/'
                   'nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq__subsampled_1000.fasta')
            ).path()
            rfam_database_path = testing_data.Data(
                resources.ROOT
                / 'test_data/miniature_databases/rfam_14_4_clustered_rep_seq__subsampled_1000.fasta'
            ).path()
            rna_central_database_path = testing_data.Data(
                resources.ROOT
                / 'test_data/miniature_databases/rnacentral_active_seq_id_90_cov_80_linclust__subsampled_1000.fasta'
            ).path()
            pdb_database_path = testing_data.Data(
                resources.ROOT / 'test_data/miniature_databases/pdb_mmcif'
            ).path()
            seqres_database_path = testing_data.Data(
                resources.ROOT
                / 'test_data/miniature_databases/pdb_seqres_2022_09_28__subsampled_1000.fasta'
            ).path()

        self._data_pipeline_config = pipeline.DataPipelineConfig(
            jackhmmer_binary_path=_JACKHMMER_BINARY_PATH,
            nhmmer_binary_path=_NHMMER_BINARY_PATH,
            hmmalign_binary_path=_HMMALIGN_BINARY_PATH,
            hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH,
            hmmbuild_binary_path=_HMMBUILD_BINARY_PATH,
            small_bfd_database_path=small_bfd_database_path,
            mgnify_database_path=mgnify_database_path,
            uniprot_cluster_annot_database_path=uniprot_cluster_annot_database_path,
            uniref90_database_path=uniref90_database_path,
            ntrna_database_path=ntrna_database_path,
            rfam_database_path=rfam_database_path,
            rna_central_database_path=rna_central_database_path,
            pdb_database_path=pdb_database_path,
            seqres_database_path=seqres_database_path,
            max_template_date=datetime.date(2021, 9, 30),
        )
        self.data_path = "/data/zmmVol2/AF3/run_test/src/alphafold3/test_data"

    def compare_golden(self, result_path: str, golden_path) -> None:
        filename = os.path.split(result_path)[1]
        golden_path = pathlib.Path(golden_path)
        with open(golden_path, 'r') as golden_file:
            golden_text = golden_file.read()
        with open(result_path, 'r') as result_file:
            result_text = result_file.read()

        diff = _generate_diff(result_text, golden_text)

        self.assertEqual(diff, "", f"Result differs from golden:\n{diff}")

    def assertAuthorNamingSchemeEqual(self, ans1, ans2):  # pylint: disable=invalid-name
        """Walks naming scheme, making sure all elements are equal."""
        if ans1 is None or ans2 is None:
            self.assertIsNone(ans1)
            self.assertIsNone(ans2)
            return
        flat_ans1 = dict(tree.flatten_with_path(dataclasses.asdict(ans1)))
        flat_ans2 = dict(tree.flatten_with_path(dataclasses.asdict(ans2)))
        for k, v in flat_ans1.items():
            self.assertEqual(v, flat_ans2[k], msg=str(k))
        for k, v in flat_ans2.items():
            self.assertEqual(v, flat_ans1[k], msg=str(k))

    def assertAllResiduesEqual(self, all_res1, all_res2):  # pylint: disable=invalid-name
        """Walks all residues, making sure alll elements are equal."""
        if all_res1 is None or all_res2 is None:
            self.assertIsNone(all_res1)
            self.assertIsNone(all_res2)
            return
        self.assertSameElements(all_res1.keys(), all_res2.keys())
        for chain_id, chain_res in all_res1.items():
            self.assertSequenceEqual(
                chain_res, all_res2[chain_id], msg=chain_id)

    def assertBioassemblyDataEqual(self, data1, data2):  # pylint: disable=invalid-name
        if data1 is None or data2 is None:
            self.assertIsNone(data1)
            self.assertIsNone(data2)
            return
        self.assertDictEqual(data1.to_mmcif_dict(), data2.to_mmcif_dict())

    def assertChemicalComponentsDataEqual(  # pylint: disable=invalid-name
        self,
        data1,
        data2,
        allow_chem_comp_data_extension,
    ):
        """Checks whether two ChemicalComponentData objects are considered equal."""
        if data1 is None or data2 is None:
            self.assertIsNone(data1)
            self.assertIsNone(data2)
            return
        if (not allow_chem_comp_data_extension) or (
            data1.chem_comp.keys() ^ data2.chem_comp.keys()
        ):
            self.assertDictEqual(data1.chem_comp, data2.chem_comp)
        else:
            mismatching_values = []
            for component_id in data1.chem_comp:
                found = data1.chem_comp[component_id]
                expected = data2.chem_comp[component_id]
                if not found.extends(expected):
                    mismatching_values.append((component_id, expected, found))

            if mismatching_values:
                mismatch_err_msgs = '\n'.join(
                    f'{component_id}: {expected} or its extension expected,'
                    f' but {found} found.'
                    for component_id, expected, found in mismatching_values
                )
                self.fail(
                    f'Mismatching values for `_chem_comp` table: {mismatch_err_msgs}',
                )

    def assertBondsEqual(self, bonds1, bonds2, atom_key1, atom_key2):  # pylint: disable=invalid-name
        """Checks whether two Bonds objects are considered equal."""
        # An empty bonds table is functionally equivalent to an empty bonds table.
        # NB: this can only ever be None in structure v1.
        if bonds1 is None or not bonds1.size or bonds2 is None or not bonds2.size:
            self.assertTrue(bonds1 is None or not bonds1.size,
                            msg=f'{bonds1=}')
            self.assertTrue(bonds2 is None or not bonds2.size,
                            msg=f'{bonds2=}')
            return

        ptnr1_indices1, ptnr2_indices1 = bonds1.get_atom_indices(atom_key1)
        ptnr1_indices2, ptnr2_indices2 = bonds2.get_atom_indices(atom_key2)
        np.testing.assert_array_equal(ptnr1_indices1, ptnr1_indices2)
        np.testing.assert_array_equal(ptnr2_indices1, ptnr2_indices2)
        np.testing.assert_array_equal(bonds1.type, bonds2.type)
        np.testing.assert_array_equal(bonds1.role, bonds2.role)

    def assertStructuresEqual(  # pylint: disable=invalid-name
        self,
        struc1,
        struc2,
        *,
        ignore_fields=None,
        allow_chem_comp_data_extension=False,
        atol=0,
    ):
        """Checks whether two Structure objects could be considered equal.

        Args:
          struc1: First Structure object.
          struc2: Second Structure object.
          ignore_fields: Fields not taken into account during comparison.
          allow_chem_comp_data_extension: Whether to allow data of `_chem_comp`
            table to differ if `struc2` is missing some fields, but `struc1` has
            specific values for them.
          atol: Absolute tolerance for floating point comparisons (in
            np.testing.assert_allclose).
        """
        for field in sorted(structure.GLOBAL_FIELDS):
            if ignore_fields and field in ignore_fields:
                continue
            if field == 'author_naming_scheme':
                self.assertAuthorNamingSchemeEqual(
                    struc1[field], struc2[field])
            elif field == 'all_residues':
                self.assertAllResiduesEqual(struc1[field], struc2[field])
            elif field == 'bioassembly_data':
                self.assertBioassemblyDataEqual(struc1[field], struc2[field])
            elif field == 'chemical_components_data':
                self.assertChemicalComponentsDataEqual(
                    struc1[field], struc2[field], allow_chem_comp_data_extension
                )
            elif field == 'bonds':
                self.assertBondsEqual(
                    struc1.bonds, struc2.bonds, struc1.atom_key, struc2.atom_key
                )
            else:
                self.assertEqual(struc1[field], struc2[field], msg=field)

        # The chain order within a structure is arbitrary so in order to
        # directly compare arrays we first align struc1 to struc2 and check that
        # the number of atoms doesn't change.
        num_atoms = struc1.num_atoms
        self.assertEqual(struc2.num_atoms, num_atoms)
        struc1 = struc1.order_and_drop_atoms_to_match(struc2)
        self.assertEqual(struc1.num_atoms, num_atoms)

        for field in sorted(structure.ARRAY_FIELDS):
            if field == 'atom_key':
                # atom_key has no external meaning, so it doesn't matter whether it
                # differs between two structures.
                continue
            if ignore_fields and field in ignore_fields:
                continue
            self.assertEqual(struc1[field] is None,
                             struc2[field] is None, msg=field)

            if np.issubdtype(struc1[field].dtype, np.inexact):
                np.testing.assert_allclose(
                    struc1[field], struc2[field], err_msg=field, atol=atol
                )
            else:
                np.testing.assert_array_equal(
                    struc1[field], struc2[field], err_msg=field
                )
