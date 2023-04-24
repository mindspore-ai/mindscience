# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"search"
import os
import time
import random
import tarfile
import logging
import stat
from io import StringIO
from pathlib import Path
import shutil
import requests
from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict
from tqdm import tqdm
from mindsponge.common import residue_constants

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
DEFAULT_API_SERVER = "https://api.colabfold.com"


def validate_and_fix_mmcif(cif_file: Path):
    """validate presence of _entity_poly_seq in cif file and add revision_date if missing"""
    # check that required poly_seq and revision_date fields are present
    cif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
    required = [
        "_chem_comp.id",
        "_chem_comp.type",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        "_entity_poly_seq.mon_id",
    ]
    for r in required:
        if r not in cif_dict:
            raise ValueError(f"mmCIF file {cif_file} is missing required field {r}.")
    if "_pdbx_audit_revision_history.revision_date" not in cif_dict:
        logger.info(
            f"Adding missing field revision_date to {cif_file}. Backing up original file to {cif_file}.bak."
        )
        shutil.copy2(cif_file, str(cif_file) + ".bak")
        os_flags = os.O_RDWR | os.O_CREAT
        os_modes = stat.S_IRWXU
        with os.fdopen(os.open(cif_file, os_flags, os_modes), 'a') as f:
            f.write(CIF_REVISION_DATE)


def convert_pdb_to_mmcif(pdb_file: Path):
    """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
    i = pdb_file.stem
    cif_file = pdb_file.parent.joinpath(f"{i}.cif")
    if cif_file.is_file():
        return
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(i, pdb_file)
    cif_io = CFMMCIFIO()
    cif_io.set_structure(structure)
    cif_io.save(str(cif_file), ReplaceOrRemoveHetatmSelect())


def mk_hhsearch_db(template_dir: str):
    '''colabsearch_db'''
    template_path = Path(template_dir)

    cif_files = template_path.glob("*.cif")
    for cif_file in cif_files:
        validate_and_fix_mmcif(cif_file)

    pdb_files = template_path.glob("*.pdb")
    for pdb_file in pdb_files:
        convert_pdb_to_mmcif(pdb_file)

    pdb70_db_files = template_path.glob("pdb70*")
    for f in pdb70_db_files:
        os.remove(f)

    os_flags = os.O_RDWR | os.O_CREAT
    os_modes = stat.S_IRWXU

    with os.fdopen(os.open(template_path.joinpath("pdb70_a3m.ffdata"), os_flags, os_modes), 'w') \
            as a3m, os.fdopen(os.open(template_path.joinpath("pdb70_cs219.ffindex"), \
                                      os_flags, os_modes), 'w') as cs219_index, os.fdopen( \
            os.open(template_path.joinpath("pdb70_a3m.ffindex"), \
                    os_flags, os_modes), 'w') as a3m_index, os.fdopen(os.open( \
            template_path.joinpath("pdb70_cs219.ffdata"), os_flags, os_modes), 'w') as cs219:
        n = 1000000
        index_offset = 0
        cif_files = template_path.glob("*.cif")
        for cif_file in cif_files:
            with open(cif_file) as f:
                cif_string = f.read()
            cif_fh = StringIO(cif_string)
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("none", cif_fh)
            models = list(structure.get_models())
            model = models[0]
            for chain in model:
                amino_acid_res = []
                for res in chain:
                    if res.id[2] != " ":
                        continue
                    amino_acid_res.append(
                        residue_constants.restype_3to1.get(res.resname, "X")
                    )

                protein_str = "".join(amino_acid_res)
                a3m_str = f">{cif_file.stem}_{chain.id}\n{protein_str}\n\0"
                a3m_str_len = len(a3m_str)
                a3m_index.write(f"{n}\t{index_offset}\t{a3m_str_len}\n")
                cs219_index.write(f"{n}\t{index_offset}\t{len(protein_str)}\n")
                index_offset += a3m_str_len
                a3m.write(a3m_str)
                cs219.write("\n\0")
                n += 1


def run_mmseqs2(x, a3m_result_path, template_path, use_env=True, use_filters=True,
                use_templates=True, filters=None, use_pairing=False,
                host_url="https://api.colabfold.com"):
    '''run_mmseqs2'''
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    def submit(seqs, mode, ns=101):
        n, query = ns, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                res = requests.post(f'{host_url}/{submission_endpoint}', data={'q': query, 'mode': mode}, timeout=6.02,
                                    verify=False)
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ids):
        while True:
            error_count = 0
            try:
                res = requests.get(f'{host_url}/ticket/{ids}', timeout=6.02, verify=False)
            except requests.exceptions.Timeout:
                logger.warning("Timeout while fetching status from MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ids, path):
        error_count = 0
        while True:
            try:
                res = requests.get(f'{host_url}/result/download/{ids}', timeout=6.02, verify=False)
            except requests.exceptions.Timeout:
                logger.warning("Timeout while fetching result from MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        os_flags = os.O_RDWR | os.O_CREAT
        os_modes = stat.S_IRWXU
        with os.fdopen(os.open(path, os_flags, os_modes), 'wb') as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filters is not None:
        use_filters = filters

    # setup mode
    if use_filters:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilters" if use_env else "nofilters"

    if use_pairing:
        mode = ""
        use_templates = False
        use_env = False

    # define path
    path = a3m_result_path
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f'{path}/out.tar.gz'
    ns, redo = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    # lets do it!

    if not os.path.isfile(tar_gz_file):
        time_estimate = 150 * len(seqs_unique)
        with tqdm(total=time_estimate, bar_format=TQDM_BAR_FORMAT) as pbar:
            while redo:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, ns)
                while out.get("status") in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out.get('status')}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, ns)

                if out.get("status") == "ERROR":
                    raise Exception(
                        f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. '
                        f'If error persists, please try again an hour later.')

                if out.get("status") == "MAINTENANCE":
                    raise Exception(f'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

                # wait for job to finish
                ids, times1 = out.get("id"), 0
                pbar.set_description(out.get("status"))
                while out.get("status") in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out.get('status')}")
                    time.sleep(t)
                    out = status(ids)
                    pbar.set_description(out.get("status"))
                    if out.get("status") == "RUNNING":
                        times1 += t
                        pbar.update(n=t)

                if out.get("status") == "COMPLETE":
                    if times1 < time_estimate:
                        pbar.update(n=(time_estimate - times1))
                    redo = False

                if out.get("status") == "ERROR":
                    raise Exception(
                        f'MMseqs2 API is giving errors. Please confirm your input is a valid protein '
                        f'sequence. If error persists, please try again an hour later.')

            # Download results
            download(ids, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # templates
    if use_templates:
        templates = {}
        for line in open(f"{path}/pdb70.m8", "r"):
            p = line.rstrip().split()
            ms, pdb, _, _ = p[0], p[1], p[2], p[10]
            ms = int(ms)
            if ms not in templates:
                templates[ms] = []
            templates.get(ms).append(pdb)

        template_paths = {}
        for k, tmpl in templates.items():
            tmpl_path = f"{template_path}_{k}"
            if not os.path.isdir(tmpl_path):
                os.mkdir(tmpl_path)
                tmpl_line = ",".join(tmpl[:20])
                response = None
                while True:
                    error_count = 0
                    try:
                        response = requests.get(f"{host_url}/template/{tmpl_line}", stream=True, timeout=6.02,
                                                verify=False)
                    except requests.exceptions.Timeout:
                        logger.warning("Timeout while submitting to template server. Retrying...")
                        continue
                    except Exception as e:
                        error_count += 1
                        logger.warning(
                            f"Error while fetching result from template server. Retrying... ({error_count}/5)")
                        logger.warning(f"Error: {e}")
                        time.sleep(5)
                        if error_count > 5:
                            raise
                        continue
                    break
                with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                    tar.extractall(path=tmpl_path)
                os.symlink("pdb70_a3m.ffindex", f"{tmpl_path}/pdb70_cs219.ffindex")
                os_flags = os.O_RDWR | os.O_CREAT
                os_modes = stat.S_IRWXU
                with os.fdopen(os.open(f"{tmpl_path}/pdb70_cs219.ffdata", os_flags, os_modes), 'w') as f:
                    f.write("")
            template_paths[k] = tmpl_path

    # gather a3m lines
    os.system(f"cp -r {tmpl_path}/* {template_path} && rm -rf {tmpl_path}")


def colabsearch(sequence, a3m_result_path, template_path):
    run_mmseqs2(sequence, a3m_result_path, template_path, use_filters=True,
                host_url="https://a3m.mmseqs.com")
    mk_hhsearch_db(template_path)
