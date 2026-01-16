# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Colab request utils module.
"""

import logging
import os
import tarfile
import time
from typing import List, Tuple, Dict

import requests
from tqdm import tqdm

TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} estimate remaining: {remaining}]"
logger = logging.getLogger(__name__)

username = "example_user"
password = "example_password"


def parse_fasta_string(fasta_string: str) -> Dict:
    fasta_dict = {}
    lines = fasta_string.strip().split("\n")
    for line in lines:
        if line.startswith(">"):
            header = line[1:].strip()
            fasta_dict[header] = ""
        else:
            fasta_dict[header] += line.strip()
    return fasta_dict


def _make_http_request_with_retry(url, method='get', data=None, timeout=6.02, headers=None):
    """Make HTTP request with retry logic."""
    error_count = 0
    while True:
        try:
            if method == 'post':
                res = requests.post(
                    url, data=data, timeout=timeout, headers=headers, verify=False)
            else:
                res = requests.get(url, timeout=timeout,
                                   headers=headers, verify=False)
            return res
        except requests.exceptions.Timeout:
            logger.warning(
                "Timeout while %sing to MSA server. Retrying...", method)
            continue
        except Exception as e:
            error_count += 1
            logger.warning(
                "Error while fetching result from MSA server. Retrying... (%s/5)",
                error_count,
            )
            logger.warning("Error: %s", e)
            time.sleep(5)
            if error_count > 5:
                raise
            continue


def _parse_json_response(res):
    """Parse JSON response from server."""
    try:
        out = res.json()
    except ValueError:
        logger.error("Server didn't reply with json: %s", res.text)
        out = {"status": "ERROR"}
    return out


def _wait_for_job_completion(host_url, job_id, headers, pbar):
    """Wait for MSA job to complete."""
    elapsed_time = 0
    while True:
        res = _make_http_request_with_retry(
            f"{host_url}/ticket/{job_id}",
            method='get',
            headers=headers,
        )
        out = _parse_json_response(res)
        pbar.set_description(out["status"])

        if out["status"] not in ["UNKNOWN", "RUNNING", "PENDING"]:
            break

        t = 60
        logger.error("Sleeping for %ss. Reason: %s", t, out["status"])
        time.sleep(t)
        if out["status"] == "RUNNING":
            elapsed_time += t
        pbar.n = min(99, int(100 * elapsed_time / (30.0 * 60)))
        pbar.refresh()

    return out


def _submit_msa_job(host_url, submission_endpoint, seqs_unique, mode, num_sequences, email, headers):
    """Submit MSA job and handle retries."""
    n_seq, query = num_sequences, ""
    for seq in seqs_unique:
        query += f"{seq}\n"
        n_seq += 1

    out = _make_http_request_with_retry(
        f"{host_url}/{submission_endpoint}",
        method='post',
        data={"q": query, "mode": mode, "email": email},
        headers=headers,
    )
    return _parse_json_response(out)


def _handle_job_submission_errors(
    out,
    seqs_unique,
    mode,
    num_sequences,
    host_url,
    submission_endpoint,
    email,
    headers,
):
    """Handle errors during job submission."""
    while out["status"] in ["UNKNOWN", "RATELIMIT"]:
        sleep_time = 60
        logger.error("Sleeping for %ss. Reason: %s", sleep_time, out["status"])
        time.sleep(sleep_time)
        out = _submit_msa_job(host_url, submission_endpoint,
                              seqs_unique, mode, num_sequences, email, headers)

    if out["status"] == "ERROR":
        raise Exception(
            "MMseqs2 API is giving errors. Please confirm your input is a "
            "valid protein sequence. If error persists, please try again "
            "an hour later."
        )

    if out["status"] == "MAINTENANCE":
        raise Exception(
            "MMseqs2 API is undergoing maintenance. Please try again in a few minutes."
        )

    return out


def _process_colabfold_non_pairing(x, prefix, env_a3m_dict, uniref_a3m_dict):
    """Process non-pairing MSA for colabfold mode."""
    query_id = str(int(x.split("\n")[0].split("_")[-1]))
    query_seq = x.split("\n")[1]
    real_non_pairing_fpath = os.path.join(
        prefix.split("msa_resmsa")[0],
        "msa",
        query_id,
        "non_pairing.a3m",
    )
    output_dir = os.path.dirname(real_non_pairing_fpath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(real_non_pairing_fpath, "w", encoding="utf-8") as f:
        f.write(f">query\n{query_seq}\n")
        for k, v in env_a3m_dict.items():
            if not k.startswith("query_"):
                f.write(f">{k}\n{v}\n")
        for k, v in uniref_a3m_dict.items():
            if not k.startswith("query_"):
                f.write(f">{k}\n{v}\n")

    return os.path.abspath(os.path.dirname(real_non_pairing_fpath))


def _process_colabfold_pairing(prefix, pair_a3m):
    """Process pairing MSA for colabfold mode."""
    with open(pair_a3m, "r", encoding="utf-8") as pair_file:
        pair_a3m_chunks = pair_file.read().split("\x00")

    for chunk in pair_a3m_chunks[:-1]:
        real_pairing_fpath = os.path.join(
            prefix.split("msa_resmsa")[0],
            "msa",
            str(int(chunk.split("\n")[0].split("_")[-1])),
            "pairing.a3m",
        )
        output_dir = os.path.dirname(real_pairing_fpath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        chunk_fasta = parse_fasta_string(chunk)
        with open(real_pairing_fpath, "w", encoding="utf-8") as f:
            for i, (k, v) in enumerate(chunk_fasta.items()):
                if k.startswith("query_"):
                    f.write(f">query\n{v}\n")
                else:
                    ks = k.split("\t")
                    ks[0] = f"{ks[0]}_{i}/"
                    k = "\t".join(ks)
                    f.write(f">{k}_{i}\n{v}\n")


def _setup_mode_and_headers(use_filter, use_env, use_pairing, pairing_strategy, user_agent):
    """Setup mode and headers for MSA service."""
    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = "pairgreedy" if pairing_strategy == "greedy" else "paircomplete"

    headers = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent (e.g., "
            "'toolname/version contact@email') to help us debug in case "
            "of problems. This warning will become an error in the future."
        )

    return mode, headers


def _process_sequences(seqs):
    """Process and deduplicate sequences."""
    seqs = [seqs] if isinstance(seqs, str) else seqs
    seqs_unique = []
    for x_seq in seqs:
        if x_seq not in seqs_unique:
            seqs_unique.append(x_seq)
    return seqs, seqs_unique


def _submit_and_wait_for_job(host_url, submission_endpoint, seqs_unique, mode,
                             num_sequences, email, headers, pbar):
    """Submit MSA job and wait for completion."""
    redo = True
    out = None
    while redo:
        pbar.set_description("SUBMIT")

        # Resubmit job until it goes through
        out = _submit_msa_job(
            host_url, submission_endpoint, seqs_unique, mode, num_sequences, email, headers)
        out = _handle_job_submission_errors(
            out, seqs_unique, mode, num_sequences, host_url, submission_endpoint, email, headers)

        # wait for job to finish
        job_id = out["id"]
        pbar.set_description(out["status"])
        out = _wait_for_job_completion(
            host_url, job_id, headers, pbar)

        if out["status"] == "COMPLETE":
            pbar.n = 100
            pbar.refresh()
            redo = False
        elif out["status"] == "ERROR":
            redo = False
            raise Exception(
                "MMseqs2 API is giving errors. Please confirm your input is a "
                "valid protein sequence. If error persists, please try again "
                "an hour later."
            )

    return out["id"]


def _download_and_extract_results(job_id, tar_gz_file, host_url, headers):
    """Download and extract MSA results."""
    def download(job_id, path):
        res = _make_http_request_with_retry(
            f"{host_url}/result/download/{job_id}",
            method='get',
            headers=headers,
        )
        with open(path, "wb") as out:
            out.write(res.content)

    download(job_id, tar_gz_file)
    with tarfile.open(tar_gz_file) as tar_gz:
        tar_gz.extractall(os.path.dirname(tar_gz_file))
    return os.listdir(os.path.dirname(tar_gz_file))


def _validate_protenix_files(files):
    """Validate that required protenix files are present."""
    if (
        "0.a3m" not in files
        or "pdb70_220313_db.m8" not in files
        or "uniref_tax.m8" not in files
    ):
        raise FileNotFoundError(
            "Files 0.a3m, pdb70_220313_db.m8, and uniref_tax.m8 "
            "not found in the directory."
        )
    print("Files downloaded and extracted successfully.")


def _process_colabfold_results(x, prefix, use_pairing):
    """Process colabfold results."""
    if not use_pairing:
        env_a3m_fpath = os.path.join(
            prefix, "bfd.mgnify30.metaeuk30.smag30.a3m"
        )
        with open(env_a3m_fpath, "r", encoding="utf-8") as f:
            env_a3m_dict = parse_fasta_string(
                f.read().replace("\x00", ""))

        uniref_a3m_fpath = os.path.join(prefix, "uniref.a3m")
        with open(uniref_a3m_fpath, "r", encoding="utf-8") as f:
            uniref_a3m_dict = parse_fasta_string(
                f.read().replace("\x00", ""))

        return _process_colabfold_non_pairing(x, prefix, env_a3m_dict, uniref_a3m_dict)

    # pairing mode
    pair_a3m = os.path.join(prefix, "pair.a3m")
    _process_colabfold_pairing(prefix, pair_a3m)
    return None


def run_mmseqs2_service(
    x,
    prefix,
    use_env=True,
    use_filter=True,
    use_templates=False,
    filter_arg=None,
    use_pairing=False,
    pairing_strategy="complete",
    host_url="https://api.colabfold.com",
    user_agent: str = "",
    email: str = "",
    server_mode: str = "protenix",
) -> Tuple[List[str], List[str]]:
    """
    Run mmseqs2 service.
    """
    # pylint: disable=unused-argument
    if server_mode == "protenix":
        if host_url != "https://protenix-server.com/api/msa":
            raise ValueError("host_url must be https://protenix-server.com/api/msa")

    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    # compatibility to old option
    if filter_arg is not None:
        use_filter = filter_arg

    if use_pairing:
        use_templates = False
        use_env = False

    mode, headers = _setup_mode_and_headers(
        use_filter, use_env, use_pairing, pairing_strategy, user_agent
    )

    # process input x
    _, seqs_unique = _process_sequences(x)

    # define path
    path = prefix
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    num_sequences = 101

    # lets do it!
    logger.info("Msa server is running.")
    if not os.path.isfile(tar_gz_file):
        time_estimate = 100
        with tqdm(total=time_estimate, bar_format=TQDM_BAR_FORMAT) as pbar:
            job_id = _submit_and_wait_for_job(
                host_url, submission_endpoint, seqs_unique, mode,
                num_sequences, email, headers, pbar
            )

            # Download results
            files = _download_and_extract_results(
                job_id, tar_gz_file, host_url, headers
            )

            if server_mode == "protenix":
                _validate_protenix_files(files)
            elif server_mode == "colabfold":
                return _process_colabfold_results(x, prefix, use_pairing)
    return None
