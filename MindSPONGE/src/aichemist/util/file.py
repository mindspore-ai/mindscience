# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
file
"""

import os
import struct
import logging
import gzip
import shutil
import zipfile
import tarfile
import hashlib
from six.moves.urllib.request import urlretrieve


logger = logging.getLogger(__name__)


def download(url, path, save_file=None, md5=None):
    """
    Download a file from the specified url.
    Skip the downloading step if there exists a file satisfying the given MD5.

    Parameters:
        url (str): URL to download
        path (str): path to store the downloaded file
        save_file (str, optional): name of save file. If not specified, infer the file name from the URL.
        md5 (str, optional): MD5 of the file
    """
    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if not os.path.exists(save_file) or compute_md5(save_file) != md5:
        logger.info("Downloading %s to %s", url, save_file)
        urlretrieve(url, save_file)
    return save_file


def extract(zip_file, member=None):
    """
    Extract files from a zip file. Currently, ``zip``, ``gz``, ``tar.gz``, ``tar`` file types are supported.

    Parameters:
        zip_file (str): file name
        member (str, optional): extract specific member from the zip file.
            If not specified, extract all members.
    """
    zip_name, extension = os.path.splitext(zip_file)
    if zip_name.endswith(".tar"):
        extension = ".tar" + extension
        zip_name = zip_name[:-4]
    save_path = os.path.dirname(zip_file)

    if extension == ".gz":
        save_files = _extract_gz(zip_file, save_path, member=member)
    elif extension in [".tar.gz", ".tgz", ".tar"]:
        save_files = _extract_tgz(zip_file, save_path, member=member)
    elif extension == ".zip":
        save_files = _extract_zip(zip_file, save_path, member=member)
    else:
        raise ValueError(f"Unknown file extension `{extension}`")

    if len(save_files) == 1:
        return save_files[0]
    return save_path


def _extract_zip(zip_file, save_path, member=None):
    """extract_zip"""
    with zipfile.ZipFile(zip_file) as zipped:
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
            logger.info("Extracting %s from %s to %s", member, zip_file, save_files[0])
        else:
            members = zipped.namelist()
            save_files = [os.path.join(save_path, m) for m in members]
            logger.info("Extracting %s to %s", zip_file, save_path)
        for member_, save_file in zip(members, save_files):
            if zipped.getinfo(member_).is_dir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or zipped.getinfo(member_).file_size != os.path.getsize(save_file):
                with zipped.open(member_, "r") as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    return save_files


def _extract_tgz(zip_file, save_path, member=None):
    """extract tgz"""
    with tarfile.open(zip_file, "r") as tar:
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
            logger.info("Extracting %s from %s to %s", member, zip_file, save_files[0])
        else:
            members = tar.getnames()
            save_files = [os.path.join(save_path, m) for m in members]
            logger.info("Extracting %s to %s", zip_file, save_path)
        for member_, save_file in zip(members, save_files):
            if tar.getmember(member_).isdir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or tar.getmember(member_).size != os.path.getsize(save_file):
                with tar.extractfile(member_) as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    return save_files


def _extract_gz(zip_file, save_path, member=None):
    """extract gz"""
    save_files = [os.path.join(save_path, member)]
    for save_file in save_files:
        with open(zip_file, "rb") as fin:
            fin.seek(-4, 2)
            file_size = struct.unpack("<I", fin.read())[0]
        with gzip.open(zip_file, "rb") as fin:
            if not os.path.exists(save_file) or file_size != os.path.getsize(save_file):
                logger.info("Extracting %s to %s", zip_file, save_file)
                with open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    return save_files


def compute_md5(file_name, chunk_size=65536):
    """
    Compute MD5 of the file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """
    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


def get_line_count(file_name, chunk_size=8192*1024):
    """
    Get the number of lines in a file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """
    count = 0
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            count += chunk.count(b"\n")
            chunk = fin.read(chunk_size)
    return count