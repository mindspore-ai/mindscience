# Copyright 2022 Huawei Technologies Co., Ltd
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
Computes and saves molecular features for a dataset.
"""
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from .src.util.utils import makedirs, load_features, save_features, load_smiles_labels
from .src.data.molfeaturegenerator import get_features_generator


class SaveFeatures:
    """SaveFeatures"""
    def __init__(self):
        pass

    def load_temp(self, temp_dir):
        """
        Loads all features saved as .npz files in load_dir.

        Assumes temporary files are named in order 0.npz, 1.npz, ...

        :param temp_dir: Directory in which temporary .npz files containing features are stored.
        :return: A tuple with a list of molecule features, where each molecule's features is a list of floats,
        and the number of temporary files.
        """
        features = []
        temp_num = 0
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

        while os.path.exists(temp_path):
            features.extend(load_features(temp_path))
            temp_num += 1
            temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

        return features, temp_num

    def generate_and_save_features(self, data_path, features_generator, save_path, save_frequency, restart, sequential):
        """
        Computes and saves features for a dataset of molecules as a 2D array in a .npz file.

        :param args: Arguments.
        """
        # Create directory for save_path
        makedirs(save_path, isfile=True)

        # Get data and features function
        mols, _ = load_smiles_labels(data_path)
        features_generator = get_features_generator(features_generator)
        temp_save_dir = save_path + '_temp'

        # Load partially complete data
        if restart:
            if os.path.exists(save_path):
                os.remove(save_path)
            if os.path.exists(temp_save_dir):
                shutil.rmtree(temp_save_dir)
        else:
            if os.path.exists(save_path):
                raise ValueError(f'"{save_path}" already exists and args.restart is False.')

            if os.path.exists(temp_save_dir):
                features, temp_num = self.load_temp(temp_save_dir)

        if not os.path.exists(temp_save_dir):
            makedirs(temp_save_dir)
            features, temp_num = [], 0

        # Build features map function
        mols = mols[len(features):]  # restrict to data for which features have not been computed yet

        if sequential:
            features_map = map(features_generator, mols)
        else:
            features_map = Pool(30).imap(features_generator, mols)

        # Get features
        temp_features = []
        for i, feats in tqdm(enumerate(features_map), total=len(mols)):
            temp_features.append(feats)

            # Save temporary features every save_frequency
            if (i > 0 and (i + 1) % save_frequency == 0) or i == len(mols) - 1:
                save_features(os.path.join(temp_save_dir, f'{temp_num}.npz'), temp_features)
                features.extend(temp_features)
                temp_features = []
                temp_num += 1

        try:
            # Save all features
            save_features(save_path, features)

            # Remove temporary features
            shutil.rmtree(temp_save_dir)
        except OverflowError:
            print('Features array is too large to save as a single file.'
                  'Instead keeping features as a directory of files.')
