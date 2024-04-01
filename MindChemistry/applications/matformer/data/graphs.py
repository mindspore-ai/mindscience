# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Module to generate networkx graphs.Implementation based on the template of ALIGNN."""

from collections import defaultdict
from typing import Optional
import numpy as np
from tqdm import tqdm
from jarvis.core.specie import chem_data, get_node_attributes


# pylint: disable=W0102

# Structure dataset
class StructureDataset():
    """Dataset of crystal DGLGraphs."""

    def __init__(
            self,
            df,
            graphs,
            target,
            atom_features="atomic_number",
            transform=None,
            line_graph=False,
            classification=False,
            id_tag="jid",
            mean_train=None,
            std_train=None,
    ):
        """Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.df = df
        self.graphs = graphs
        self.target = target
        self.line_graph = line_graph

        self.ids = self.df[id_tag]
        self.atoms = self.df['atoms']

        self.labels = np.array(self.df[target])
        if mean_train is None:
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
        else:
            self.labels = (self.labels - mean_train) / std_train
        self.transform = transform
        features = self._get_attribute_lookup(atom_features)
        graphs_map = {"x": 0, "edge_index": 1, "edge_attr": 2}

        for g in graphs:
            z = g[graphs_map["x"]]
            g.append(z)
            z = np.squeeze(z).astype(np.int32)
            f = features[z].astype(np.float32)
            if g[graphs_map["x"]].shape[0] == 1:
                f = np.expand_dims(f, 0)
            g[graphs_map["x"]] = f
        if line_graph:
            self.graphs = []
            for g in tqdm(graphs):
                self.graphs.append(g)
            self.line_graphs = self.graphs

        if classification:
            self.labels = self.labels.view(-1).long()

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            return g, self.line_graphs[idx], label, label

        return g, label


def canonize_edge(
        src_id,
        dst_id,
        src_image,
        dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_submit(
        atoms=None,
        cutoff=8,
        max_neighbors=12,
        id_num=None,
        use_canonize=False,
        use_lattice=False,
):
    """Construct k-NN edge list."""
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id_num=id_num,
        )
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, _, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))
        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 1, 0])))
    return edges


def build_undirected_edgedata(
        atoms=None,
        edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)

    u = np.array(u)
    v = np.array(v)
    r = np.array(r, dtype=np.float32)

    return u, v, r


class StructureGraph():
    """Generate a graph object."""

    def __init__(
            self,
            nodes=[],
            node_attributes=[],
            edges=[],
            edge_attributes=[],
            color_map=None,
            labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
            atoms=None,
            cutoff=8.0,
            max_neighbors=12,
            atom_features="cgcnn",
            id_num: Optional[str] = None,
            use_canonize: bool = False,
            use_lattice: bool = False,
    ):
        """
        get dgl multigraph
        """
        edges = nearest_neighbor_edges_submit(
            atoms=atoms,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            id_num=id_num,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
        )

        u, v, r = build_undirected_edgedata(atoms, edges)

        # build up atom attribute tensor
        sps_features = []
        for _, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features.append(feat)
        sps_features = np.array(sps_features)

        node_features = np.array(sps_features, np.float32)

        edge_index = np.concatenate((np.expand_dims(u, 0), np.expand_dims(v, 0)), axis=0)
        g = [node_features, edge_index, r]

        return g
