"""
This **module** gives functions and classes to use MDAnalysis to analyze the trajectories
"""
import os.path
import numpy as np
from ..helper import Xopen, set_global_alternative_names
try:
    from MDAnalysis.coordinates import base
    from MDAnalysis.lib import util
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "'MDAnalysis' package needed. Maybe you need 'pip install MDAnalysis'") from exc


class _SpongeTrajectoryReader(base.ReaderBase):
    """
    This **class** is the interface to MDAnalysis.

    .. TIP::

        You do not need to call the class yourself. Use the function set_universe_sponge_trajectory.

    :param dat_file_name: the name of the SPONGE dat trajectory file
    :param box: the name of the box file or a list of 6 ``int`` or ``float`` \
representing the 3 box lengths and 3 box angles.
    :param n_atoms: the number of atoms
    """
    def __init__(self, dat_file_name, box, n_atoms, **kwargs):
        super().__init__(dat_file_name, **kwargs)
        if isinstance(box, str):
            self.boxname = box
            self.box = None
            self._get_box_offset()
        else:
            self.boxname = None
            self.box = box
        self._n_atoms = n_atoms
        self._n_frames = os.path.getsize(dat_file_name) // 12 // self.n_atoms
        self.trajfile = None
        self.boxfile = None
        self.ts = base.Timestep(self.n_atoms, **self._ts_kwargs)
        self._read_next_timestep()

    @property
    def n_frames(self):
        """
        The total number of frames in the trajectory file
        """
        return self._n_frames

    @property
    def n_atoms(self):
        """
        The total number of atoms in the trajectory file
        """
        return self._n_atoms

    def close(self):
        """
        Close all the opened file

        :return: None
        """
        if self.trajfile is not None:
            self.trajfile.close()
            self.trajfile = None
        if self.boxfile is not None:
            self.boxfile.close()
            self.boxfile = None

    def open_trajectory(self):
        """
        Open the trajectory file

        :return: trajectory file and box file
        """
        self.trajfile = util.anyopen(self.filename, "rb")
        if self.box is None:
            self.boxfile = util.anyopen(self.boxname)
        ts = self.ts
        ts.frame = -1
        return self.trajfile, self.boxfile

    def _reopen(self):
        self.close()
        self.open_trajectory()

    def _read_frame(self, frame):
        """

        :param frame:
        :return:
        """
        if self.trajfile is None:
            self.open_trajectory()
        if self.boxfile is not None:
            self.boxfile.seek(self._offsets[frame])
        self.trajfile.seek(self.n_atoms * 12 * frame)
        self.ts.frame = frame - 1
        return self._read_next_timestep()

    def _read_next_timestep(self):
        """

        :return:
        """
        ts = self.ts
        if self.trajfile is None:
            self.open_trajectory()
        t = self.trajfile.read(12 * self.n_atoms)
        if not t:
            raise EOFError
        setattr(ts, "_pos", np.frombuffer(t, dtype=np.float32).reshape(self.n_atoms, 3))

        if self.box is not None:
            ts.dimensions = self.box
        else:
            ts.dimensions = list(map(float, self.boxfile.readline().split()))
        ts.frame += 1
        return ts

    def _get_box_offset(self):
        """

        :return:
        """
        self._offsets = [0]
        with util.openany(self.boxname) as f:
            line = f.readline()
            while line:
                self._offsets.append(f.tell())
                line = f.readline()
        self._offsets.pop()


def set_universe_sponge_trajectory(universe, trajname, box, **kwargs):
    """
    This **function** sets the trajectory to an MDAnalysis.Universe instance

    :param universe: the MDAnalysis.Universe object
    :param trajname: the name of the trajectory file
    :param box: the name of the box file or a list of 6 ``int`` or ``float`` \
representing the 3 box lengths and 3 box angles.
    :return: None
    """
    setattr(universe, "_trajectory", _SpongeTrajectoryReader(trajname, box, len(u.atoms), **kwargs))


class TrajWriter():
    """
    This **class** is used to write the SPONGE trajectory (xxx.dat and xxx.box)

    usage example::

        import Xponge.analysis.md_analysis as xmda
        import MDAnalysis as mda
        from MDAnalysis.tests.datafiles import PDB, XTC

        u = mda.Universe(PDB, XTC)

        with xmda.TrajWriter("mda_test") as W:
            for ts in u.trajectory:
                W.write(u)

    :param prefix: the prefix of the output files
    """
    def __init__(self, prefix):
        self.datname = prefix + ".dat"
        self.boxname = prefix + ".box"
        self.datfile = None
        self.boxfile = None
        set_attribute_alternative_name(self, self.open)
        set_attribute_alternative_name(self, self.close)
        set_attribute_alternative_name(self, self.write)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """
        This **function** opens the trajectory files

        :return: None
        """
        self.datfile = Xopen(self.datname, "wb")
        self.boxfile = Xopen(self.boxname, "w")

    def close(self):
        """
        This **function** closes the trajectory files

        :return: None
        """
        self.datfile.close()
        self.boxfile.close()

    def write(self, u):
        """
        This **function** writes the coordinates of the Universe to the output files

        :param u: an MDAnalysis.Universe instance
        :return: None
        """
        self.datfile.write(u.coord.positions.astype(np.float32).tobytes())
        self.boxfile.write(" ".join([f"{i}" for i in u.coord.dimensions]) + "\n")

set_global_alternative_names(globals())
