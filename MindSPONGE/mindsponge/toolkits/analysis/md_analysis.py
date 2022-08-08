"""
This **module** gives functions and classes to use MDAnalysis to analyze the trajectories
"""
import os.path
import numpy as np
from ..helper import Xopen, set_global_alternative_names, set_attribute_alternative_name
try:
    from MDAnalysis.coordinates import base
    from MDAnalysis.lib import util
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "'MDAnalysis' package needed. Maybe you need 'pip install MDAnalysis'") from exc


class SpongeTrajectoryReader(base.ReaderBase):
    """
    This **class** is the interface to MDAnalysis.

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


class SpongeTrajectoryWriter():
    """
    This **class** is used to write the SPONGE trajectory (xxx.dat and xxx.box)

    usage example::

        import Xponge.analysis.md_analysis as xmda
        import MDAnalysis as mda
        from MDAnalysis.tests.datafiles import PDB, XTC

        u = mda.Universe(PDB, XTC)

        with xmda.SpongeTrajectoryWriter("mda_test") as W:
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


class SpongeCoordinateReader(base.ReaderBase):
    """
    This **class** is the interface to MDAnalysis.

    :param file_name: the name of the SPONGE coordinate trajectory file
    """
    def __init__(self, file_name, n_atoms, **kwargs):
        super().__init__(file_name, **kwargs)
        self._n_atoms = n_atoms
        self._n_frames = 1
        self.file = None
        self.start = 0
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
        if self.file is not None:
            self.file.close()
            self.file = None

    def open_file(self):
        """
        Open the coordinate file

        :return: trajectory file and box file
        """
        self.file = util.anyopen(self.filename, "r")
        self.file.readline()
        self.start = self.file.tell()

    def _reopen(self):
        self.close()
        self.open_file()

    def _read_frame(self, frame):
        """

        :param frame:
        :return:
        """
        if self.file is None:
            self.open_file()
        self.file.seek(self.start)
        self.ts.frame = frame - 1
        return self._read_next_timestep()

    def _read_next_timestep(self):
        """

        :return:
        """
        ts = self.ts
        if self.file is None:
            self.open_file()
        if self.file.tell() != self.start:
            raise EOFError
        t = np.loadtxt(self.file, max_rows=self.n_atoms)
        setattr(ts, "_pos", t)
        box = list(map(float, self.file.readline().split()))
        ts.dimensions = box
        ts.frame += 1
        return ts


class SpongeCoordinateWriter():
    """
    This **class** is used to write the SPONGE coordinate file

    usage example::

        import Xponge.analysis.md_analysis as xmda
        import MDAnalysis as mda
        from MDAnalysis.tests.datafiles import PDB, XTC

        u = mda.Universe(PDB, XTC)

        with xmda.SpongeCoordinateWriter("mda_test") as W:
            for ts in u.trajectory:
                W.write(u)

    :param file_name: the name of the output file
    :param n_atoms: the total number of atoms this Timestep describes
    """
    def __init__(self, file_name, n_atoms=None):
        self.filename = file_name
        self.file = None
        self.n_atoms = n_atoms
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
        This **function** opens the coordinate file

        :return: None
        """
        self.file = Xopen(self.filename, "w")

    def close(self):
        """
        This **function** closes the coordinate file

        :return: None
        """
        self.file.close()

    def write(self, u):
        """
        This **function** writes the coordinates of the Universe to the output files

        :param u: an MDAnalysis.Universe instance
        :return: None
        """
        if self.n_atoms is None:
            self.n_atoms = len(u.coord.positions)
        towrite = f"{self.n_atoms}\n"
        for crd in u.coord.positions[:self.n_atoms]:
            towrite += f"{crd[0]} {crd[1]} {crd[2]}\n"
        if u.coord.dimensions:
            towrite += " ".join([f"{i}" for i in u.coord.dimensions]) + "\n"
        else:
            towrite += "999 999 999 90 90 90\n"
        self.file.write(towrite)


set_global_alternative_names()
