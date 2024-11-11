import os

import h5py


def print_hdf5(
    h5py_obj, level=-1, print_full_name: bool = False, print_attrs: bool = True
) -> None:
    """Prints the name and shape of datasets in a H5py HDF5 file.

    Parameters
    ----------
    h5py_obj: [h5py.File, h5py.Group]
        the h5py.File or h5py.Group object
    level: int
        What level of the file tree you are in
    print_full_name
        If True, the full tree will be printed as the name, e.g. /group0/group1/group2/dataset: ...
        If False, only the current node will be printed, e.g. dataset:
    print_attrs
        If True: print all attributes in the file
    Returns
    -------
    None

    """

    def is_group(f):
        return type(f) == h5py._hl.group.Group

    def is_dataset(f):
        return type(f) == h5py._hl.dataset.Dataset

    def print_level(level, n_spaces=5) -> str:
        if level == -1:
            return ""
        prepend = "|" + " " * (n_spaces - 1)
        prepend *= level
        tree = "|" + "-" * (n_spaces - 2) + " "
        return prepend + tree

    for key in h5py_obj.keys():
        entry = h5py_obj[key]
        name = entry.name if print_full_name else os.path.basename(entry.name)
        if is_group(entry):
            print(f"{print_level(level)}{name}")
            print_hdf5(entry, level + 1, print_full_name=print_full_name)
        elif is_dataset(entry):
            shape = entry.shape
            dtype = entry.dtype
            print(f"{print_level(level)}{name}: {shape} {dtype} {entry[()]}")
    if level == -1:
        if print_attrs:
            print("attrs: ")
            for key, value in h5py_obj.attrs.items():
                print(f" {key}: {value}")
