import argparse

import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    a = p.parse_args()
    with h5py.File(a.path, "r") as f:
        print("attributes", dict(f.attrs))
        f.visititems(lambda name, obj: print(name, getattr(obj, "shape", "group"), getattr(obj, "dtype", "")))


if __name__ == "__main__":
    main()
