import h5py
import numpy as np

def print_hdf5_structure(file_path, max_items=10, indent=0):
    """
    Print the structure of an HDF5 file, showing at most max_items children per group.
    The first Dataset encountered prints its actual data; other Datasets only print shape/dtype.
    Attributes are printed only once per file.
    """

    attributes_printed = {"done": False}
    first_dataset_printed = {"done": False}

    def _print(name, obj, indent):
        space = "  " * indent

        if isinstance(obj, h5py.Group):
            print(f"{space}Group: {name}/")

            # Print attributes only once
            if (not attributes_printed["done"]) and len(obj.attrs) > 0:
                print(f"{space}  Attributes:")
                for key, val in obj.attrs.items():
                    print(f"{space}    {key}: {val}")
                attributes_printed["done"] = True

            keys = list(obj.keys())
            total = len(keys)

            for k in keys[:max_items]:
                _print(k, obj[k], indent + 1)

            if total > max_items:
                print(f"{space}  ... ({total - max_items} more not shown)")

        elif isinstance(obj, h5py.Dataset):
            print(f"{space}Dataset: {name}  shape={obj.shape}  dtype={obj.dtype}")

            # Print attributes only once
            if (not attributes_printed["done"]) and len(obj.attrs) > 0:
                print(f"{space}  Attributes:")
                for key, val in obj.attrs.items():
                    print(f"{space}    {key}: {val}")
                attributes_printed["done"] = True

            # Print the actual data of the first dataset
            if not first_dataset_printed["done"]:
                data = obj[()]  # Read the entire dataset
                print(f"{space}  Data (first 10 rows):")
                print(np.array2string(data[:10], separator=', '))
                first_dataset_printed["done"] = True

    with h5py.File(file_path, "r") as f:
        print("HDF5 file structure:")
        _print("/", f, indent)

print_hdf5_structure("D:/download/chunk2/chunk2.hdf5", max_items=10)

import h5py

file_path = "D:/download/chunk2/chunk2.hdf5"
f = h5py.File(file_path, "r")  # Open the file in read-only mode by default
print(list(f.keys()))  # Groups at the root level, e.g. ['data']
print(list(f['data'].keys())[:10])  # First 10 dataset names under 'data'

dataset_name = "109C.TA_20070109140205_EV"
data = f['data'][dataset_name][()]  # Returns a numpy array
print(data.shape)  # (6000, 3)
ds = f['data'][dataset_name]
print(ds.attrs.keys())  # All attribute names
print(ds.attrs['back_azimuth_deg'])  # Read a specific attribute

import numpy as np
import matplotlib.pyplot as plt

# Assume data is your ndarray with shape = (6000, 3)
# data = f['data'][dataset_name][()]

# Simulated data
# data = np.random.randn(6000, 3)

# Create the x axis
x = np.arange(data.shape[0])  # 0, 1, 2, ..., 5999

# Plot three lines
plt.figure(figsize=(12, 6))
plt.plot(x, data[:, 0], label='Channel 1')
plt.plot(x, data[:, 1], label='Channel 2')
plt.plot(x, data[:, 2], label='Channel 3')

# Add title and labels
plt.title("Three-channel Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.show()
