import h5py
import numpy as np
import argparse
import os

# --- Configuration ---
# Bitshuffle requires another compressor (e.g., LZF or ZSTD) to function
DEFAULT_COMPRESSOR = 'lzf'
# For ZSTD (ID 32015) compression level: 1-22. 9 is a good balance.
ZSTD_COMPRESSION_LEVEL = 9

def recompress_mask_file(input_filename, output_filename, dataset_path, compression_type):
    """
    Reads data from an existing HDF5 file and writes it to a new file 
    with Bitshuffle and the specified compression applied.
    """
    print(f"Reading data from: {input_filename} at path: {dataset_path}")
    
    # 1. READ DATA AND METADATA
    try:
        with h5py.File(input_filename, 'r') as f_in:
            if dataset_path not in f_in:
                print(f"ERROR: Dataset path '{dataset_path}' not found in {input_filename}")
                return
                
            original_dset = f_in[dataset_path]
            
            # Read the data (full read of the uncompressed data)
            data = original_dset[()]
            
            # Get original shape and dtype for chunks and output type
            original_shape = data.shape
            original_dtype = data.dtype
            
            # Use the full data shape as the chunk size for single-dataset optimization
            chunk_shape = original_shape
            
    except Exception as e:
        print(f"ERROR reading input file: {e}")
        return

    # 2. CONFIGURE COMPRESSION
    compression_kwargs = {
        'chunks': chunk_shape,           # Chunk the entire dataset
        'shuffle': True,                 # **Activates Bitshuffle (Filter 32001)**
        'fletcher32': True,              # Optional: Add checksum for integrity check
        'dtype': original_dtype          # Maintain original data type
    }
    
    if compression_type == 'lzf':
        # LZF (Filter 32008) - Fast and effective
        compression_kwargs['compression'] = 'lzf'
        print("Applying Bitshuffle + LZF compression.")
    elif compression_type == 'zstd':
        # ZSTD (Filter 32015) - Best compression ratio, slightly slower
        # Requires the hdf5plugin package
        compression_kwargs['compression'] = 32015
        compression_kwargs['compression_opts'] = ZSTD_COMPRESSION_LEVEL
        print(f"Applying Bitshuffle + ZSTD (level {ZSTD_COMPRESSION_LEVEL}) compression.")
    else:
        # Fallback to Bitshuffle only (rarely recommended)
        print("Applying Bitshuffle only (no secondary compression).")

    # 3. WRITE MAXIMALLY COMPRESSED DATA
    print(f"Writing compressed data to: {output_filename}")
    
    try:
        with h5py.File(output_filename, 'w') as f_out:
            # Create necessary groups (assuming /data/data standard for masks)
            f_out.create_group(os.path.dirname(dataset_path))
            
            dset_out = f_out.create_dataset(dataset_path, data=data, **compression_kwargs)
            
            # --- Verification and Report ---
            size_uncompressed_bytes = np.prod(data.shape) * data.dtype.itemsize
            size_compressed_bytes = dset_out.size * dset_out.dtype.itemsize
            
            print("\n--- Compression Success ---")
            print(f"Output Dataset: {dset_out.name}")
            print(f"Shape: {dset_out.shape}")
            print(f"Dtype: {dset_out.dtype}")
            print(f"Bitshuffle Active (shuffle): {dset_out.shuffle}")
            print(f"Compression Active (compression): {dset_out.compression if dset_out.compression else 'None'}")
            print(f"Uncompressed Data Size: {size_uncompressed_bytes / (1024**2):.2f} MiB")
            print(f"Compressed HDF5 File Size (Actual): {os.path.getsize(output_filename) / (1024**2):.2f} MiB")
            
    except Exception as e:
        print(f"\nERROR writing compressed file: {e}")
        print("CRITICAL CHECK: If you see 'filter not registered', you MUST install hdf5plugin:")
        print("               pip install hdf5plugin")


def main():
    parser = argparse.ArgumentParser(description="Recompress an existing HDF5 mask file using Bitshuffle and LZF/ZSTD.")
    parser.add_argument("input_file", type=str, help="Path to the existing uncompressed HDF5 file.")
    parser.add_argument("-o", "--output_file", type=str, default="compressed_mask.h5",
                        help="Name of the new, compressed output HDF5 file (default: compressed_mask.h5).")
    parser.add_argument("-p", "--path", type=str, default="/data/data",
                        help="Internal HDF5 path to the mask dataset (default: /data/data).")
    parser.add_argument("-c", "--compressor", type=str, default=DEFAULT_COMPRESSOR,
                        choices=['lzf', 'zstd', 'none'],
                        help="Secondary compressor to use: 'lzf' (fast) or 'zstd' (max ratio). Default: lzf.")
    
    args = parser.parse_args()

    # ZSTD requires hdf5plugin, so check for it if selected
    if args.compressor == 'zstd' and not h5py.h5z.filter_avail(32015):
        print("WARNING: ZSTD selected but filter 32015 (ZSTD) is not available.")
        print("Please run 'pip install hdf5plugin'. Falling back to LZF.")
        args.compressor = 'lzf'

    recompress_mask_file(args.input_file, args.output_file, args.path, args.compressor)

if __name__ == "__main__":
    main()
