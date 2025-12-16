#!/usr/bin/env python3

import argparse
import h5py
import numpy as np

# HDF5 Filter IDs (Common ones)
# 32001: Bitshuffle
# 32008: LZF
# 1: Szip
# 2: GZIP

def read_frame_and_props(filename, frame):
    """
    Reads a single frame as a 3D array (1, Y, X), extracts data, 
    and robustly checks ALL creation properties.
    """
    with h5py.File(filename, "r") as f:
        original_dset = f["entry/data/data"]
        
        # 1. Extract the frame data
        # Slicing 'frame:frame+1' preserves the leading dimension,
        # resulting in a 3D array of shape (1, Y, X).
        img = original_dset[frame:frame+1, :, :]
        
        # 2. Extract basic creation properties
        dset_props = {
            'dtype': original_dset.dtype,
            'chunks': original_dset.chunks,
            'fletcher32': original_dset.fletcher32,
        }
        
        dset_props = {k: v for k, v in dset_props.items() if v is not None}

        # --- ROBUST FILTER CHECK START ---
        
        dcpl = original_dset.id.get_create_plist()
        
        new_dset_props = {
            'dtype': dset_props['dtype'],
            'chunks': dset_props.get('chunks'),
            'fletcher32': dset_props.get('fletcher32')
        }

        active_filters = []
        
        try:
            nfilters = dcpl.get_nfilters()
        except AttributeError:
            nfilters = 0
            print("WARNING: Could not determine filter count. Assuming zero filters.")
        
        
        for i in range(nfilters):
            # (filter_id, flags, num_params, params)
            filter_id, flags, num_params, params = dcpl.get_filter(i)
            
            # Decode the filter and set the h5py creation properties
            if filter_id == 1: # Szip
                new_dset_props['compression'] = 'szip'
                new_dset_props['compression_opts'] = (params[0], params[1])
                active_filters.append('Szip')
            elif filter_id == 2: # GZIP (zlib)
                new_dset_props['compression'] = 'gzip'
                new_dset_props['compression_opts'] = params[0]
                active_filters.append('GZIP')
            elif filter_id == 32001: # Bitshuffle (custom filter)
                new_dset_props['shuffle'] = True
                active_filters.append('Bitshuffle')
            elif filter_id == 32008: # LZF (custom filter)
                new_dset_props['compression'] = 'lzf'
                active_filters.append('LZF')

        if new_dset_props.get('chunks'):
            original_chunks = new_dset_props['chunks']
            if len(original_chunks) == 3:
                print(f"INFO: Kept 3D chunking {original_chunks} for the new 3D dataset.")
            else:
                del new_dset_props['chunks']
                print("WARNING: Original chunk shape was not 3D. Removed 'chunks' property.")
        
        # --- REPORT FILTERS ---
        if active_filters:
            print(f"INFO: Identified active filters: {', '.join(active_filters)}")
        else:
            print("INFO: No active filters identified in the source dataset.")

        final_dset_props = {k: v for k, v in new_dset_props.items() if v is not None}
        
        return img, final_dset_props

def write_frame(filename, img, dset_props):
    with h5py.File(filename, "w") as f:
        entry = f.create_group("entry")
        data_grp = entry.create_group("data")
        data_grp.create_dataset("data", data=img, **dset_props)

def main():
    parser = argparse.ArgumentParser(description="Extract one frame from an HDF5 file")
    parser.add_argument("input", type=str, help="Input HDF5 file")
    parser.add_argument("frame", type=int, help="Frame index (0-based)")
    parser.add_argument("-o", "--output", type=str, default="oneframe.h5",
                        help="Output HDF5 file (default: oneframe.h5)")
    args = parser.parse_args()

    img, dset_props = read_frame_and_props(args.input, args.frame)
    
    has_filters = 'compression' in dset_props or 'shuffle' in dset_props
    if has_filters:
        print(f"SUCCESS: Compression/Shuffling properties identified and will be applied: {dset_props}")
    else:
        print("Warning: No compression or shuffling properties were identified after robust check. Writing uncompressed.")

    write_frame(args.output, img, dset_props)

    print(f"Wrote frame {args.frame} to {args.output} at entry/data/data")

    # --- VERIFICATION STEP START ---
    print(f"\n--- Verifying Output File {args.output} ---")
    
    try:
        with h5py.File(args.output, "r") as f_out:
            dset_out = f_out["entry/data/data"]
            
            out_compression = dset_out.compression if dset_out.compression else "None"
            out_shuffle = dset_out.shuffle
            
            dcpl_out = dset_out.id.get_create_plist()
            out_filter_ids = []
            
            for i in range(dcpl_out.get_nfilters()):
                # The verification step also expects 4 values from get_filter(i)
                out_filter_ids.append(dcpl_out.get_filter(i)[0])
            
            print(f"* Output Dataset Shape: {dset_out.shape}")
            print(f"* Output Compression: {out_compression}")
            print(f"* Output Shuffle (Bitshuffle): {out_shuffle}")
            print(f"* Output Filter IDs Found: {out_filter_ids}")

    except Exception as e:
        print(f"ERROR during verification: {e}")

if __name__ == "__main__":
    main()
