# maskmaker
A graphical interface for making an HDF5 pixel mask using CrystFEL geometry

```
$ python maskMakerGUI.py -h
usage: maskMakerGUI.py [-h] [-g GEOMETRY] [-m MASK] [-mp MASK_H5PATH]
                       image_filename h5path

maskmake - mask making, but with a mouse!

positional arguments:
  image_filename        filename for the hdf5 cspad image file
  h5path                hdf5 path for the 2D cspad data

optional arguments:
  -h, --help            show this help message and exit
  -g GEOMETRY, --geometry GEOMETRY
                        path to the CrystFEL geometry file for the image
  -m MASK, --mask MASK  path to the h5file of the starting mask
  -mp MASK_H5PATH, --mask_h5path MASK_H5PATH
                        path inside the h5file of the starting mask
```

### Requires
- python (>= 3.8)
- h5py (>= 3.0)
- pyqtgraph
- scipy
- numpy
- hdf5plugin (for mask compression using Bitshuffle/LZF/ZSTD)


### It is highly recommended to install the dependencies using pip:
```bash
pip install numpy scipy pyqtgraph h5py hdf5plugin

### Also included are two scripts: 
1. repack_mask.py to compress your mask.h5 files created earlier 
(output from maskmaker should be well compressed)
2. export_oneframe_frommultiframeH5.py  to grab a single frame from e.g. a multi-frame HDF5 file from EIGER detectors, for use with maskmaker. So you don't have to copy massive multi-frame HDF5s.

### Example
```
$ git clone https://github.com/nzatsepin/maskmaker.git
$ cd maskmaker
$ ./maskMakerGUI.py example/LA93-r0014-CxiDs1-darkcal.h5 data/data -g example/cspad-cxib2313-v9.geom
```

### Thanks to Andrew Morgan who wrote the original CsPadMaskMaker and Kenneth Beyerlein for keeping the git repo alive so I could rediscover it in 2025!
