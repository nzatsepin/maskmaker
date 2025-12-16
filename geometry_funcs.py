from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

cspad_psana_shape = (4, 8, 185, 388)
cspad_geom_shape  = (1480, 1552)


def pixel_maps_from_geometry_file(fnam):
    """
    Return pixel and radius maps from the geometry file
    
    Input: geometry filename
    
    Output: x: slab-like pixel map with x coordinate of each slab pixel in the reference system of the detector
            y: slab-like pixel map with y coordinate of each slab pixel in the reference system of the detector
            z: slab-like pixel map with distance of each pixel from the center of the reference system.

    Note:
            ss || y
            fs || x
            
            vectors should be given by [x, y]
    """
    f = open(fnam, 'r')
    f_lines = []
    for line in f:
        f_lines.append(line)

    keyword_list = ['min_fs', 'min_ss', 'max_fs', 'max_ss', 'fs', 'ss', 'corner_x', 'corner_y']

    detector_dict = {}

    panel_lines = [ x for x in f_lines if '/' in x ]

    for pline in panel_lines:
        items = pline.split('=')[0].split('/')
        if len(items) == 2 :
            panel = items[0].strip()
            property = items[1].strip()
            if property in keyword_list:
                if panel not in detector_dict.keys():
                    detector_dict[panel] = {}
                detector_dict[panel][property] = pline.split('=')[1].split(';')[0].strip()

    parsed_detector_dict = {}

    for p in detector_dict.keys():

        parsed_detector_dict[p] = {}

        parsed_detector_dict[p]['min_fs'] = int( detector_dict[p]['min_fs'] )
        parsed_detector_dict[p]['max_fs'] = int( detector_dict[p]['max_fs'] )
        parsed_detector_dict[p]['min_ss'] = int( detector_dict[p]['min_ss'] )
        parsed_detector_dict[p]['max_ss'] = int( detector_dict[p]['max_ss'] )
        parsed_detector_dict[p]['fs'] = []
        parsed_detector_dict[p]['fs'].append( float( detector_dict[p]['fs'].split('x')[0] ) )
        parsed_detector_dict[p]['fs'].append( float( detector_dict[p]['fs'].split('x')[1].split('y')[0] ) )
        parsed_detector_dict[p]['ss'] = []
        parsed_detector_dict[p]['ss'].append( float( detector_dict[p]['ss'].split('x')[0] ) )
        parsed_detector_dict[p]['ss'].append( float( detector_dict[p]['ss'].split('x')[1].split('y')[0] ) )
        parsed_detector_dict[p]['corner_x'] = float( detector_dict[p]['corner_x'] )
        parsed_detector_dict[p]['corner_y'] = float( detector_dict[p]['corner_y'] )

    max_slab_fs = np.array([parsed_detector_dict[k]['max_fs'] for k in parsed_detector_dict.keys()]).max()
    max_slab_ss = np.array([parsed_detector_dict[k]['max_ss'] for k in parsed_detector_dict.keys()]).max()

    x = np.zeros((max_slab_ss+1, max_slab_fs+1), dtype=np.float32)
    y = np.zeros((max_slab_ss+1, max_slab_fs+1), dtype=np.float32)

    for p in parsed_detector_dict.keys():
        i, j = np.meshgrid( np.arange(parsed_detector_dict[p]['max_ss'] - parsed_detector_dict[p]['min_ss'] + 1),
                               np.arange(parsed_detector_dict[p]['max_fs'] - parsed_detector_dict[p]['min_fs'] + 1), indexing='ij')

        dx  = parsed_detector_dict[p]['fs'][1] + 1J * parsed_detector_dict[p]['fs'][0]
        dy  = parsed_detector_dict[p]['ss'][1] + 1J * parsed_detector_dict[p]['ss'][0]
        r_0 = parsed_detector_dict[p]['corner_y'] + 1J * parsed_detector_dict[p]['corner_x']

        r   = i * dy + j * dx + r_0

        y[parsed_detector_dict[p]['min_ss']: parsed_detector_dict[p]['max_ss'] + 1, parsed_detector_dict[p]['min_fs']: parsed_detector_dict[p]['max_fs'] + 1] = r.real
        x[parsed_detector_dict[p]['min_ss']: parsed_detector_dict[p]['max_ss'] + 1, parsed_detector_dict[p]['min_fs']: parsed_detector_dict[p]['max_fs'] + 1] = r.imag
            
    return x, y

def read_geometry_file(fnam, return_preamble = False):
    f = open(fnam, 'r')
    f_lines = []
    for line in f:
        if len(line.lstrip()) > 0 and line.lstrip()[0] != ';':
            f_lines.append(line)
    
    shape = (4, 16)
    min_fs   = np.zeros(shape, np.int16)
    min_ss   = np.zeros(shape, np.int16)
    max_fs   = np.zeros(shape, np.int16)
    max_ss   = np.zeros(shape, np.int16)
    fs       = np.zeros((shape[0], shape[1], 2), np.float64)
    ss       = np.zeros((shape[0], shape[1], 2), np.float64)
    corner_x = np.zeros((shape[0], shape[1]), np.float64)
    corner_y = np.zeros((shape[0], shape[1]), np.float64)

    preamble = {}

    def _get_preamble_float(key, default=None):
        hits = [l for l in f_lines if key in l]
        if not hits:
            return default
        return float(hits[0].rsplit()[2])

    preamble['coffset']    = _get_preamble_float('coffset', default=0.0)
    preamble['adu_per_eV'] = _get_preamble_float('adu_per_eV', default=None)
    preamble['res']        = _get_preamble_float('res', default=1.0)

    for q in range(4):
        for a in range(16):
            qa_string = 'q' + str(q) + 'a' + str(a) + '/'
            min_fs_string = qa_string + 'min_fs'
            min_ss_string = qa_string + 'min_ss'
            max_fs_string = qa_string + 'max_fs'
            max_ss_string = qa_string + 'max_ss'
            fs_string = qa_string + 'fs'
            ss_string = qa_string + 'ss'
            corner_x_string = qa_string + 'corner_x'
            corner_y_string = qa_string + 'corner_y'

            line_min_fs = [l for l in f_lines if min_fs_string in l][0]
            line_min_ss = [l for l in f_lines if min_ss_string in l][0]
            line_max_fs = [l for l in f_lines if max_fs_string in l][0]
            line_max_ss = [l for l in f_lines if max_ss_string in l][0]
            line_fs = [l for l in f_lines if fs_string in l][0]
            line_ss = [l for l in f_lines if ss_string in l][0]
            line_corner_x = [l for l in f_lines if corner_x_string in l][0]
            line_corner_y = [l for l in f_lines if corner_y_string in l][0]

            min_fs[q, a] = int( float(line_min_fs.rsplit()[2] ))
            min_ss[q, a] = int( float(line_min_ss.rsplit()[2] ))
            max_fs[q, a] = int( float(line_max_fs.rsplit()[2] ))
            max_ss[q, a] = int( float(line_max_ss.rsplit()[2] ))
            fs[q, a, 0] = float( line_fs.rsplit()[3][:-1] )
            fs[q, a, 1] = float( line_fs.rsplit()[2][:-1] )
            ss[q, a, 0] = float( line_ss.rsplit()[3][:-1] )
            ss[q, a, 1] = float( line_ss.rsplit()[2][:-1] )
            corner_x[q, a] = float( line_corner_x.rsplit()[2] )
            corner_y[q, a] = float( line_corner_y.rsplit()[2] )
    if return_preamble :
        return min_fs, min_ss, max_fs, max_ss, fs, ss, corner_x, corner_y, preamble
    else :
        return min_fs, min_ss, max_fs, max_ss, fs, ss, corner_x, corner_y

#def make_yx_from_1480_1552(geom_fnam):
#    x = np.zeros((1480, 1552), dtype=np.float32)
#    y = np.zeros((1480, 1552), dtype=np.float32)
#
#    min_fs, min_ss, max_fs, max_ss, fs, ss, corner_x, corner_y = read_geometry_file(geom_fnam)
#
#    for q in range(4):
#        for a in range(16):
#            i, j = np.meshgrid(np.arange(max_ss[q, a] - min_ss[q, a] + 1),
#                               np.arange(max_fs[q, a] - min_fs[q, a] + 1), indexing='ij')
#
#            dx  = fs[q, a][0] + 1J * fs[q, a][1]
#            dy  = ss[q, a][0] + 1J * ss[q, a][1]
#            r_0 = corner_y[q, a] + 1J * corner_x[q, a]
#
#            r   = i * dy + j * dx + r_0
#
#            y[min_ss[q, a]: max_ss[q, a] + 1, min_fs[q, a]: max_fs[q, a]+1] = r.real
#            x[min_ss[q, a]: max_ss[q, a] + 1, min_fs[q, a]: max_fs[q, a]+1] = r.imag
#    return y, x

def make_yx_from_1480_1552(geom_fnam):
    # Generic for any CrystFEL geometry (EIGER, CSPAD, etc.)
    x, y = pixel_maps_from_geometry_file(geom_fnam)
    return y, x
    
def get_ij_slab_shaped(geom_fnam):
    x, y = pixel_maps_from_geometry_file(geom_fnam)

    N = 2 * int(max(abs(y.max()), abs(y.min()))) + 2
    M = 2 * int(max(abs(x.max()), abs(x.min()))) + 2

    i = np.array(y, dtype=int) + N//2 - 1
    j = np.array(x, dtype=int) + M//2 - 1

    ij = (i.flatten(), j.flatten())
    cspad_geom_shape = (N, M)
    return ij, cspad_geom_shape

def make_asic_map_from_1480_1552(geom_fnam):
    asics_slab = np.zeros((1480, 1552), dtype=np.int8)
    asic_shape = (185, 194)
    
    min_fs, min_ss, max_fs, max_ss, fs, ss, corner_x, corner_y = read_geometry_file(geom_fnam)
    
    for q in range(min_fs.shape[0]):
        for a in range(min_fs.shape[1]):
            asics_slab[min_ss[q, a] : max_ss[q, a] + 1, min_fs[q, a] : max_fs[q, a] + 1] = q*16 + a

    asics_geom = apply_geom(geom_fnam, asics_slab + 1)
    asics_geom -= 1
    return asics_geom
    
def ijkl_to_ss_fs(cspad_ijkl):
    if cspad_ijkl.shape != cspad_psana_shape :
        raise ValueError('CSPAD input is not the required shape:' + str(cspad_psana_shape) )
    
    cspad_ij = np.zeros(cspad_geom_shape, dtype=cspad_ijkl.dtype)
    for i in range(4):
        cspad_ij[:, i * cspad_psana_shape[3]: (i+1) * cspad_psana_shape[3]] = cspad_ijkl[i].reshape((cspad_psana_shape[1] * cspad_psana_shape[2], cspad_psana_shape[3]))
    
    return cspad_ij

def ss_fs_to_ijkl(cspad_ij):
    if cspad_ij.shape != cspad_geom_shape :
        raise ValueError('cspad input is not the required shape:' + str(cspad_geom_shape) )
    
    cspad_ijkl = np.zeros(cspad_psana_shape, dtype=cspad_ij.dtype)
    for i in range(4):
        cspad_ijkl[i] = cspad_ij[:, i * cspad_psana_shape[3]: (i+1) * cspad_psana_shape[3]].reshape((cspad_ijkl.shape[1:]))
    
    return cspad_ijkl

def make_yx_from_4_8_185_388(geom_fnam):
    x = np.zeros(cspad_psana_shape, dtype=np.float32)
    y = np.zeros(cspad_psana_shape, dtype=np.float32)

    min_fs, min_ss, max_fs, max_ss, fs, ss, corner_x, corner_y = read_geometry_file(geom_fnam)

    for q in range(4):
        for a_8 in range(8):
            for a_2 in range(2):
                a = a_8 * 2 + a_2
                i, j = np.meshgrid(np.arange(max_ss[q, a] - min_ss[q, a] + 1),
                                   np.arange(max_fs[q, a] - min_fs[q, a] + 1), indexing='ij')

                dx  = fs[q, a][0] + 1J * fs[q, a][1]
                dy  = ss[q, a][0] + 1J * ss[q, a][1]
                r_0 = corner_y[q, a] + 1J * corner_x[q, a]

                r   = i * dy + j * dx + r_0

                y[q, a_8, :, a_2 * 194 : (a_2+1)*194] = r.real
                x[q, a_8, :, a_2 * 194 : (a_2+1)*194] = r.imag
    return y, x

def apply_geom_ij_yx(yx, cspad_np):
    y = yx[0]
    x = yx[1]
    
    N = 2 * int(max(abs(y.max()), abs(y.min()))) + 2
    M = 2 * int(max(abs(x.max()), abs(x.min()))) + 2

    cspad_geom = np.zeros((N, M), dtype=cspad_np.dtype)

    i = np.array(y, dtype=int) + N//2 - 1
    j = np.array(x, dtype=int) + M//2 - 1

    cspad_geom[i.flatten(), j.flatten()] = cspad_np.flatten()
    return cspad_geom

def apply_geom(geom_fnam, cspad_np):
    if cspad_np.shape == (4, 8, 185, 388) :
        y, x = make_yx_from_4_8_185_388(geom_fnam)
    else :
        x, y = pixel_maps_from_geometry_file(geom_fnam)
        
    cspad_geom = apply_geom_ij_yx((y, x), cspad_np)
    return cspad_geom

def get_ij_psana_shaped(geom_fnam):
    y, x = make_yx_from_4_8_185_388(geom_fnam)
    
    N = 2 * int(max(abs(y.max()), abs(y.min()))) + 2
    M = 2 * int(max(abs(x.max()), abs(x.min()))) + 2

    i = np.array(y, dtype=int) + N//2 - 1
    j = np.array(x, dtype=int) + M//2 - 1

    ij = (i.flatten(), j.flatten())
    cspad_geom_shape = (N, M)
    return ij, cspad_geom_shape

def get_corners_ss_fs(q, a, cspad_geom_shape, geom_fnam):
    min_fs, min_ss, max_fs, max_ss, fs, ss, corner_x, corner_y = read_geometry_file(geom_fnam)

    x_asic = cspad_psana_shape[-1] // 2
    y_asic = cspad_psana_shape[-2]

    dx  = fs[q, a][0] + 1J * fs[q, a][1]
    dy  = ss[q, a][0] + 1J * ss[q, a][1]
    r_00 = corner_y[q, a] + 1J * corner_x[q, a]
    r_01 = r_00 + x_asic * dx
    r_11 = r_00 + y_asic * dy + x_asic * dx
    r_10 = r_00 + y_asic * dy

    x = np.array([r_00.imag, r_01.imag, r_11.imag, r_10.imag, r_00.imag])
    y = np.array([r_00.real, r_01.real, r_11.real, r_10.real, r_00.real])

    i = y + cspad_geom_shape[0]//2 - 1
    j = x + cspad_geom_shape[1]//2 - 1
    return i, j

def polarization_map(geom_fnam, z, polarization_axis='x'):
    min_fs, min_ss, max_fs, max_ss, fs, ss, corner_x, corner_y, preamble = read_geometry_file(geom_fnam, return_preamble=True)
    du = 1 / preamble['res']
    y, x = make_yx_from_1480_1552(geom_fnam)
    y *= du
    x *= du
    if polarization_axis == 'x':
        polarization_map = 1 - x**2 / (z**2 + x**2 + y**2)
    return polarization_map
