"""
Functions for data loading
"""

import numpy as np

# DVS event characteristics
EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event

# configuration for N-Cars event
x_mask = 0x00003FFF
y_mask = 0x0FFFC000
pol_mask = 0x10000000
x_shift = 0
y_shift = 14
pol_shift = 28

# configuration for DVS event
##x_mask = 0xFE
##x_shift = 1

# masks
##y_mask = 0x7F00
##y_shift = 8

polarity_mask = 1
polarity_shift = None

valid_mask = 0x80000000
valid_shift = 31


def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()    # convert the line to string
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p
    

def load_raw_events(
    fp, bytes_skip=0, bytes_trim=0, filter_dvs=False, time_first=False
):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()

    # take portion of the 
    if bytes_trim > 0:
        data = data[:-bytes_trim]
        
    data = np.fromstring(data, dtype=">u4")
    
    # check the vector length
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('-----')
        print(data[1:21:2])
        raise ValueError("odd number of data elements")
    
    # read the address and time steps from the binary string
    raw_addr = data[::2]
    time_stamp = data[1::2]
    
    if time_first:
        time_stamp, raw_addr = raw_addr, time_stamp

    # filter the frames based on the predefined masks
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        time_stamp = time_stamp[valid]
        raw_addr = raw_addr[valid]
    return time_stamp, raw_addr

def parse_raw_address(
    addr, 
    x_mask=x_mask,
    x_shift=x_shift,
    y_mask=y_mask,
    y_shift=y_shift,
    polarity_mask=polarity_mask,
    polarity_shift=polarity_shift
):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity

def load_dvs_events(fp, filter_dvs=False, **kwargs):
    time_stamp, addr = load_raw_events(fp, filter_dvs)
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return time_stamp, x, y, polarity

def load_bin_events(fp):
    r"""
    Load dvs events from the binary file (.bin)

    The binary string consists of the following: 
    bit 39 - 32: Xaddress (in pixels)
    bit 31 - 24: Yaddress (in pixels)
    bit 23: Polarity (0 for OFF, 1 for ON)
    bit 22 - 0: Timestamp (in microseconds)
    """

    raw_data = np.fromstring(fp.read(), dtype=np.uint8)
    raw_data = raw_data.astype(np.uint32)
    x = raw_data[::5]
    y = raw_data[1::5]
    polarity = ((raw_data[2::5] & 128) >> 7).astype(np.bool)
    time = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    valid = y != 240

    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]
    time = time[valid].astype(np.int64)
    coords = np.stack((x, y), axis=-1).astype(np.int64)
    return time, coords, polarity
    
def load_atis_events(fp):
    # strip header
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "%":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    fp.seek(p + 2)
    data = np.fromstring(fp.read(), dtype="<u4")

    time = data[::2]
    coords = data[1::2]

    x = read_bits(coords, x_mask, x_shift)
    y = read_bits(coords, y_mask, y_shift)
    pol = read_bits(coords, pol_mask, pol_shift)
    coords = np.stack((x, y), axis=-1)
    return time, coords, pol.astype(np.bool)