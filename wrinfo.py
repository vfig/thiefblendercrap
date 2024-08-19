import struct
import sys

def print_worldrep_info(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    def read_uint32(offset):
        return struct.unpack_from('L', buffer=data, offset=offset)[0]
    def read_chars(count, offset):
        return struct.unpack_from(f'{count}s', buffer=data, offset=offset)[0]
    toc_offset = read_uint32(0)
    toc_count = read_uint32(toc_offset)
    wr_offset = 0
    wr_type = ''
    wr_value0 = 0
    wr_value1 = 0
    wr_value2 = 0
    wr_value3 = 0
    wr_value4 = 0
    for i in range(toc_count):
        offset = toc_offset+4+i*20
        raw_name = read_chars(12, offset)
        chunk_offset = read_uint32(offset+12)
        name, _, _ = raw_name.partition(b'\x00')
        name = name.decode('ascii')
        if name in ('WR', 'WRRGB', 'WREXT'):
            wr_offset = chunk_offset
            wr_type = name
            break
    else:
        print(f"{filename}: No worldrep chunk")
        return
    # Get the worldrep version
    major = read_uint32(wr_offset+12)
    minor = read_uint32(wr_offset+12+4)
    if wr_type=='WREXT':
        wr_value0 = read_uint32(wr_offset+24)
        wr_value1 = read_uint32(wr_offset+28)
        wr_value2 = read_uint32(wr_offset+32)
        wr_value3 = read_uint32(wr_offset+36)
        wr_value4 = read_uint32(wr_offset+40)
        print(f"{filename}:\t{wr_type} version {major}.{minor}\t(a:{wr_value0},\tb:{wr_value1},\tc:{wr_value2},\td:{wr_value3},\te:{wr_value4})")
    else:
        print(f"{filename}:\t{wr_type} version {major}.{minor}")

if __name__=='__main__':
    for filename in sys.argv[1:]:
        try:
            print_worldrep_info(filename)
        except Exception as e:
            print(f"{filename}: Error, {e}")
