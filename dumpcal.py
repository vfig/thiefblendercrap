import itertools
import math
import os, os.path
import random
import struct
import zlib

from array import array
from typing import NewType, get_type_hints

class PrimitiveTypeMixin:
    _format_string = ''
    _name = ''
    @classmethod
    def size(cls):
        return struct.calcsize(cls._format_string)
    @classmethod
    def read(cls, view, offset=0):
        values = struct.unpack_from(cls._format_string, view, offset=offset)
        return cls(values[0])
    def write(self, f):
        f.write(struct.pack(self.__class__._format_string, self))
    def __repr__(self):
        return f"<PrimitiveType {self._name}>"

def make_primitive_type(name_, base_type, format_string):
    class _PrimitiveType(PrimitiveTypeMixin, base_type):
        _format_string = format_string
        _name = name_
    _PrimitiveType.__name__ = name_
    return _PrimitiveType

int8 = make_primitive_type('int8', int, 'b')
int16 = make_primitive_type('int16', int, 'h')
int32 = make_primitive_type('int32', int, 'l')
uint8 = make_primitive_type('uint8', int, 'B')
uint16 = make_primitive_type('uint16', int, 'H')
uint32 = make_primitive_type('uint32', int, 'L')
float32 = make_primitive_type('float32', float, 'f')
bytes4 = make_primitive_type('bytes4', bytes, '4s')
bytes16 = make_primitive_type('bytes16', bytes, '16s')

class ArrayInstance:
    typeref = None
    count = 0
    def __init__(self, values):
        if len(values) != self.count:
            raise ValueError(f"Expected {self.count} values, got {values!r}")
        self.values = [self.typeref(v) for v in values]
    def __getitem__(self, i):
        return self.values[i]
    def __len__(self):
        return self.count
    def __repr__(self):
        return f"<ArrayInstance of {self.count} x {self.typeref.__name__}>"

def Array(typeref_, count_):
    class TypedArrayInstance(ArrayInstance):
        typeref = typeref_
        count = count_
        @classmethod
        def size(cls):
            return cls.count*cls.typeref.size()
        @classmethod
        def read(cls, view, offset=0):
            values = []
            stride = cls.typeref.size()
            for i in range(cls.count):
                value = cls.typeref.read(view, offset=offset)
                offset += stride
                values.append(value)
            return cls(values)
        def write(self, f):
            for i in range(self.count):
                self.values[i].write(f)
    TypedArrayInstance.__name__ = f"{typeref_.__name__}x{count_}"
    return TypedArrayInstance

# TODO: can i turn Struct into a decorate that uses @dataclass and adds the
# read/write/size methods??
# TODO: also rename this because it clashes with struct.Struct haha
class Struct:
    def __init__(self, values=None):
        if self.__class__==Struct:
            raise TypeError("Cannot instantiate Struct itself, only subclasses")
        hints = get_type_hints(self.__class__)
        if len(hints)==0:
            raise TypeError(f"{self.__class__.__name__} has no fields defined")
        if values is None:
            values = self.default_values()
        if len(values)!=len(hints):
            raise ValueError(f"Expected {len(self.hints)} values")
        for (name, typeref), value in zip(hints.items(), values):
            setattr(self, name, typeref(value))

    @classmethod
    def size(cls):
        size = 0
        # TODO: this ignores padding and alignment!
        hints = get_type_hints(cls)
        for name, typeref in hints.items():
            size += typeref.size()
        return size

    @classmethod
    def read(cls, data, offset=0):
        hints = get_type_hints(cls)
        values = []
        for name, typeref in hints.items():
            value = typeref.read(data, offset=offset)
            offset += typeref.size()
            values.append(value)
        return cls(values)

    def write(self, f):
        hints = get_type_hints(self.__class__)
        for name, typeref in hints.items():
            value = getattr(self, name)
            value.write(f)

class StructView:
    def __init__(self, view, typeref, *, offset=0, count=-1, size=-1):
        self.view = view
        self.typeref = typeref
        self.stride = typeref.size()
        self.offset = offset
        if count==-1 and size==-1:
            raise ValueError("Must provide either count, or size, or neither (to use entire view)")
        if count!=-1:
            self.count = count
        elif size!=-1:
            self.count = size//self.stride
        else:
            self.count = len(view)//self.stride

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.step is not None and i.step != 1:
                raise ValueError("Slices with step size other than 1 are not supported.")
            if i.start >= 0:
                start = min(max(0, i.start), self.count-1)
            else:
                start = min(max(0, self.count+i.start), self.count-1)
            offset = self.offset+start*self.stride
            if i.stop >= 0:
                count = min(self.count, i.stop-i.start)
            else:
                count = min(self.count, self.count+i.stop-i.start)
            return self.__class__(self.view, self.typeref,
                offset=offset, count=count)
        else:
            if not (0<=i<self.count):
                raise IndexError(i)
            offset = self.offset+i*self.stride
            return self.typeref.read(self.view, offset=offset)

    def size(self):
        return self.count*self.stride

class LGVector(Struct):
    x: float32
    y: float32
    z: float32
    def __len__(self):
        return 3
    def __getitem__(self, i):
        return getattr(self, ['x','y','z'][i])
    def __str__(self):
        return f"({self.x},{self.y},{self.z})"

    @classmethod
    def default_values(cls):
        return (0.0,0.0,0.0)

class LGMMHeader(Struct):
    magic: bytes4       # 'LGMM'
    version: uint32     # 1 or 2
    radius: float32     # bounding radius (always 0 on disk?)
    flags: uint32       # always 0?
    app_data: uint32    # always 0?
    layout: uint8       # 0: in material order; 1: in segment order
    segs: uint8         # count of segments
    smatrs: uint8       # count of single-material regions
    smatsegs: uint8     # count of single-material segments
    pgons: uint16       # count of polygons
    verts: uint16       # count of vertices
    weights: uint16     # count of weights
    pad: uint16
    map_off: uint32     # offset to mappings (uint8) from seg/smatr to smatsegs
    seg_off: uint32     # relative to start of the model, used to generate pointers
    smatr_off: uint32
    smatseg_off: uint32
    pgon_off: uint32
    norm_off: uint32        # offset to array of pgon normal vectors
    vert_vec_off: uint32    # offset to array of mxs_vectors of vertex positions
    vert_uvn_off: uint32    # offset to array of other vertex data (uvs, normals etc)
    weight_off: uint32      # offset to array of weights (float32)

    @classmethod
    def default_values(cls):
        return (
            b'LGMM',    # magic
            2,          # version
            0.0,        # radius
            0,          # flags
            0,          # app_data
            0,          # layout
            0,          # segs
            0,          # smatrs
            0,          # smatsegs
            0,          # pgons
            0,          # verts
            0,          # weights
            0,          # pad
            0,          # map_off
            0,          # seg_off
            0,          # smatr_off
            0,          # smatseg_off
            0,          # pgon_off
            0,          # norm_off
            0,          # vert_vec_off
            0,          # vert_uvn_off
            0,          # weight_off
            )

def _to_signed16(u):
    sign = -1 if (u&0x8000) else 1
    return sign*(u&0x7fff)

def _to_unsigned16(u):
    sign = 0x8000 if u<0 else 0x000
    return (int(abs(u))&0x7fff|sign)

def pack_normal(v):
    return (
         ((_to_unsigned16(v[0]*16384.0)&0xFFC0)<<16)
        |((_to_unsigned16(v[1]*16384.0)&0xFFC0)<<6)
        |((_to_unsigned16(v[2]*16384.0)&0xFFC0)>>4)
        )

def unpack_normal(norm):
    return (
        _to_signed16((norm>>16)&0xFFC0)/16384.0,
        _to_signed16((norm>>6)&0xFFC0)/16384.0,
        _to_signed16((norm<<4)&0xFFC0)/16384.0,
        )

class LGMMUVNorm(Struct):
    u: float32
    v: float32
    norm: uint32    # packed normal

    @classmethod
    def default_values(cls):
        return (0.0,0.0,0)

class LGMMPolygon(Struct):
    vert: Array(uint16, 3)
    smatr_id: uint16
    d: float32
    norm: uint16
    pad: uint16

    @classmethod
    def default_values(cls):
        return (
            [0.0,0.0,0.0],  # vert
            -1,             # smatr_id
            0.0,            # d
            0,              # norm
            0,              # pad
            )

class LGMMSMatrV1(Struct):
    name: bytes16
    handle: uint32
    # union:
    uv: float32
    # ipal: uint32
    mat_type: uint8     # 0 = texture, 1 = virtual color
    smatsegs: uint8
    map_start: uint8
    flags: uint8
    pgons: uint16       # from here, only for material order
    pgon_start: uint16
    verts: uint16
    vert_start: uint16
    weight_start: uint16    # number of weights = num vertices in segment
    pad: uint16
    # For forward compatibility with V2:
    @property
    def caps(self): return uint32(0)
    @property
    def alpha(self): return float32(1.0)
    @property
    def self_illum(self): return float32(0.0);
    @property
    def for_rent(self): return uint32(0)

    @classmethod
    def default_values(cls):
        return (
            b'\x00'*16, # name
            0,          # handle
            0.0,        # uv
            0,          # mat_type
            0,          # smatsegs
            0,          # map_start
            0,          # flags
            0,          # pgons
            0,          # pgon_start
            0,          # verts
            0,          # vert_start
            0,          # weight_start
            0,          # pad
            )

class LGMMSMatrV2(Struct):
    name: bytes16
    caps: uint32            # 1 = use alpha; 2 = use self_illum
    alpha: float32          # 0.0 = transparent; 1.0 = opaque
    self_illum: float32     # 0.0 = none; 1.0 = full self-illumination
    for_rent: uint32        # junk
    handle: uint32          # zero on disk
    # union:
    uv: float32
    # ipal: uint32
    mat_type: uint8         # 0 = texture, 1 = virtual color
    smatsegs: uint8         # count of smatsegs with this material
    map_start: uint8        # index of first smatseg_id in mappings
    flags: uint8            # zero
    pgons: uint16           # count of pgons with this material
    pgon_start: uint16      # index of first pgon with this material
    verts: uint16           # zero
    vert_start: uint16      # zero
    weight_start: uint16    # zero
    pad: uint16             # junk

    @classmethod
    def default_values(cls):
        return (
            b'abcdefghijkl.png',    # name
            0,                      # caps
            0.0,                    # alpha
            0.0,                    # self_illum
            0,                      # for_rent
            0,                      # handle
            0.0,                    # uv
            0,                      # mat_type
            0,                      # smatsegs
            0,                      # map_start
            0,                      # flags
            0,                      # pgons
            0,                      # pgon_start
            0,                      # verts
            0,                      # vert_start
            0,                      # weight_start
            0,                      # pad
            )

class LGMMSegment(Struct):
    bbox: uint32            # always zero
    joint_id: uint8
    smatsegs: uint8         # count of smatsegs in segment
    map_start: uint8        # index in map of first smatseg_id
    flags: uint8            # 1 = stretchy segment
    pgons: uint16           # always zero (material-ordered .bin)
    pgon_start: uint16      #   "
    verts: uint16           #   "
    vert_start: uint16      #   "
    weight_start: uint16    #   "
    pad: uint16             # junk

    @classmethod
    def default_values(cls):
        return (
            0,  # bbox
            0,  # joint_id
            0,  # smatsegs
            0,  # map_start
            0,  # flags
            0,  # pgons
            0,  # pgon_start
            0,  # verts
            0,  # vert_start
            0,  # weight_start
            0,  # pad
            )

class LGMMSmatSeg(Struct):
    pgons: uint16
    pgon_start: uint16
    verts: uint16
    vert_start: uint16
    weight_start: uint16 # number of weights = num vertices in segment
    pad: uint16
    smatr_id: uint16
    seg_id: uint16

    @classmethod
    def default_values(cls):
        return (
            0,  # pgons
            0,  # pgon_start
            0,  # verts
            0,  # vert_start
            0,  # weight_start
            0,  # pad
            0,  # smatr_id
            0,  # seg_id
            )

class LGCALHeader(Struct):
    version: uint32     # only know version 1
    torsos: uint32
    limbs: uint32

class LGCALTorso(Struct):
    joint: int32
    parent: int32
    fixed_points: int32
    joint_id: Array(int32, 16)
    pts: Array(LGVector, 16)

class LGCALLimb(Struct):
    torso_id: int32
    bend: int32
    segments: int32
    joint_id: Array(int16, 17)
    seg: Array(LGVector, 16)
    seg_len: Array(float32, 16)

class LGCALFooter(Struct):
    scale: float32

def vec_add(a,b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def vec_mulf(f,a):
    return (a[0]*f, a[1]*f, a[2]*f)

def vec_fmt(a):
    return f"({a[0]:0.6f},{a[1]:0.6f},{a[2]:0.6f})"

def dump_cal(cal_filename):
    with open(cal_filename, 'rb') as f:
        cal_data = f.read()
    #dump_filename = os.path.splitext(cal_filename)[0]+'.dump'
    #dumpf = open(dump_filename, 'w')
    import sys
    dumpf = sys.stdout

    # Parse the .cal file
    cal_view = memoryview(cal_data)
    offset = 0
    cal_header = LGCALHeader.read(cal_view, offset=offset)
    if cal_header.version not in (1,):
        raise ValueError("Only version 1 .cal files are supported")
    offset += LGCALHeader.size()
    p_torsos = StructView(cal_view, LGCALTorso, offset=offset, count=cal_header.torsos)
    offset += p_torsos.size()
    p_limbs = StructView(cal_view, LGCALLimb, offset=offset, count=cal_header.limbs)
    offset += p_limbs.size()
    cal_footer = LGCALFooter.read(cal_view, offset=offset)

    print("CAL:", file=dumpf)
    print(f"  version: {cal_header.version}", file=dumpf)
    print(f"  torsos: {cal_header.torsos}", file=dumpf)
    print(f"  limbs: {cal_header.limbs}", file=dumpf)
    for i, torso in enumerate(p_torsos):
        print(f"torso {i}:", file=dumpf)
        print(f"  joint: {torso.joint}", file=dumpf)
        print(f"  parent: {torso.parent}", file=dumpf)
        print(f"  fixed_points: {torso.fixed_points}", file=dumpf)
        print(f"  joint_id:", file=dumpf)
        k = torso.fixed_points
        for joint_id in torso.joint_id[:k]:
            print(f"    {joint_id}", file=dumpf)
        print(f"  pts:", file=dumpf)
        for pt in torso.pts[:k]:
            print(f"    {pt.x}, {pt.y}, {pt.z}", file=dumpf)
    for i, limb in enumerate(p_limbs):
        print(f"limb {i}:", file=dumpf)
        print(f"  torso_id: {limb.torso_id}", file=dumpf)
        print(f"  bend: {limb.bend}", file=dumpf)
        print(f"  segments: {limb.segments}", file=dumpf)
        print(f"  joint_id:", file=dumpf)
        k = limb.segments
        for joint_id in limb.joint_id[:k+1]:
            print(f"    {joint_id}", file=dumpf)
        print(f"  seg:", file=dumpf)
        for seg in limb.seg[:k]:
            print(f"    {seg}")
        print(f"  seg_len:", file=dumpf)
        for seg_len in limb.seg_len[:k]:
            print(f"    {seg_len}", file=dumpf)
    print(f"scale: {cal_footer.scale}", file=dumpf)
    print(file=dumpf)

    #dumpf.close()

    # Find the origin point of every joint
    head_by_joint_id = {}
    tail_by_joint_id = {}
    parent_by_joint_id = {}
    is_connected_by_joint_id = {}
    is_limb_end_by_joint_id = {}
    for torso in p_torsos:
        if torso.parent == -1:
            j = torso.joint
            # BUG!  burrick.cal fails this assertion! because torso 0 (legs) and
            #       torso 1 (tail) *both* are on the BUTT joint, with no parent.
            assert j not in head_by_joint_id, f"joint {j} already in head list!"
            assert j not in tail_by_joint_id, f"joint {j} already in tail list!"
            head_by_joint_id[j] = (0,0,0)
            tail_by_joint_id[j] = (1,0,0)
            assert j not in parent_by_joint_id, f"joint {j} already in parent list!"
            parent_by_joint_id[j] = -1
        else:
            j = torso.joint
            assert j in head_by_joint_id, f"joint {j} not found in head list!"
            assert j not in tail_by_joint_id, f"joint {j} already in tail list!"
            tail_by_joint_id[j] = vec_add(head_by_joint_id[j], (1,0,0))
            assert j not in parent_by_joint_id, f"joint {j} already in parent list!"
            parent_by_joint_id[j] = p_torsos[torso.parent].joint
        is_connected_by_joint_id[j] = False
        is_limb_end_by_joint_id[j] = False
        root = head_by_joint_id[torso.joint]
        k = torso.fixed_points
        parts = zip(
            torso.joint_id[:k],
            torso.pts[:k])
        for j, pt in parts:
            assert j not in head_by_joint_id, f"joint {j} already in head list!"
            head_by_joint_id[j] = vec_add(root, pt)
    for limb in p_limbs:
        j = limb.joint_id[0]
        assert j in head_by_joint_id, f"joint {j} not found in head list!"
        head = head_by_joint_id[j]
        k = limb.segments
        parts = zip(
            limb.joint_id[:k+1],
            limb.seg[:k] + [limb.seg[k-1]], # finger etc. bones, tail gets wrist vector
            limb.seg_len[:k] + [0.25])      # finger etc. bones, get fixed length
        pj = j
        for i, (j, seg, seg_len) in enumerate(parts):
            head_by_joint_id[j] = head
            tail = vec_add(head, vec_mulf(seg_len, seg))
            tail_by_joint_id[j] = tail
            head = tail
            assert j not in parent_by_joint_id, f"joint {j} already in parent list!"
            if i==0:
                parent_by_joint_id[j] = p_torsos[limb.torso_id].joint
                is_connected_by_joint_id[j] = False
            else:
                parent_by_joint_id[j] = pj
                is_connected_by_joint_id[j] = True
            pj = j

    parent_joint_ids = set(parent_by_joint_id.values())
    for j, _ in enumerate(head_by_joint_id):
        is_limb_end_by_joint_id[j] = (j not in parent_joint_ids)

    assert sorted(head_by_joint_id.keys())==sorted(tail_by_joint_id.keys())

    bones_by_joint_id = {}
    for j in sorted(head_by_joint_id.keys()):
        # name, parent, connected, head_pos, tail_pos, limb_end)
        print(
            f"{j}: ('xxnamexx', "
            f"{parent_by_joint_id[j]}, "
            f"{is_connected_by_joint_id[j]}, "
            f"{vec_fmt(head_by_joint_id[j])}, "
            f"{vec_fmt(tail_by_joint_id[j])}, "
            f"{is_limb_end_by_joint_id[j]}),")

if __name__=='__main__':
    import sys
    cal_filename = sys.argv[1]
    dump_cal(cal_filename)