import glob
import itertools
import math
import os, os.path
import random
import struct
import sys
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
    alpha: float32          # NOTE: caps, alpha, and self_illum seem to be
    self_illum: float32     #       unimplemented.
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

def read_mesh(bin_filename, dump=False):
    cal_filename = os.path.splitext(bin_filename)[0]+'.cal'
    with open(bin_filename, 'rb') as f:
        bin_data = f.read()
    with open(cal_filename, 'rb') as f:
        cal_data = f.read()

    # Parse the .bin file
    bin_view = memoryview(bin_data)
    header = LGMMHeader.read(bin_view)
    if header.magic != b'LGMM':
        raise ValueError("File is not a .bin mesh (LGMM)")
    if header.version not in (1, 2):
        raise ValueError("Only version 1 and 2 .bin files are supported")
    if header.layout!=0:
        raise ValueError("Only material-layout (layout=0) meshes are supported")

    map_count = header.seg_off-header.map_off
    p_maps = StructView(bin_view, uint8, offset=header.map_off, count=map_count)
    p_segs = StructView(bin_view, LGMMSegment, offset=header.seg_off, count=header.segs)
    p_smatrs = StructView(bin_view, (LGMMSMatrV2 if header.version==2 else LGMMSMatrV1),
        offset=header.smatr_off, count=header.smatrs)
    p_smatsegs = StructView(bin_view, LGMMSmatSeg, offset=header.smatseg_off, count=header.smatsegs)
    p_pgons = StructView(bin_view, LGMMPolygon, offset=header.pgon_off, count=header.pgons)
    p_norms = StructView(bin_view, LGVector, offset=header.norm_off, count=header.pgons) # TODO: is count correct??
    p_verts = StructView(bin_view, LGVector, offset=header.vert_vec_off, count=header.verts)
    p_uvnorms = StructView(bin_view, LGMMUVNorm, offset=header.vert_uvn_off, count=header.verts)
    p_weights = StructView(bin_view, float32, offset=header.weight_off, count=header.weights)

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

    # Now do some checks
    ALL_CHECKS = False # if True, include checks which commonly fail
    def check(test, message):
        prefix = "(ok)  " if test else "(FAIL)"
        if not test: print(prefix, message, " - ", bin_filename)

    # FAILS only for TG expzom.bin
    check((map_count==2*len(p_smatsegs)), f"mappings ({map_count}) = 2x smatseg count ({len(p_smatsegs)})")

    for smatr_id, smatr in enumerate(p_smatrs):

        # OKAY, caps _is_ used. alpha and self-illum work as expected!
        # used for diver's faceplate, and robots' eyes and boilers.
        if ALL_CHECKS:
            zero_caps = (smatr.caps==0)
            check(zero_caps, f"smatr {smatr_id} has zero caps")
            if not zero_caps:
                print(f"  caps: {smatr.caps:x}, alpha {smatr.alpha}, self_illum {smatr.self_illum}")

        # OKAY, so smatrs do list all their verts. thats okay, we can handle
        # that because we already sorted verts by material.
        if ALL_CHECKS:
            check(smatr.verts==0, f"smatr {smatr_id} has zero verts")

        # NEVER FAILED: so handle is unused
        check(smatr.handle==0, f"smatr {smatr_id} has nonzero handle")

        check(smatr.mat_type==0, f"smatr {smatr_id} has mat_type 0")
    for seg_id, seg in enumerate(p_segs):
        check(seg.verts==0, f"seg {seg_id} has zero verts")
        check(seg.pgons==0, f"seg {seg_id} has zero pgons")
        is_stretchy = bool(seg.flags & 1)
        map_start = seg.map_start
        map_end = map_start+seg.smatsegs
        for smatseg_id in p_maps[map_start:map_end]:
            smatseg = p_smatsegs[smatseg_id]
            # OKAY, so quite a lot of models have polys listed for non-stretchy
            # smatsegs! dig into the code again, see if/how this is used.
            # (i think it is not used, but this could be wrong!)
            # also check out some of these models; it could be these are polys
            # exclusive to these smatsegs??
            if ALL_CHECKS:
                if not is_stretchy:
                    check(smatseg.pgons==0, f"non-stretchy smatseg {smatseg_id} has zero pgons")

    if not dump: return
    dumpf = sys.stdout
    print("BIN:", file=dumpf)
    print("Segments", file=dumpf)
    for i, seg in enumerate(p_segs):
        print(f"  Seg {i}:", file=dumpf)
        print(f"    bbox: {seg.bbox}", file=dumpf)
        print(f"    joint_id: {seg.joint_id}", file=dumpf)
        print(f"    smatsegs: {seg.smatsegs}", file=dumpf)
        print(f"    map_start: {seg.map_start}", file=dumpf)
        print(f"    flags: {seg.flags}", file=dumpf)
        print(f"    pgons: {seg.pgons}", file=dumpf)
        print(f"    pgon_start: {seg.pgon_start}", file=dumpf)
        print(f"    verts: {seg.verts}", file=dumpf)
        print(f"    vert_start: {seg.vert_start}", file=dumpf)
        print(f"    weight_start: {seg.weight_start}", file=dumpf)
    print("Smatrs", file=dumpf)
    for i, smatr in enumerate(p_smatrs):
        print(f"  Smatr {i}:", file=dumpf)
        print(f"    name: {smatr.name}", file=dumpf)
        print(f"    caps: {smatr.caps}", file=dumpf)
        print(f"    alpha: {smatr.alpha}", file=dumpf)
        print(f"    self_illum: {smatr.self_illum}", file=dumpf)
        print(f"    for_rent: {smatr.for_rent}", file=dumpf)
        print(f"    handle: {smatr.handle}", file=dumpf)
        print(f"    uv: {smatr.uv}", file=dumpf)
        print(f"    mat_type: {smatr.mat_type}", file=dumpf)
        print(f"    smatsegs: {smatr.smatsegs}", file=dumpf)
        print(f"    map_start: {smatr.map_start}", file=dumpf)
        print(f"    flags: {smatr.flags}", file=dumpf)
        print(f"    pgons: {smatr.pgons}", file=dumpf)
        print(f"    pgon_start: {smatr.pgon_start}", file=dumpf)
        print(f"    verts: {smatr.verts}", file=dumpf)
        print(f"    vert_start: {smatr.vert_start}", file=dumpf)
        print(f"    weight_start: {smatr.weight_start}", file=dumpf)
        print(f"    pad: {smatr.pad}", file=dumpf)
    print("SmatSegs", file=dumpf)
    for i, smatseg in enumerate(p_smatsegs):
        print(f"  Smatseg {i}:", file=dumpf)
        print(f"    pgons: {smatseg.pgons}", file=dumpf)
        print(f"    pgon_start: {smatseg.pgon_start}", file=dumpf)
        print(f"    verts: {smatseg.verts}", file=dumpf)
        print(f"    vert_start: {smatseg.vert_start}", file=dumpf)
        print(f"    weight_start: {smatseg.weight_start}", file=dumpf)
        print(f"    pad: {smatseg.pad}", file=dumpf)
        print(f"    smatr_id: {smatseg.smatr_id}", file=dumpf)
        print(f"    seg_id: {smatseg.seg_id}", file=dumpf)
    print(file=dumpf)

    print("Maps:", file=dumpf)
    for i, m in enumerate(p_maps):
        print(f"  {i}: {m}", file=dumpf)
    print("Pgons:", file=dumpf)
    for i, p in enumerate(p_pgons):
        print(f"  {i}: verts {p.vert[0]},{p.vert[1]},{p.vert[2]}; smatr {p.smatr_id}; norm: {p.norm}, d: {p.d}", file=dumpf)
    print("Verts:", file=dumpf)
    for i, v in enumerate(p_verts):
        print(f"  {i}: {v.x},{v.y},{v.z}", file=dumpf)
    print("Norms:", file=dumpf)
    for i, n in enumerate(p_norms):
        print(f"  {i}: {n.x},{n.y},{n.z}", file=dumpf)
    print("UVNorms:", file=dumpf)
    for i, n in enumerate(p_uvnorms):
        print(f"  {i}: {n.u},{n.v}; norm {n.norm}", file=dumpf)
    print("Weights:", file=dumpf)
    for i, w in enumerate(p_weights):
        print(f"  {i}: {w}", file=dumpf)

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

def main():
    read_mesh("e:/dev/thief/T2FM/test_arachnorig/mesh/expzom.bin ", dump=True)
    return
    base_path = sys.argv[1]
    pattern = os.path.join(base_path, "*.bin")
    for filename in glob.glob(pattern):
        try:
            read_mesh(filename)
        except:
            print(f"Reading {filename}:")
            import traceback
            traceback.print_exc(file=sys.stdout)

if __name__=='__main__':
    main()
