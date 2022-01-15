__all__ = (
    'LGVector',
    'LGMMHeader',
    'LGMMUVNorm',
    'LGMMPolygon',
    'LGMMSMatrV1',
    'LGMMSMatrV2',
    'LGMMSegment',
    'LGMMSmatSeg',
    'LGCALHeader',
    'LGCALTorso',
    'LGCALLimb',
    'LGCALFooter',
    'LGCALFile',
    'pack_normal',
    'unpack_normal',
    )

import sys
from typing import Sequence
# god python's packaging and imports suck
if not __package__:
    from binstruct import *
else:
    from .binstruct import *

#---------------------------------------------------------------------------#
# LGMM (skinned mesh .bin) data structures

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
    magic: ByteString(4)    # 'LGMM'
    version: uint32         # 1 or 2
    radius: float32         # bounding radius (always 0 on disk?)
    flags: uint32           # always 0?
    app_data: uint32        # always 0?
    layout: uint8           # 0: in material order; 1: in segment order
    segs: uint8             # count of segments
    smatrs: uint8           # count of single-material regions
    smatsegs: uint8         # count of single-material segments
    pgons: uint16           # count of polygons
    verts: uint16           # count of vertices
    weights: uint16         # count of weights
    pad: uint16
    map_off: uint32         # offset to mappings (uint8) from seg/smatr to smatsegs
    seg_off: uint32         # relative to start of the model, used to generate pointers
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
    name: ByteString(16)
    handle: uint32
    # union: { uv, ipal }
    uv: float32         # (when mat_type==0)
    # ipal: uint32      # (when mat_type==1)
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
    name: ByteString(16)
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

#---------------------------------------------------------------------------#
# .cal (skeleton rest pose) data structures

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

class LGCALFile:
    header: LGCALHeader
    p_torsos: Sequence[LGCALTorso]
    p_limbs: Sequence[LGCALLimb]
    footer: LGCALFooter

    def __init__(self, filename='', data=None):
        if data is None:
            with open(filename, 'rb') as f:
                data = f.read()
        view = memoryview(data)
        offset = 0
        header = LGCALHeader.read(view, offset=offset)
        if header.version not in (1,):
            raise ValueError("Only version 1 .cal files are supported")
        offset += LGCALHeader.size()
        p_torsos = StructView(view, LGCALTorso, offset=offset, count=header.torsos)
        offset += p_torsos.size()
        p_limbs = StructView(view, LGCALLimb, offset=offset, count=header.limbs)
        offset += p_limbs.size()
        footer = LGCALFooter.read(view, offset=offset)
        self.header = header
        self.p_torsos = p_torsos
        self.p_limbs = p_limbs
        self.footer = footer

    def dump(self, f=sys.stdout):
        print("CAL:", file=f)
        print(f"  version: {self.header.version}", file=f)
        print(f"  torsos: {self.header.torsos}", file=f)
        print(f"  limbs: {self.header.limbs}", file=f)
        for i, torso in enumerate(self.p_torsos):
            print(f"torso {i}:", file=f)
            print(f"  joint: {torso.joint}", file=f)
            print(f"  parent: {torso.parent}", file=f)
            print(f"  fixed_points: {torso.fixed_points}", file=f)
            print(f"  joint_id:", file=f)
            k = torso.fixed_points
            for joint_id in torso.joint_id[:k]:
                print(f"    {joint_id}", file=f)
            print(f"  pts:", file=f)
            for pt in torso.pts[:k]:
                print(f"    {pt.x}, {pt.y}, {pt.z}", file=f)
        for i, limb in enumerate(self.p_limbs):
            print(f"limb {i}:", file=f)
            print(f"  torso_id: {limb.torso_id}", file=f)
            print(f"  bend: {limb.bend}", file=f)
            print(f"  segments: {limb.segments}", file=f)
            print(f"  joint_id:", file=f)
            k = limb.segments
            for joint_id in limb.joint_id[:k+1]:
                print(f"    {joint_id}", file=f)
            print(f"  seg:", file=f)
            for seg in limb.seg[:k]:
                print(f"    {seg}")
            print(f"  seg_len:", file=f)
            for seg_len in limb.seg_len[:k]:
                print(f"    {seg_len}", file=f)
        print(f"scale: {self.footer.scale}", file=f)
        print(file=f)
