import bpy
import bmesh
import itertools
import math
import mathutils
import os, os.path
import random
import struct
import zlib

from array import array
from dataclasses import dataclass
from bpy.props import EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from mathutils import Vector
from mathutils.bvhtree import BVHTree
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

#---------------------------------------------------------------------------#
# Import

def create_color_table(size):
    rand = random.Random(0)
    def random_color(alpha=1.0):
        r = rand.uniform(0.0, 1.0)
        g = rand.uniform(0.0, 1.0)
        b = rand.uniform(0.0, 1.0)
        return (r, g, b, alpha)
    return [random_color() for i in range(size)]
ID_COLOR_TABLE = create_color_table(1024)
def id_color(id):
    return ID_COLOR_TABLE[abs(id)%len(ID_COLOR_TABLE)]

def create_empty(name, location, context=None, link=True):
    o = bpy.data.objects.new(name, None)
    o.location = location
    o.empty_display_size = 0.125
    o.empty_display_type = 'PLAIN_AXES'
    if context and link:
        coll = context.view_layer.active_layer_collection.collection
        coll.objects.link(o)
    return o

def create_object(name, mesh, location, context=None, link=True):
    o = bpy.data.objects.new(name, mesh)
    o.location = location
    if context and link:
        coll = context.view_layer.active_layer_collection.collection
        coll.objects.link(o)
    return o

def create_armature(name, location, context=None, link=True, display_type='OCTAHEDRAL'):
    arm = bpy.data.armatures.new(name)
    arm.display_type = display_type
    o = bpy.data.objects.new(name, arm)
    o.location = location
    o.show_in_front = True
    if context and link:
        coll = context.view_layer.active_layer_collection.collection
        coll.objects.link(o)
    return o

def do_import_mesh(context, bin_filename):
    cal_filename = os.path.splitext(bin_filename)[0]+'.cal'
    with open(bin_filename, 'rb') as f:
        bin_data = f.read()
    with open(cal_filename, 'rb') as f:
        cal_data = f.read()
    dump_filename = os.path.splitext(bin_filename)[0]+'.dump'
    dumpf = open(dump_filename, 'w')

    # Parse the .bin file
    bin_view = memoryview(bin_data)
    header = LGMMHeader.read(bin_view)
    if header.magic != b'LGMM':
        raise ValueError("File is not a .bin mesh (LGMM)")
    if header.version not in (1, 2):
        raise ValueError("Only version 1 and 2 .bin files are supported")
    if header.layout!=0:
        raise ValueError("Only material-layout (layout=0) meshes are supported")
    print(f"magic: {header.magic}", file=dumpf)
    print(f"version: {header.version}", file=dumpf)
    print(f"radius: {header.radius:f}", file=dumpf)
    print(f"flags: {header.flags:04x}", file=dumpf)
    print(f"app_data: {header.app_data:04x}", file=dumpf)
    print(f"layout: {header.layout}", file=dumpf)
    print(f"segs: {header.segs}", file=dumpf)
    print(f"smatrs: {header.smatrs}", file=dumpf)
    print(f"smatsegs: {header.smatsegs}", file=dumpf)
    print(f"pgons: {header.pgons}", file=dumpf)
    print(f"verts: {header.verts}", file=dumpf)
    print(f"weights: {header.weights}", file=dumpf)
    print(f"map_off: {header.map_off:08x}", file=dumpf)
    print(f"seg_off: {header.seg_off:08x}", file=dumpf)
    print(f"smatr_off: {header.smatr_off:08x}", file=dumpf)
    print(f"smatseg_off: {header.smatseg_off:08x}", file=dumpf)
    print(f"pgon_off: {header.pgon_off:08x}", file=dumpf)
    print(f"norm_off: {header.norm_off:08x}", file=dumpf)
    print(f"vert_vec_off: {header.vert_vec_off:08x}", file=dumpf)
    print(f"vert_uvn_off: {header.vert_uvn_off:08x}", file=dumpf)
    print(f"weight_off: {header.weight_off:08x}", file=dumpf)
    print(file=dumpf)

    # TODO: does this match number of segs? smatrs? anything?
    map_count = header.seg_off-header.map_off
    print(f"maps: {map_count}", file=dumpf)
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

    dumpf.close()

    # test: check that only stretchy segments have pgons in their smatsegs:
    # UNTRUE! segment 9 (head) is non-stretchy, but has 30 pgons!
    # however, i still dont think it matters, for hardware rendering?
    for seg_id, seg in enumerate(p_segs):
        is_stretchy = bool(seg.flags & 1)
        map_start = seg.map_start
        map_end = map_start+seg.smatsegs
        for smatseg_id in p_maps[map_start:map_end]:
            smatseg = p_smatsegs[smatseg_id]
            if is_stretchy:
                print(f"stretchy seg {seg_id}, smatseg {smatseg_id} has {smatseg.pgons} pgons")
            else:
                print(f"non-stretchy seg {seg_id} smatseg {smatseg_id} has {smatseg.pgons} pgons")

    # Find the origin point of every joint
    head_by_joint_id = {}
    tail_by_joint_id = {}
    parent_by_joint_id = {}
    is_connected_by_joint_id = {}
    for torso in p_torsos:
        if torso.parent == -1:
            j = torso.joint
            assert j not in head_by_joint_id
            assert j not in tail_by_joint_id
            head_by_joint_id[j] = Vector((0,0,0))
            tail_by_joint_id[j] = Vector((1,0,0))
            assert j not in parent_by_joint_id
            parent_by_joint_id[j] = -1
        else:
            j = torso.joint
            assert j in head_by_joint_id
            assert j not in tail_by_joint_id
            tail_by_joint_id[j] = head_by_joint_id[j]+Vector((1,0,0))
            assert j not in parent_by_joint_id
            parent_by_joint_id[j] = p_torsos[torso.parent].joint
        is_connected_by_joint_id[j] = False
        root = head_by_joint_id[torso.joint]
        k = torso.fixed_points
        parts = zip(
            torso.joint_id[:k],
            torso.pts[:k])
        for j, pt in parts:
            assert j not in head_by_joint_id
            head_by_joint_id[j] = root+Vector(pt)
    for limb in p_limbs:
        j = limb.joint_id[0]
        assert j in head_by_joint_id
        head = head_by_joint_id[j]
        k = limb.segments
        parts = zip(
            limb.joint_id[:k+1],
            limb.seg[:k] + [limb.seg[k-1]], # finger etc. bones, tail gets wrist vector
            limb.seg_len[:k] + [0.25])      # finger etc. bones, get fixed length
        pj = j
        for i, (j, seg, seg_len) in enumerate(parts):
            head_by_joint_id[j] = head
            tail = head+seg_len*Vector(seg)
            tail_by_joint_id[j] = tail
            head = tail
            assert j not in parent_by_joint_id
            if i==0:
                parent_by_joint_id[j] = p_torsos[limb.torso_id].joint
                is_connected_by_joint_id[j] = False
            else:
                parent_by_joint_id[j] = pj
                is_connected_by_joint_id[j] = True
            pj = j
    assert sorted(head_by_joint_id.keys())==sorted(tail_by_joint_id.keys())
    #print("Bone positions:")
    HUMAN_JOINTS = [
        'LToe',     #  0
        'RToe',     #  1
        'LAnkle',   #  2
        'RAnkle',   #  3
        'LKnee',    #  4
        'RKnee',    #  5
        'LHip',     #  6
        'RHip',     #  7
        'Butt',     #  8
        'Neck',     #  9
        'LShldr',   # 10
        'RShldr',   # 11
        'LElbow',   # 12
        'RElbow',   # 13
        'LWrist',   # 14
        'RWrist',   # 15
        'LFinger',  # 16
        'RFinger',  # 17
        'Abdomen',  # 18
        'Head',     # 19
        'LShldrIn', # 20
        'RShldrIn', # 21
        'LWeap',    # 22
        'RWeap',    # 23
        ]
    for j in sorted(head_by_joint_id.keys()):
        h = head_by_joint_id[j]
        t = tail_by_joint_id[j]
        #print(f"joint {j}: head {h.x},{h.y},{h.z}; tail {t.x},{t.y},{t.z}")
        #create_empty(f"{HUMAN_JOINTS[j]} (Joint {j})", h, context=context)
    #print()

    # Build segment/material/joint tables for later lookup
    segment_by_vert_id = {}
    smatseg_by_vert_id = {}
    material_by_vert_id = {}
    joint_by_vert_id = {}
    for mi, smatr in enumerate(p_smatrs):
        print(f"smatr {mi}: {smatr.name!r}, {smatr.pgons} pgons, {smatr.verts} verts")
        map_start = smatr.map_start
        map_end = map_start+smatr.smatsegs
        for smatseg_id in p_maps[map_start:map_end]:
            smatseg = p_smatsegs[smatseg_id]
            seg = p_segs[smatseg.seg_id]
            vert_start = smatseg.vert_start
            vert_end = vert_start+smatseg.verts
            for vi in range(vert_start, vert_end):
                assert vi not in segment_by_vert_id
                segment_by_vert_id[vi] = smatseg.seg_id
                assert  vi not in smatseg_by_vert_id
                smatseg_by_vert_id[vi] = smatseg_id
                assert vi not in material_by_vert_id
                material_by_vert_id[vi] = smatseg.smatr_id
                assert vi not in joint_by_vert_id
                joint_by_vert_id[vi] = seg.joint_id

    # Build the bare mesh
    vertices = [Vector(v) for v in p_verts]
    # Offset each vert by the joint its attached to
    assert len(joint_by_vert_id)==len(p_verts)
    for i in range(len(vertices)):
        v = vertices[i]
        j = joint_by_vert_id[i]
        head = head_by_joint_id[j]
        vertices[i] = head+v
    faces = [tuple(p.vert) for p in p_pgons]
    name = f"TEST"
    mesh = bpy.data.meshes.new(f"{name} mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=True)
    print(f"mesh vertices: {len(mesh.vertices)}, loops: {len(mesh.loops)}, polygons: {len(mesh.polygons)}")

    # BUG: if we use the collection returned from .new(), then we sometimes get
    # a RuntimeError: "bpy_prop_collection[index]: internal error, valid index
    # X given in Y sized collection, but value not found" -- so we create the
    # collections first, then look them up to use them.
    mesh.vertex_colors.new(name="SmatSegCol", do_init=False)
    mesh.vertex_colors.new(name="SegCol", do_init=False)
    mesh.vertex_colors.new(name="MatCol", do_init=False)
    mesh.vertex_colors.new(name="JointCol", do_init=False)
    mesh.vertex_colors.new(name="StretchyCol", do_init=False)
    mesh.vertex_colors.new(name="WeightCol", do_init=False)
    mesh.vertex_colors.new(name="UVCol", do_init=False)
    smatseg_colors = mesh.vertex_colors["SmatSegCol"]
    seg_colors = mesh.vertex_colors["SegCol"]
    mat_colors = mesh.vertex_colors["MatCol"]
    joint_colors = mesh.vertex_colors["JointCol"]
    stretchy_colors = mesh.vertex_colors["StretchyCol"]
    weight_colors = mesh.vertex_colors["WeightCol"]
    uv_colors = mesh.vertex_colors["UVCol"]
    for li, loop in enumerate(mesh.loops):
        vi = loop.vertex_index
        smatseg_colors.data[li].color = id_color( smatseg_by_vert_id[vi] )
        seg_colors.data[li].color = id_color( segment_by_vert_id[vi] )
        mat_colors.data[li].color = id_color( material_by_vert_id[vi] )
        joint_colors.data[li].color = id_color( joint_by_vert_id[vi] )
        seg = p_segs[ segment_by_vert_id[vi] ]
        is_stretchy = bool(seg.flags & 1)
        stretchy_colors.data[li].color = id_color(0) if is_stretchy else id_color(1)
        uvn = p_uvnorms[vi]
        uv_colors.data[li].color = (float(uvn.u),float(uvn.v),0.0,1.0)

    # Create the object
    mesh_obj = create_object(name, mesh, Vector((0,0,0)), context=context)

    print("Vertex groups:")
    weight_by_vert_id = {}
    for seg_id, seg in enumerate(p_segs):
        print(f"  seg {seg_id} flags {seg.flags} smatsegs {seg.smatsegs}")
        j = seg.joint_id
        is_stretchy = bool(seg.flags & 1)
        group_name = HUMAN_JOINTS[j]
        try:
            group = mesh_obj.vertex_groups[group_name]
        except KeyError:
            group = mesh_obj.vertex_groups.new(name=group_name)
        # TODO: do we need to do this same collection dance?
        #group = mesh_obj.vertex_groups[group_name]
        map_start = seg.map_start
        map_end = map_start+seg.smatsegs
        print(f"  map_start: {map_start}, map_end: {map_end}")
        for smatseg_id in p_maps[map_start:map_end]:
            print(f"    smatseg {smatseg_id}")
            smatseg = p_smatsegs[smatseg_id]
            for i in range(smatseg.verts):
                vi = smatseg.vert_start+i
                print(f"      vert {vi}")
                # OKAY, so this is wrong. but why?
                # by definition a 'stretchy' area should have
                # weights for _two_ different bones, right? but not
                # in this format! each smatseg belongs to _one_ segment,
                # i.e. one joint. so what delimits this?
                # from the code it looks like the parent bone transform
                # gets applied, then the child bone transform gets applied
                # _with weighting_. in blender weight terms, i am not sure
                # if this should be interpreted as vertex weights:
                #     parent bone: 1.0
                #     child bone: weight
                # or as vertex weights:
                #     parent bone: 1.0-weight
                #     child bone: weight
                # TODO: compare extreme poses in game and in blender to
                # compare how each strategy places the stretchy vertices.
                if is_stretchy:
                    wi = smatseg.weight_start+i
                    print(f"      weight {wi}")
                    weight = p_weights[wi]
                    weight_by_vert_id[vi] = weight # TODO just for weight debugging
                else:
                    weight = 1.0
                group.add([vi], weight, 'REPLACE')
    for li, loop in enumerate(mesh.loops):
        vi = loop.vertex_index
        if vi in weight_by_vert_id:
            w = weight_by_vert_id[vi]
            weight_colors.data[li].color = (w,w,w,1.0)
        else:
            weight_colors.data[li].color = (1.0,0.0,0.0,1.0)

    # Smatr vertex groups
    print("Smatr Vertex groups:")
    for smatr_id, smatr in enumerate(p_smatrs):
        group_name = f"Smatr{smatr_id}"
        try:
            group = mesh_obj.vertex_groups[group_name]
        except KeyError:
            group = mesh_obj.vertex_groups.new(name=group_name)
        for i in range(smatr.verts):
            vi = smatr.vert_start+i
            group.add([vi], 1.0, 'REPLACE')

    # Seg vertex groups
    print("Seg Vertex groups:")
    for seg_id, seg in enumerate(p_segs):
        group_name = f"Seg{seg_id}"
        try:
            group = mesh_obj.vertex_groups[group_name]
        except KeyError:
            group = mesh_obj.vertex_groups.new(name=group_name)
        map_start = seg.map_start
        map_end = map_start+seg.smatsegs
        print(f"  map_start: {map_start}, map_end: {map_end}")
        for smatseg_id in p_maps[map_start:map_end]:
            smatseg = p_smatsegs[smatseg_id]
            for i in range(smatseg.verts):
                vi = smatseg.vert_start+i
                group.add([vi], 1.0, 'REPLACE')

    # SmatSmeg vertex groups
    print("SmatSmeg Vertex groups:")
    for smatseg_id, smatseg in enumerate(p_smatsegs):
        group_name = f"SmatSmeg{smatseg_id}"
        try:
            group = mesh_obj.vertex_groups[group_name]
        except KeyError:
            group = mesh_obj.vertex_groups.new(name=group_name)
        for i in range(smatseg.verts):
            vi = smatseg.vert_start+i
            group.add([vi], 1.0, 'REPLACE')

    # Create an armature for it
    arm_obj = create_armature("Human", Vector((0,0,0)), context=context, display_type='WIRE')
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    edit_bones = arm_obj.data.edit_bones
    bones_by_joint_id = {}
    for j in sorted(head_by_joint_id.keys()):
        bone_name = HUMAN_JOINTS[j]
        #print(j, bone_name, head_by_joint_id[j], tail_by_joint_id[j])
        b = edit_bones.new(bone_name)
        b.head = head_by_joint_id[j]
        b.tail = tail_by_joint_id[j]
        # TODO -- roll is all set to zero, but some of the bones are
        # getting created _not_ actually flat in world space??
        # BUT: check if that matches the imported skeleton anyway? with motions
        # etc, and see if they seem to match e.g. sword roll angle.
        # may need to use .align_roll()?
        # TODO: check extreme poses in game versus in blender to see if
        # a difference in roll is visible.
        b.roll = 0.0
        bones_by_joint_id[j] = b
        # TODO -- should we not create bones for FINGER, TOE?? they dont get
        # their own transforms... (i dont think - can experiment with that
        # when doing motion export). BUT they can be used as anchor points for
        # detailattachements and so on, so their positions are important!
        # SO: should create bones for them regardless (so that their
        # positions are clear), but they need to be non-deform bones. also
        # might want to change the appearance of them?
    for j in sorted(head_by_joint_id.keys()):
        bone_name = HUMAN_JOINTS[j]
        b = edit_bones[bone_name]
        pj = parent_by_joint_id[j]
        if pj != -1:
            b.use_connect = is_connected_by_joint_id[j]
            b.parent = bones_by_joint_id[pj]
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add an armature modifier
    arm_mod = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
    arm_mod.object = arm_obj
    arm_mod.use_vertex_groups = True

    context.view_layer.objects.active = mesh_obj
    return mesh_obj

#---------------------------------------------------------------------------#
# Export

def do_export_mesh(context, mesh_obj, bin_filename):
    cal_filename = os.path.splitext(bin_filename)[0]+'.cal'
    dump_filename = os.path.splitext(bin_filename)[0]+'.export_dump'
    dumpf = open(dump_filename, 'w')

    # TODO: note that if we import the spider this way, the 'sac' thats in the
    # .e and in the .map will just be part of the 'base' vertex group. and
    # that is exactly how it should be! not relevant here ofc, just noting why
    # it is not listed.
    SPIDER_JOINTS = [
        'base',     # 0
        'lmand',    # 1
        'lmelbow',  # 2
        'rmand',    # 3
        'rmelbow',  # 4
        'r1shldr',  # 5
        'r1elbow',  # 6
        'r1wrist',  # 7
        'r2shldr',  # 8
        'r2elbow',  # 9
        'r2wrist',  # 10
        'r3shldr',  # 11
        'r3elbow',  # 12
        'r3wrist',  # 13
        'r4shldr',  # 14
        'r4elbow',  # 15
        'r4wrist',  # 16
        'l1shldr',  # 17
        'l1elbow',  # 18
        'l1wrist',  # 19
        'l2shldr',  # 20
        'l2elbow',  # 21
        'l2wrist',  # 22
        'l3shldr',  # 23
        'l3elbow',  # 24
        'l3wrist',  # 25
        'l4shldr',  # 26
        'l4elbow',  # 27
        'l4wrist',  # 28
        'r1finger', # 29
        'r2finger', # 30
        'r3finger', # 31
        'r4finger', # 32
        'l1finger', # 33
        'l2finger', # 34
        'l3finger', # 35
        'l4finger', # 36
        'ltip',     # 37
        'rtip',     # 38
        ]
    SPIDER_JOINT_INDICES = {name: j
        for j, name in enumerate(SPIDER_JOINTS)}

    arm_mod = [mod for mod in mesh_obj.modifiers if mod.type == 'ARMATURE'][0]
    arm_obj = arm_mod.object
    arm = arm_obj.data.copy()
    arm.pose_position = 'REST'
    mesh = mesh_obj.data.copy()
    mesh.calc_loop_triangles()
    # TODO: smoothing amount?
    mesh.calc_normals_split()

    # TODO: verify that the armature matches the hardcoded skeleton!
    #       (optionally omitting trailing chains of bones)

    bone_head_pos = [(0.0,0.0,0.0) for j in SPIDER_JOINTS]
    bone_directions = [(1.0,0.0,0.0) for j in SPIDER_JOINTS]
    bone_lengths = [(1.0,0.0,0.0) for j in SPIDER_JOINTS]
    for b in arm.bones:
        ji = SPIDER_JOINTS.index(b.name)
        bone_head_pos[ji] = tuple(Vector(arm_obj.location)+Vector(b.head_local))
        d = (Vector(b.tail_local)-Vector(b.head_local)).normalized()
        bone_directions[ji] = tuple(d)
        bone_lengths[ji] = b.length

    vertex_group_joint_ids = [
        SPIDER_JOINT_INDICES.get(g.name.lower(), 0)
        for g in mesh_obj.vertex_groups]

    # TODO: ensure one uv layer

    # Vectors are all kinds of weird (only sometimes hashable), so
    # just keep everything as tuples unless we are doing maths on it!

    # TODO: actually get actual material info from the object!
    material_temp = []
    material_temp.append("fake")
    # Vertex attributes, one entry per split vertex. For now these are not in
    # final sorted order, so we call these indices u[nsorted]v[ertex]i[index].
    vertex_pos = []
    vertex_normal = []
    vertex_uv = []
    vertex_material_id = []
    vertex_joint_id = []
    vertex_weight = []
    # Polygon attributes, one entry per triangle.
    poly_vertex_ids = []
    poly_material_id = []
    poly_normal = []
    poly_distance = []
    # Lookup table for keeping track of split vertices.
    vertex_tuples = {}
    # Cache some attribute lookups.
    mesh_vertices = mesh.vertices
    mesh_uvloops = mesh.uv_layers[0].data
    for tri in mesh.loop_triangles:
        tri_vertex_ids = []
        for i in range(3):
            # Gather vertex attributes.
            vertex_index = tri.vertices[i]
            mesh_vert = mesh_vertices[vertex_index]
            world_pos = tuple(mesh_vert.co)
            normal = tuple(tri.split_normals[i])
            uv = tuple(mesh_uvloops[ tri.loops[i] ].uv)
            material_id = tri.material_index # TODO: handle materials!

            # Split a vertex if it has multiple materials, uvs or split normals.
            tup = (vertex_index, uv, material_id, normal)
            uvi = vertex_tuples.get(tup, -1)
            if uvi==-1:
                uvi = len(vertex_pos)
                # Determine the joint for this vertex from its vertex groups.
                group_count = len(mesh_vert.groups)
                if group_count == 0:
                    joint_id = 0
                    weight = 1.0
                elif group_count == 1:
                    g = mesh_vert.groups[0]
                    joint_id = vertex_group_joint_ids[g.group]
                    weight = g.weight # TODO: double-check if this is correct weight handling!
                elif group_count == 2:
                    g0 = mesh_vert.groups[0]
                    g1 = mesh_vert.groups[1]
                    # TODO: only allow this if one group is the parent (with weight 1.0)
                    #       and the other is the child (with whatever weight); then this
                    #       will be a stretchy vertex!
                    raise NotImplementedError("Stretchy vertices not yet implemented")
                else:
                    raise ValueError(f"Vertex {vertex_index} is in more than 2 vertex groups")
                # vertex positions must be relative to the bone they belong to.
                # (TODO: confirm they are in local-world-space, not bone-space)
                pos = Vector(world_pos)-Vector(bone_head_pos[ji])
                vertex_pos.append(pos)
                vertex_normal.append(normal)
                vertex_uv.append(uv)
                vertex_material_id.append(material_id) # TODO: handle materials!
                vertex_joint_id.append(joint_id)
                vertex_weight.append(weight)
                vertex_tuples[tup] = uvi
            # Add this vertex to the polygon
            tri_vertex_ids.append(uvi)
        # Gather polygon attributes.
        material_id = tri.material_index # TODO: handle materials!
        normal = tuple(tri.normal)
        distance = tri.center.length
        poly_vertex_ids.append(tuple(tri_vertex_ids))
        poly_material_id.append(material_id)
        poly_normal.append(normal)
        poly_distance.append(distance)
    del vertex_tuples
    del mesh_vertices
    del mesh_uvloops

    # From here on out, we are working with our arrays of data, and don't use
    # any more Blender objects (except for Vectors that we construct).

    print(file=dumpf)
    print("MATERIALS:", file=dumpf)
    for mi, (name) \
    in enumerate(material_temp):
        print(f"  {mi}: name {name}", file=dumpf)
    print("VERTICES:", file=dumpf)
    for vi, (pos, normal, uv, material_id, joint_id, weight) \
    in enumerate(zip(vertex_pos, vertex_normal, vertex_uv, vertex_material_id, vertex_joint_id, vertex_weight)):
        print(f"  {vi}: mat {material_id}; joint_id {joint_id}", file=dumpf)
        #print(f"  {vi}: pos {pos}; normal {normal}; uv {uv}; joint_id {joint_id}; weight {weight}", file=dumpf)
    print("TRIANGLES:", file=dumpf)
    for pi, (vertex_ids, material_id, normal, distance) \
    in enumerate(zip(poly_vertex_ids, poly_material_id, poly_normal, poly_distance)):
        print(f"  {pi}: vertex_ids {vertex_ids}; material_id {material_id}; normal {normal}; distance {distance}", file=dumpf)
    print(file=dumpf)

    print(f"{len(material_temp)} materials", file=dumpf)
    print(f"{len(vertex_pos)} vertices", file=dumpf)
    print(f"{len(poly_vertex_ids)} triangles", file=dumpf)
    print(file=dumpf)

    dumpf.flush()

    # Vertices need to be sorted by material, then joint id, so that all the
    # verts in a smatr and in a smatseg are contiguous.
    #
    # TODO: this is not taking into account stretchy vertices, which need their
    #       own sort flag!
    #
    # Sort all the vertex attributes together.
    unsorted_vertex_ids = list(range(len(vertex_pos)))
    temp = sorted(zip(vertex_material_id, vertex_joint_id, unsorted_vertex_ids,
        vertex_pos, vertex_normal, vertex_uv, vertex_weight))
    (vertex_material_id, vertex_joint_id, unsorted_vertex_ids,
        vertex_pos, vertex_normal, vertex_uv, vertex_weight) = zip(*temp)

    # Rewrite the polygon vertex ids with the sorted ids.
    vi_from_uvi = [-1]*len(unsorted_vertex_ids)
    for vi, uvi in enumerate(unsorted_vertex_ids):
        vi_from_uvi[uvi] = vi
    for pi in range(len(poly_vertex_ids)):
        poly_vertex_ids[pi] = tuple(vi_from_uvi[uvi] for uvi in poly_vertex_ids[pi])

    # Polygons need to be sorted by material, so that all pgons in a smatr are
    # also contiguous.
    # TODO: stretchy smatsegs also need to reference their pgons in a contiguous
    #       group! this implies: no poly can be part of two stretchy smatsegs;
    #       (although a stretchy joint can have multiple smatsegs; see mecsub03.bin
    #       for an example where the neck and breathing tube are different materials
    #       but both part of the stretchy neck segment). note however that you
    #       _could_ allow polys to be in multiple stretchy smatsegs if you kept
    #       subdividing them into more smatsegs until the polys were contiguous...
    unsorted_poly_ids = list(range(len(poly_vertex_ids)))
    temp = sorted(zip(poly_material_id, unsorted_poly_ids,
        poly_vertex_ids, poly_normal, poly_distance))
    (poly_material_id, unsorted_poly_ids,
        poly_vertex_ids, poly_normal, poly_distance) = zip(*temp)

    # TODO: more figuring out what needs to be done to properly output stretchy
    #       smatsegs!

    # TODO: could compact the normals table here, to eliminate duplicates.
    #       probably only a tiny space savings though.

    # TEMP: test the grouping looks okay?
    vi = 0; vi_end = len(vertex_pos)
    pi = 0; pi_end = len(poly_vertex_ids)
    smatseg_id = 0
    mi = 0
    ji = 0
    while True:
        print(f"smatsmeg {smatseg_id}:", file=dumpf)
        print(f"  start at vi {vi}, pi {pi}", file=dumpf)
        # Advance to the next change in vertex material/joint:
        while (vi<vi_end
        and vertex_material_id[vi]==mi
        and vertex_joint_id[vi]==ji):
            vi += 1

        # Advance to the next change in polygon material/joint:
        # TODO: need to sort polys by material (and joint??) first!
        # while (pi<pi_end
        # and poly_material_id[vi]==mi
        # and poly_joint_id[vi]==ji):
        #     vi += 1
        # TODO: rather than this weird awkward doublestep, maybe we
        # could store poly indices with the verts? that way the polys would
        # get the right sorting order too?? unsure.
        print(f"    end at vi {vi}, pi {pi}", file=dumpf)
        if vi==vi_end: break
        # Next smatseg begins with the next material/joint
        mi = vertex_material_id[vi]
        ji = vertex_joint_id[vi]
        smatseg_id += 1
        if smatseg_id>10000: raise RuntimeError("Whoa boy, you screwed up!")
    print(file=dumpf)
    dumpf.flush()

    # From here we build up the file-specific data formats.

    vert_count = len(vertex_pos)
    pgon_count = len(poly_vertex_ids)
    verts = [
        LGVector(vertex_pos[vi])
        for vi in range(vert_count)
        ]
    uvnorms = [
        LGMMUVNorm(values=(
            vertex_uv[vi][0],
            vertex_uv[vi][1],
            pack_normal(vertex_normal[vi])))
        for vi in range(vert_count)
        ]
    pgons = [
        LGMMPolygon(values=(
            poly_vertex_ids[pi],
            poly_material_id[pi],
            poly_distance[pi],
            pi,
            0))
        for pi in range(pgon_count)
        ]
    norms = [
        LGVector(poly_normal[pi])
        for pi in range(pgon_count)
        ]

    print("VERTS:", file=dumpf)
    for i, v in enumerate(verts):
        print(f"  {i}: {v.x},{v.y},{v.z}", file=dumpf)

    print("UVNORMS:", file=dumpf)
    for i, uvn in enumerate(uvnorms):
        print(f"  {i}: {uvn.u},{uvn.v},{uvn.norm:08x}", file=dumpf)

    print("NORMS:", file=dumpf)
    for i, n in enumerate(norms):
        print(f"  {i}: {n.x},{n.y},{n.z}", file=dumpf)

    print("PGONS:", file=dumpf)
    for i, p in enumerate(pgons):
        print(f"  {i}:", file=dumpf)
        print(f"    vert0: {p.vert[0]}", file=dumpf)
        print(f"    vert1: {p.vert[1]}", file=dumpf)
        print(f"    vert2: {p.vert[2]}", file=dumpf)
        print(f"    smatr_id: {p.smatr_id}", file=dumpf)
        print(f"    d: {p.d}", file=dumpf)
        print(f"    norm: {p.norm}", file=dumpf)

    raise NotImplementedError("rewrite in progress up to here")

    # HERE: ...

    # TEMP: test the grouping looks okay?
    smatsegs = []
    grouped_vis = itertools.groupby(range(vert_count),
        key=lambda vi: (vertex_material_id[vi], vertex_joint_id[vi]))
    for (mi, ji), group in grouped_vis:
        smatseg_id = len(smatsegs)

        smatseg = LGMMSmatSeg()
        smatseg.pgons = uint16(0)
        smatseg.pgon_start = uint16(0)
        smatseg.verts = uint16(len(smatseg_source_verts))
        smatseg.vert_start = uint16(smatseg_source_verts[0].vert_id)
        smatseg.weight_start = uint16(0) # TODO: smatsegs of stretchy segs need weights
        smatseg.smatr_id = uint16(0) # TODO: do materials
        smatseg.seg_id = uint16(smatseg_id) # TODO: if there are multiple materials, or


    mi = 0
    ji = 0
    while True:
        print(f"smatsmeg {smatseg_id}:", file=dumpf)
        print(f"  start at vi {vi}, pi {pi}", file=dumpf)
        # Advance to the next change in vertex material/joint:
        while (vi<vi_end
        and vertex_material_id[vi]==mi
        and vertex_joint_id[vi]==ji):
            vi += 1

        # Advance to the next change in polygon material/joint:
        # TODO: need to sort polys by material (and joint??) first!
        # while (pi<pi_end
        # and poly_material_id[vi]==mi
        # and poly_joint_id[vi]==ji):
        #     vi += 1
        # TODO: rather than this weird awkward doublestep, maybe we
        # could store poly indices with the verts? that way the polys would
        # get the right sorting order too?? unsure.
        print(f"    end at vi {vi}, pi {pi}", file=dumpf)
        if vi==vi_end: break
        # Next smatseg begins with the next material/joint
        mi = vertex_material_id[vi]
        ji = vertex_joint_id[vi]
        smatseg_id += 1
        if smatseg_id>10000: raise RuntimeError("Whoa boy, you screwed up!")
    print(file=dumpf)
    dumpf.flush()



    smatsegs = []
    for smatseg_id, (sort_key, smatseg_source_verts) \
    in enumerate(itertools.groupby(source_verts, key=SourceVert.sort_key)):
        temp_mat_id = sort_key[0]
        group_id = sort_key[1]
        group_name = mesh_obj.vertex_groups[group_id].name
        bone_head = bone_head_pos[group_name]
        smatseg_source_verts = list(smatseg_source_verts)
        print(f"Smatseg {smatseg_id}: material {temp_mat_id}, vertex group {group_id}", file=dumpf)
        smatseg = LGMMSmatSeg()
        smatseg.pgons = uint16(0)
        smatseg.pgon_start = uint16(0)
        smatseg.verts = uint16(len(smatseg_source_verts))
        smatseg.vert_start = uint16(smatseg_source_verts[0].vert_id)
        smatseg.weight_start = uint16(0) # TODO: smatsegs of stretchy segs need weights
        smatseg.smatr_id = uint16(0) # TODO: do materials
        smatseg.seg_id = uint16(smatseg_id) # TODO: if there are multiple materials, or
                                    # stretchy segs, then this cannot be 1:1
        smatsegs.append(smatseg)
        for source_vert in smatseg_source_verts:
            vi = source_vert.vert_id
            pos = Vector((verts[vi].x,verts[vi].y,verts[vi].z))
            local_pos = pos - bone_head
            verts[vi] = LGVector(values=local_pos)
            # TODO: do we want to use undeformed_co? or do we want to ensure that
            # the armature is in rest pose for the export? (cause there might
            # reasonably be other modifiers that we _do_ want to have affecting
            # the mesh!
            pos = source_vert.mesh_vert.co
            print(f"  {vi}: mesh_vert_id {source_vert.mesh_vert_id} at {pos[0]},{pos[1]},{pos[2]}", file=dumpf)


    # TODO: if there are multiple materials, or
    # stretchy segs, then this cannot be 1:1
    segs = []
    for seg_id, (sort_key, seg_source_verts) \
    in enumerate(itertools.groupby(source_verts, key=SourceVert.sort_key)):
        temp_mat_id = sort_key[0]
        group_id = sort_key[1]
        group_name = mesh_obj.vertex_groups[group_id].name
        smatseg_source_verts = list(smatseg_source_verts)
        joint_id = SPIDER_JOINTS.index(group_name)
        seg = LGMMSegment()
        seg.joint_id = uint8(joint_id)
        seg.smatsegs = uint8(1) # TODO: fix this when we do materials and stretchies
        seg.map_start = uint8(seg_id) # TODO: fix this when we do materials and stretchies
        seg.flags = uint8(0) # TODO: fix this for stretchies
        seg.pgons = uint16(0) # no stretchies and only hardware rendering, so we ignore this
        seg.pgon_start = uint16(0) # ditto
        seg.verts = uint16(0) # No verts for material-first layout
        seg.vert_start = uint16(0)
        seg.weight_start = uint16(0) # TODO: support weights for stretchies
        segs.append(seg)

    smatrs = []
    # TODO: dont hardcode 1 material!
    smatr = LGMMSMatrV2()
    smatr.name = bytes16(b'face.png\x00\x00\x00\x00\x00\x00\x00\x00')
    smatr.caps = uint32(0)
    smatr.alpha = float32(1.0) # TODO: i think this is 1.0 for opaque?
    smatr.self_illum = float32(0.0)
    smatr.for_rent = uint32(0.0)
    smatr.handle = uint32(0)
    smatr.uv = float32(1.0) # TODO: i think this is uv scale?
    smatr.mat_type = uint8(0)
    smatr.smatsegs = uint8(len(smatsegs))
    smatr.map_start = uint8(0) # maps for materials are just smatseg indices in order
    smatr.flags = uint8(0) # TODO: what are these flags?
    smatr.pgons = uint16(len(pgons))
    smatr.pgon_start = uint16(0)
    smatr.verts = uint16(len(verts))
    smatr.vert_start = uint16(0)
    smatr.weight_start = uint16(0)
    smatrs.append(smatr)

    maps = [uint8(i) for i in range(len(smatsegs))]

    print("Segments", file=dumpf)
    for i, seg in enumerate(segs):
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
    for i, smatr in enumerate(smatrs):
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
    for i, smatseg in enumerate(smatsegs):
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

    dumpf.close()

    header = LGMMHeader()
    header.segs = uint8(len(segs))
    header.smatrs = uint8(len(smatrs))
    header.smatsegs = uint8(len(smatsegs))
    header.pgons = uint16(len(pgons))
    header.verts = uint16(len(verts))
    header.weights = uint16(0) # TODO: support weights

    # TODO: account for alignment?
    offset = 0
    offset += LGMMHeader.size()
    header.map_off = uint32(offset)
    offset += len(maps)*uint8.size()
    header.seg_off = uint32(offset)
    offset += len(segs)*LGMMSegment.size()
    header.smatr_off = uint32(offset)
    offset += len(smatrs)*LGMMSMatrV2.size()
    header.smatseg_off = uint32(offset)
    offset += len(smatsegs)*LGMMSmatSeg.size()
    header.pgon_off = uint32(offset)
    offset += len(pgons)*LGMMPolygon.size()
    header.norm_off = uint32(offset)
    offset += len(norms)*LGVector.size()
    header.vert_vec_off = uint32(offset)
    offset += len(verts)*LGVector.size()
    header.vert_uvn_off = uint32(offset)
    offset += len(uvnorms)*LGMMUVNorm.size()
    header.weight_off = uint32(offset)
    ## TODO: support weights
    # offset += len(weights)*float32.size()
    file_size = offset

    with open(bin_filename, 'wb') as f:
        header.write(f)
        for m in maps: m.write(f)
        for seg in segs: seg.write(f)
        for smatr in smatrs: smatr.write(f)
        for smatseg in smatsegs: smatseg.write(f)
        for pgon in pgons: pgon.write(f)
        for norm in norms: norm.write(f)
        for vert in verts: vert.write(f)
        for uvnorm in uvnorms: uvnorm.write(f)
        # for w in weights: w.write(f)

    # TODO: do a proper export of the .cal (probably means harcoding knowledge
    # of torsos/limbs of all the stock skeletons...
    def make_torso(joint_id, parent_id, *bone_names):
        limb_joint_ids = [-1 for _ in range(16)]
        limb_pts = [LGVector() for _ in range(16)]
        for i, name in enumerate(bone_names):
            limb_joint_ids[i] = SPIDER_JOINTS.index(name)
            limb_pts[i] = bone_head_pos[name]
        return LGCALTorso((
            joint_id,  # joint
            parent_id, # parent
            len(bone_names),  # fixed_points
            Array(int32, 16)(limb_joint_ids), # joint_id
            Array(LGVector, 16)(limb_pts), # pts
            ))
    def make_limb(torso_id, bend, *bone_names):
        limb_joint_ids = [-1 for _ in range(17)]
        limb_segs = [LGVector() for _ in range(16)]
        limb_lengths = [0.0 for _ in range(16)]
        for i, name in enumerate(bone_names):
            limb_joint_ids[i] = SPIDER_JOINTS.index(name)
        for i, name in enumerate(bone_names[:-1]):
            limb_segs[i] = bone_directions[name]
            limb_lengths[i] = bone_lengths[name]
        return LGCALLimb((
            torso_id,  # torso_id
            bend,  # bend (TODO: maybe wrong for this abuse of the skeleton?)
            len(bone_names)-1,  # segments
            Array(int16, 17)(limb_joint_ids), # joint_id
            Array(LGVector, 16)(limb_segs), # seg
            Array(float32, 16)(limb_lengths), # seg_len
            ))
    # NOTE: hardcoded spider skeleton (because its hardcoded in the game)
    torsos = [
        make_torso(0, -1, 'lmand', 'rmand', 'r1shldr'),
        make_torso(0, -1, 'r2shldr', 'r3shldr', 'r4shldr'),
        make_torso(0, -1, 'l1shldr', 'l2shldr', 'l3shldr'),
        make_torso(0, -1, 'l4shldr'),
        ]
    limbs = [
        make_limb(0, 1, 'rmand', 'rmelbow', 'rtip'),
        make_limb(0, 1, 'lmand', 'lmelbow', 'ltip'),
        make_limb(0, 1, 'r1shldr', 'r1elbow', 'r1wrist', 'r1finger'),
        make_limb(0, 1, 'r2shldr', 'r2elbow', 'r2wrist', 'r2finger'),
        make_limb(0, 1, 'r3shldr', 'r3elbow', 'r3wrist', 'r3finger'),
        make_limb(0, 1, 'r4shldr', 'r4elbow', 'r4wrist', 'r4finger'),
        make_limb(0, 1, 'l1shldr', 'l1elbow', 'l1wrist', 'l1finger'),
        make_limb(0, 1, 'l2shldr', 'l2elbow', 'l2wrist', 'l2finger'),
        make_limb(0, 1, 'l3shldr', 'l3elbow', 'l3wrist', 'l3finger'),
        make_limb(0, 1, 'l4shldr', 'l4elbow', 'l4wrist', 'l4finger'),
        ]
    header = LGCALHeader((
        1,  # version
        len(torsos), # torsos
        len(limbs), # limbs
        ))
    footer = LGCALFooter((1.0,))

    with open(cal_filename, 'wb') as f:
        header.write(f)
        for t in torsos: t.write(f)
        for l in limbs: l.write(f)
        footer.write(f)

    print("done.")

#---------------------------------------------------------------------------#
# Armatures

# TODO: derive the stock skeletons from the rest pose in the appropriate .cal!
SKELETONS = {
    # Joints, in order by id:
    # (name, parent, connected, head_pos, tail_pos, limb_end)

    "HUMANOID": [
        ('LToe',     2, True,  ( 0.884971, 0.245466,-3.349379), ( 1.132570, 0.241114,-3.383671), True),
        ('RToe',     3, True,  ( 0.884971,-0.363442,-3.332439), ( 1.133693,-0.370478,-3.356683), True),
        ('LAnkle',   4, True,  ( 0.108441, 0.259114,-3.241831), ( 0.884971, 0.245466,-3.349379), False),
        ('RAnkle',   5, True,  ( 0.082902,-0.340752,-3.254256), ( 0.884971,-0.363442,-3.332439), False),
        ('LKnee',    6, True,  ( 0.187853, 0.270482,-1.811662), ( 0.108441, 0.259114,-3.241831), False),
        ('RKnee',    7, True,  ( 0.261306,-0.335208,-1.776900), ( 0.082902,-0.340752,-3.254256), False),
        ('LHip',     8, False, ( 0.294860, 0.223039,-0.634390), ( 0.187853, 0.270482,-1.811662), False),
        ('RHip',     8, False, ( 0.291799,-0.312492,-0.590255), ( 0.261306,-0.335208,-1.776900), False),
        ('Butt',    -1, False, ( 0.000000, 0.000000, 0.000000), ( 1.000000, 0.000000, 0.000000), False),
        ('Neck',    18, False, ( 0.085717, 0.097775, 1.827333), ( 0.142746, 0.055724, 2.665782), False),
        ('LShldr',  18, False, ( 0.197217, 0.630274, 1.443415), ( 0.224013, 1.648588, 1.285926), False),
        ('RShldr',  18, False, ( 0.276273,-0.583319, 1.471145), ( 0.026511,-1.489041, 1.286486), False),
        ('LElbow',  10, True,  ( 0.224013, 1.648588, 1.285926), ( 0.574695, 2.420023, 1.343401), False),
        ('RElbow',  11, True,  ( 0.026511,-1.489041, 1.286486), ( 0.363334,-2.383466, 1.213473), False),
        ('LWrist',  12, True,  ( 0.574695, 2.420023, 1.343401), ( 0.514888, 2.927348, 1.200230), False),
        ('RWrist',  13, True,  ( 0.363334,-2.383466, 1.213473), ( 1.306753,-5.192559, 0.963154), False),
        ('LFinger', 14, True,  ( 0.514888, 2.927348, 1.200230), ( 0.486705, 3.166417, 1.132763), True),
        ('RFinger', 15, True,  ( 1.306753,-5.192559, 0.963154), ( 1.386063,-5.428709, 0.942111), True),
        ('Abdomen',  8, False, ( 0.090587, 0.041403, 0.487539), ( 1.090587, 0.041403, 0.487539), False),
        ('Head',     9, True,  ( 0.142746, 0.055724, 2.665782), ( 0.159690, 0.043230, 2.914894), True),
        ],

    "PLAYER_ARM": [
        ('Butt',   -1, False, ( 0.000000, 0.000000, 0.000000), ( 1.000000, 0.000000, 0.000000), False),
        ('Shldr',   0, False, ( 0.276273,-0.583319, 1.471145), ( 0.026511,-1.489041, 1.286486), False),
        ('Elbow',   1, True,  ( 0.026511,-1.489041, 1.286486), ( 0.363334,-2.383466, 1.213473), False),
        ('Wrist',   2, True,  ( 0.363334,-2.383466, 1.213473), (-0.441815,-4.966849, 3.078860), False),
        ('Finger',  3, True,  (-0.441815,-4.966849, 3.078860), (-0.503060,-5.163358, 3.220753), True),
        ],

    "PLAYER_BOWARM": [
        ('Butt',   -1, False, ( 0.000000, 0.000000, 0.000000), ( 1.000000, 0.000000, 0.000000), False),
        ('Shldr',   0, False, ( 0.483372, 0.705298, 1.615229), ( 1.483372, 0.705298, 1.615229), False),
        ('Elbow',   1, False, ( 0.250679, 1.844825, 1.438993), ( 1.250679, 1.844825, 1.438993), False),
        ('Wrist',   2, False, ( 0.643103, 2.708087, 1.503309), ( 1.643103, 2.708087, 1.503309), False),
        ('TopMid',  3, False, ( 0.626592, 3.087608, 2.083993), ( 0.674318, 2.914177, 3.885759), False),
        ('Top',     4, True,  ( 0.674318, 2.914177, 3.885759), ( 0.680907, 2.890232, 4.134522), True),
        ('BotMid',  3, False, ( 0.600050, 3.075381, 0.839002), ( 0.565506, 3.006636,-0.365237), False),
        ('BotTom',  6, True,  ( 0.565506, 3.006636,-0.365237), ( 0.558349, 2.992394,-0.614728), True),
        ],

    "BURRICK": [
        ('LToe', 2, True, (1.335589,0.972176,-3.005818), (1.569843,0.953108,-3.091031), True),
        ('RToe', 3, True, (1.349016,-1.180253,-2.989248), (1.584376,-1.171291,-3.073065), True),
        ('LAnkle', 4, True, (0.797779,1.015953,-2.810183), (1.335589,0.972176,-3.005818), False),
        ('RAnkle', 5, True, (0.695107,-1.205152,-2.756373), (1.349016,-1.180253,-2.989248), False),
        ('LKnee', 6, True, (1.137058,0.935009,-1.344652), (0.797779,1.015953,-2.810183), False),
        ('RKnee', 7, True, (0.940370,-1.163753,-1.223413), (0.695107,-1.205152,-2.756373), False),
        ('LHip', 8, False, (0.583813,0.718669,-0.323938), (1.137058,0.935009,-1.344652), False),
        ('RHip', 8, False, (0.339136,-0.940503,-0.149914), (0.940370,-1.163753,-1.223413), False),
        ('Butt', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Neck', 18, False, (2.404610,0.065632,0.279207), (5.143918,-0.026617,1.288440), False),
        ('LShldr', 18, False, (2.039973,0.586115,-0.126543), (1.890414,0.872405,-1.038573), False),
        ('RShldr', 18, False, (2.008532,-0.477547,0.021423), (1.809595,-1.119616,-0.668464), False),
        ('LElbow', 10, True, (1.890414,0.872405,-1.038573), (2.375596,0.482246,-0.773101), False),
        ('RElbow', 11, True, (1.809595,-1.119616,-0.668464), (2.445686,-0.724941,-0.667395), False),
        ('LWrist', 12, True, (2.375596,0.482246,-0.773101), (2.902885,-0.039512,-0.897464), False),
        ('RWrist', 13, True, (2.445686,-0.724941,-0.667395), (3.002772,-0.591466,-1.044919), False),
        ('LFinger', 14, True, (2.902885,-0.039512,-0.897464), (3.078146,-0.212934,-0.938800), True),
        ('RFinger', 15, True, (3.002772,-0.591466,-1.044919), (3.205773,-0.542827,-1.182488), True),
        ('Abdomen', 8, False, (0.907647,0.026340,0.317664), (1.907647,0.026340,0.317664), False),
        ('Head', 9, True, (5.143918,-0.026617,1.288440), (5.378386,-0.034514,1.374824), True),
        ('Tail', 8, False, (-4.658150,0.000000,-1.001281), (-3.658150,0.000000,-1.001281), True),
        ],

    "SPIDER": [
        ('Base', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('LMand', 0, False, (-0.598144,-0.189434,-0.134901), (-0.752332,-0.252489,-0.005184), False),
        ('LMElbow', 1, True, (-0.752332,-0.252489,-0.005184), (-0.945764,-0.099278,-0.167966), False),
        ('RMand', 0, False, (-0.596624,0.189012,-0.132310), (-0.751041,0.259943,-0.005947), False),
        ('RMElbow', 3, True, (-0.751041,0.259943,-0.005947), (-0.953029,0.090079,-0.167966), False),
        ('R1Shldr', 0, False, (-0.359804,0.535749,0.005395), (-0.799312,1.151136,0.796658), False),
        ('R1Elbow', 5, True, (-0.799312,1.151136,0.796658), (-1.603009,2.397810,0.151402), False),
        ('R1Wrist', 6, True, (-1.603009,2.397810,0.151402), (-1.858665,2.942651,-1.110226), False),
        ('R2Shldr', 0, False, (-0.150710,0.561875,0.005395), (-0.378542,1.282958,0.796658), False),
        ('R2Elbow', 8, True, (-0.378542,1.282958,0.796658), (-0.757660,2.716972,0.151402), False),
        ('R2Wrist', 9, True, (-0.757660,2.716972,0.151402), (-0.830254,3.422897,-1.110226), False),
        ('R3Shldr', 0, False, (0.053001,0.579420,0.005395), (0.123963,1.321691,0.796658), False),
        ('R3Elbow', 11, True, (0.123963,1.321691,0.796658), (0.118969,2.805266,0.151402), False),
        ('R3Wrist', 12, True, (0.118969,2.805266,0.151402), (0.109922,3.456340,-1.110226), False),
        ('R4Shldr', 0, False, (0.271775,0.571460,0.005395), (0.671226,1.196353,0.796658), False),
        ('R4Elbow', 14, True, (0.671226,1.196353,0.796658), (1.108372,2.618391,0.151402), False),
        ('R4Wrist', 15, True, (1.108372,2.618391,0.151402), (1.304304,3.255024,-1.110226), False),
        ('L1Shldr', 0, False, (-0.352239,-0.547810,0.005395), (-0.783112,-1.169273,0.796657), False),
        ('L1Elbow', 17, True, (-0.783112,-1.169273,0.796657), (-1.569325,-2.427047,0.151402), False),
        ('L1Wrist', 18, True, (-1.569325,-2.427047,0.151402), (-1.867067,-3.127483,-1.110226), False),
        ('L2Shldr', 0, False, (-0.142800,-0.571014,0.005395), (-0.360542,-1.295208,0.796658), False),
        ('L2Elbow', 20, True, (-0.360542,-1.295208,0.796658), (-0.719601,-2.734375,0.151402), False),
        ('L2Wrist', 21, True, (-0.719601,-2.734375,0.151402), (-0.767832,-3.389762,-1.110226), False),
        ('L3Shldr', 0, False, (0.061135,-0.585713,0.005395), (0.142454,-1.326921,0.796658), False),
        ('L3Elbow', 23, True, (0.142454,-1.326921,0.796658), (0.158174,-2.810421,0.151402), False),
        ('L3Wrist', 24, True, (0.158174,-2.810421,0.151402), (0.156040,-3.443302,-1.110226), False),
        ('L4Shldr', 0, False, (0.279777,-0.574699,0.005395), (0.687914,-1.193954,0.796658), False),
        ('L4Elbow', 26, True, (0.687914,-1.193954,0.796658), (1.144872,-2.609750,0.151402), False),
        ('L4Wrist', 27, True, (1.144872,-2.609750,0.151402), (1.341830,-3.219061,-1.110226), False),
        ('R1Finger', 7, True, (-1.858665,2.942651,-1.110226), (-1.904389,3.040096,-1.335867), True),
        ('R2Finger', 10, True, (-0.830254,3.422897,-1.110226), (-0.842792,3.544817,-1.328121), True),
        ('R3Finger', 13, True, (0.109922,3.456340,-1.110226), (0.108329,3.570986,-1.332383), True),
        ('R4Finger', 16, True, (1.304304,3.255024,-1.110226), (1.338638,3.366583,-1.331304), True),
        ('L1Finger', 19, True, (-1.867067,-3.127483,-1.110226), (-1.917586,-3.246329,-1.324290), True),
        ('L2Finger', 22, True, (-0.767832,-3.389762,-1.110226), (-0.776308,-3.504943,-1.331950), True),
        ('L3Finger', 25, True, (0.156040,-3.443302,-1.110226), (0.155662,-3.555398,-1.333686), True),
        ('L4Finger', 28, True, (1.341830,-3.219061,-1.110226), (1.376633,-3.326726,-1.333154), True),
        ('LTip', 2, True, (-0.945764,-0.099278,-0.167966), (-1.109349,0.030293,-0.305630), True),
        ('RTip', 4, True, (-0.953029,0.090079,-0.167966), (-1.116090,-0.047048,-0.298760), True),
        ],

    "BUGBEAST": [
        ('LToe',     2, True,  ( 1.170711, 0.518669,-3.742350), ( 1.417519, 0.520483,-3.782134), True),
        ('RToe',     3, True,  ( 1.180363,-0.434027,-3.739797), ( 1.427355,-0.429722,-3.778223), True),
        ('LAnkle',   4, True,  ( 0.326978, 0.512466,-3.606343), ( 1.170711, 0.518669,-3.742350), False),
        ('RAnkle',   5, True,  ( 0.273294,-0.449841,-3.598678), ( 1.180363,-0.434027,-3.739797), False),
        ('LKnee',    6, True,  ( 0.380458, 0.512837,-1.936105), ( 0.326978, 0.512466,-3.606343), False),
        ('RKnee',    7, True,  ( 0.406784,-0.448344,-1.855310), ( 0.273294,-0.449841,-3.598678), False),
        ('LHip',     8, False, ( 0.534280, 0.520394,-0.321374), ( 0.380458, 0.512837,-1.936105), False),
        ('RHip',     8, False, ( 0.352594,-0.508449,-0.274951), ( 0.406784,-0.448344,-1.855310), False),
        ('Butt',    -1, False, ( 0.000000, 0.000000, 0.000000), ( 1.000000, 0.000000, 0.000000), False),
        ('Neck',    18, False, ( 0.374203, 0.027908, 2.207789), ( 0.409234, 0.057085, 3.074788), False),
        ('LShldr',  18, False, ( 0.342205, 0.761709, 1.850430), ( 0.302868, 1.738829, 1.897623), False),
        ('RShldr',  18, False, ( 0.365449,-0.676601, 1.887203), ( 0.586837,-1.623167, 2.054458), False),
        ('LElbow',  10, True,  ( 0.302868, 1.738829, 1.897623), ( 0.555697, 2.678110, 1.823117), False),
        ('RElbow',  11, True,  ( 0.586837,-1.623167, 2.054458), ( 1.143516,-2.395677, 2.043423), False),
        ('LWrist',  12, True,  ( 0.555697, 2.678110, 1.823117), ( 0.393444, 3.250001, 2.002098), False),
        ('RWrist',  13, True,  ( 1.143516,-2.395677, 2.043423), ( 1.130717,-2.912326, 2.338445), False),
        ('LFinger', 14, True,  ( 0.393444, 3.250001, 2.002098), ( 0.575506, 4.319257, 1.837476), False),
        ('RFinger', 15, True,  ( 1.130717,-2.912326, 2.338445), ( 1.415684,-3.975024, 2.197463), False),
        ('Abdomen',  8, False, ( 0.158710, 0.011265, 0.636473), ( 1.158710, 0.011265, 0.636473), False),
        ('Head',     9, True,  ( 0.409234, 0.057085, 3.074788), ( 0.419321, 0.065486, 3.324443), True),
        ('LClaw',   16, True,  ( 0.575506, 4.319257, 1.837476), ( 0.616994, 4.562920, 1.799962), True),
        ('RClaw',   17, True,  ( 1.415684,-3.975024, 2.197463), ( 1.479910,-4.214535, 2.165688), True),
        ],

    # TODO: when importing/creating CRAYMAN armatures, special-case the left shoulder/elbow/wrist
    #       so the bones are connected as per usual; but when exporting, recognise the topology
    #       from the pincher fork, and make sure to export the correct torsos vs limbs.
    "CRAYMAN": [
        ('LToe',      2, True,  ( 1.017140, 0.450631,-3.251434), ( 1.263947, 0.452445,-3.291219), True),
        ('RToe',      3, True,  ( 1.025526,-0.377092,-3.249217), ( 1.272518,-0.372787,-3.287643), True),
        ('LAnkle',    4, True,  ( 0.284086, 0.445242,-3.133269), ( 1.017140, 0.450631,-3.251434), False),
        ('RAnkle',    5, True,  ( 0.237444,-0.390832,-3.126609), ( 1.025526,-0.377092,-3.249217), False),
        ('LKnee',     6, True,  ( 0.330551, 0.445564,-1.682129), ( 0.284086, 0.445242,-3.133269), False),
        ('RKnee',     7, True,  ( 0.353423,-0.389531,-1.611933), ( 0.237444,-0.390832,-3.126609), False),
        ('LHip',      8, False, ( 0.464194, 0.452130,-0.279218), ( 0.330551, 0.445564,-1.682129), False),
        ('RHip',      8, False, ( 0.306341,-0.441752,-0.238883), ( 0.353423,-0.389531,-1.611933), False),
        ('Butt',     -1, False, ( 0.000000, 0.000000, 0.000000), ( 1.000000, 0.000000, 0.000000), False),
        ('Neck',     18, False, ( 0.325116, 0.024248, 1.918174), ( 0.355552, 0.049597, 2.671442), False),
        ('LShldr',   18, False, ( 0.297316, 0.661789, 1.607693), ( 1.297316, 0.661789, 1.607693), False),
        ('RShldr',   18, False, ( 0.317510,-0.587845, 1.639642), ( 0.268345,-1.463741, 1.727563), False),
        ('LElbow',   10, False, ( 0.154666, 1.470835, 1.675749), ( 1.154666, 1.470835, 1.675749), False),
        ('RElbow',   11, True,  ( 0.268345,-1.463741, 1.727563), ( 0.483673,-2.380519, 1.641099), False),
        ('TPincher', 12, False, ( 0.409548, 2.635859, 1.414143), ( 0.763567, 5.092938, 2.131115), False),
        ('RWrist',   13, True,  ( 0.483673,-2.380519, 1.641099), ( 1.229813,-5.233884, 1.264511), False),
        ('TTip',     14, True,  ( 0.763567, 5.092938, 2.131115), ( 0.797819, 5.330666, 2.200484), True),
        ('RFinger',  15, True,  ( 1.229813,-5.233884, 1.264511), ( 1.292551,-5.473804, 1.232847), True),
        ('Abdomen',   8, False, ( 0.137891, 0.009787, 0.552981), ( 1.137891, 0.009787, 0.552981), False),
        ('Head',      9, True,  ( 0.355552, 0.049597, 2.671442), ( 0.365639, 0.057998, 2.921097), True),
        ('BPincher', 12, False, ( 0.409548, 2.635859, 1.412012), ( 0.714571, 4.804078, 0.584941), False),
        ('BTip',     20, True,  ( 0.714571, 4.804078, 0.584941), ( 0.747151, 5.035669, 0.496600), True),
        ],

    "CONSTANTINE": [
        ('LToe', 2, True, (1.061637,0.729181,-3.187923), (1.307049,0.749264,-3.231159), True),
        ('RToe', 3, True, (0.994403,-0.895230,-3.193205), (1.232736,-0.940769,-3.253401), True),
        ('LAnkle', 20, True, (0.198115,0.658517,-3.035788), (1.061637,0.729181,-3.187923), False),
        ('RAnkle', 21, True, (0.296578,-0.761894,-3.016956), (0.994403,-0.895230,-3.193205), False),
        ('LKnee', 6, True, (0.322444,0.559972,-1.621091), (-0.596783,0.264132,-2.144972), False),
        ('RKnee', 7, True, (0.392263,-0.646469,-1.547782), (-0.503117,-0.591822,-2.074647), False),
        ('LHip', 8, False, (0.364440,0.634608,-0.252734), (0.322444,0.559972,-1.621091), False),
        ('RHip', 8, False, (0.273766,-0.639172,-0.236045), (0.392263,-0.646469,-1.547782), False),
        ('Butt', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Neck', 18, False, (0.085568,0.000325,1.954369), (0.142497,-0.006470,2.880650), False),
        ('LShldr', 18, False, (0.305988,0.733122,1.600833), (0.186058,1.454224,1.516937), False),
        ('RShldr', 18, False, (0.350919,-0.604184,1.589910), (0.251847,-1.398904,1.675816), False),
        ('LElbow', 10, True, (0.186058,1.454224,1.516937), (0.423436,2.218672,1.575835), False),
        ('RElbow', 11, True, (0.251847,-1.398904,1.675816), (0.492085,-2.111223,1.503657), False),
        ('LWrist', 12, True, (0.423436,2.218672,1.575835), (0.491954,3.061949,1.834016), False),
        ('RWrist', 13, True, (0.492085,-2.111223,1.503657), (0.779424,-2.818241,1.677361), False),
        ('LFinger', 14, True, (0.491954,3.061949,1.834016), (0.511318,3.300278,1.906984), True),
        ('RFinger', 15, True, (0.779424,-2.818241,1.677361), (0.871202,-3.044069,1.732844), True),
        ('Abdomen', 8, False, (0.178078,0.024918,0.552869), (1.178078,0.024918,0.552869), False),
        ('Head', 9, True, (0.142497,-0.006470,2.880650), (0.157833,-0.008300,3.130172), True),
        ('LDogLeg', 4, True, (-0.596783,0.264132,-2.144972), (0.198115,0.658517,-3.035788), False),
        ('RDogLeg', 5, True, (-0.503117,-0.591822,-2.074647), (0.296578,-0.761894,-3.016956), False),
        ('Tail', 8, False, (-4.157607,0.016612,0.509623), (-3.157607,0.016612,0.509623), True),
        ],

    "APPARITION": [
        ('Toe', 8, False, (-1.648353,0.000000,-3.563158), (-0.648353,0.000000,-3.563158), False),
        # TODO: apparition doesnt use joint ids 1-7 ! but this current skeleton definition needs them :(
        ('Dummy', 8, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), True),
        ('Dummy', 8, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), True),
        ('Dummy', 8, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), True),
        ('Dummy', 8, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), True),
        ('Dummy', 8, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), True),
        ('Dummy', 8, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), True),
        ('Dummy', 8, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), True),
        # Actual apparition bones below:
        ('Butt', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Neck', 18, False, (0.083846,0.095641,1.787435), (0.139629,0.054508,2.607577), False),
        ('LShldr', 18, False, (0.422523,0.616513,1.411899), (0.219122,1.612593,1.257849), False),
        ('RShldr', 18, False, (0.270241,-0.570582,1.439023), (0.025932,-1.456528,1.258396), False),
        ('LElbow', 10, True, (0.219122,1.612593,1.257849), (0.562147,2.367184,1.314068), False),
        ('RElbow', 11, True, (0.025932,-1.456528,1.258396), (0.355401,-2.331424,1.186977), False),
        ('LWrist', 12, True, (0.562147,2.367184,1.314068), (0.563087,3.082435,1.174023), False),
        ('RWrist', 13, True, (0.355401,-2.331424,1.186977), (0.509001,-3.064059,1.091953), False),
        ('LFinger', 14, True, (0.563087,3.082435,1.174023), (0.563409,3.327776,1.125985), True),
        ('RFinger', 15, True, (0.509001,-3.064059,1.091953), (0.559891,-3.306792,1.060470), True),
        ('Abdomen', 8, False, (0.088609,0.040500,0.476893), (1.088609,0.040500,0.476893), False),
        ('Head', 9, True, (0.139629,0.054508,2.607577), (0.156573,0.042014,2.856689), True),
        ],

    # There's no .cal for the sweel, so these positions are just a best guess.
    # TODO: there *are* sweel motions, so maybe extract a possible rest pose from those?
    "SWEEL": [
        ('Base', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Back', 0, False, (0.000000,0.000000,1.000000), (0.000000,0.000000,2.000000), False),
        ('Shoulder', 1, True, (0.000000,0.000000,2.000000), (0.000000,0.000000,3.000000), False),
        ('Neck', 2, True, (0.000000,0.000000,3.000000), (0.000000,0.000000,4.000000), False),
        ('Head', 3, True, (0.000000,0.000000,4.000000), (0.000000,0.000000,4.250000), True),
        ('Tail', 0, False, (-1.000000,0.000000,0.000000), (-2.000000,0.000000,0.000000), False),
        ('Tip', 5, True, (-2.000000,0.000000,0.000000), (-2.250000,0.000000,0.000000), True),
        ],

    "ROPE": [
        ('Node_0', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Node_1', 0, False, (0.004206,0.000000,-1.009076), (-0.000000,-0.002121,-2.031790), False),
        ('Node_2', 1, True, (-0.000000,-0.002121,-2.031790), (-0.000000,-0.002121,-3.013595), False),
        ('Node_3', 2, True, (-0.000000,-0.002121,-3.013595), (0.002103,0.000000,-4.022671), False),
        ('Node_4', 3, True, (0.002103,0.000000,-4.022671), (-0.000000,0.002121,-5.018113), False),
        ('Node_5', 4, True, (-0.000000,0.002121,-5.018113), (0.004206,0.002121,-5.999917), False),
        ('Node_6', 5, True, (0.004206,0.002121,-5.999917), (0.002103,0.000000,-7.036266), False),
        ('Node_7', 6, True, (0.002103,0.000000,-7.036266), (-0.000372,-0.001032,-8.010067), False),
        ('Node_8', 7, True, (-0.000372,-0.001032,-8.010067), (-0.001007,-0.001297,-8.260067), True),
        ],

    # TODO: merge with HUMANOID
    "ZOMBIE": [
        ('LToe', 2, True, (0.935079,0.322488,-3.211668), (1.182213,0.344501,-3.242333), True),
        ('RToe', 3, True, (0.893562,-0.400766,-3.195425), (1.141494,-0.423312,-3.218257), True),
        ('LAnkle', 4, True, (0.103983,0.248461,-3.108542), (0.935079,0.322488,-3.211668), False),
        ('RAnkle', 5, True, (0.079495,-0.326742,-3.120456), (0.893562,-0.400766,-3.195425), False),
        ('LKnee', 6, True, (0.180130,0.259361,-1.737175), (0.103983,0.248461,-3.108542), False),
        ('RKnee', 7, True, (0.250563,-0.321426,-1.703842), (0.079495,-0.326742,-3.120456), False),
        ('LHip', 8, False, (0.282737,0.213869,-0.608307), (0.180130,0.259361,-1.737175), False),
        ('RHip', 8, False, (0.279803,-0.299643,-0.565987), (0.250563,-0.321426,-1.703842), False),
        ('Butt', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Neck', 18, False, (0.082193,0.093755,1.752201), (0.136877,-0.025031,2.570484), False),
        ('LShldr', 18, False, (0.414195,0.604360,1.384068), (0.214803,1.580805,1.233055), False),
        ('RShldr', 18, False, (0.264915,-0.559335,1.410658), (0.025422,-1.427818,1.233591), False),
        ('LElbow', 10, True, (0.214803,1.580805,1.233055), (0.551067,2.320523,1.288166), False),
        ('RElbow', 11, True, (0.025422,-1.427818,1.233591), (0.348396,-2.285468,1.163580), False),
        ('LWrist', 12, True, (0.551067,2.320523,1.288166), (0.675945,2.907196,1.200737), False),
        ('RWrist', 13, True, (0.348396,-2.285468,1.163580), (0.541926,-2.833545,1.041128), False),
        ('LFinger', 14, True, (0.675945,2.907196,1.200737), (0.727449,3.149161,1.164678), True),
        ('RFinger', 15, True, (0.541926,-2.833545,1.041128), (0.623378,-3.064216,0.989591), True),
        ('Abdomen', 8, False, (0.086863,0.039701,0.467493), (1.086863,0.039701,0.467493), False),
        ('Head', 9, True, (0.136877,-0.025031,2.570484), (0.153374,-0.060867,2.817352), True),
        ],

    # TODO: merge with SPIDER
    "SMALL_SPIDER": [
        ('Base', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('LMand', 0, False, (-0.194652,-0.060155,-0.043805), (-0.244861,-0.080280,-0.001684), False),
        ('LMElbow', 1, True, (-0.244861,-0.080280,-0.001684), (-0.307323,-0.030092,-0.054541), False),
        ('RMand', 0, False, (-0.193301,0.062726,-0.042964), (-0.243280,0.086108,-0.001931), False),
        ('RMElbow', 3, True, (-0.243280,0.086108,-0.001931), (-0.309252,0.031410,-0.054541), False),
        ('R1Shldr', 0, False, (-0.115617,0.174778,0.001752), (-0.256934,0.375596,0.258687), False),
        ('R1Elbow', 5, True, (-0.256934,0.375596,0.258687), (-0.515075,0.782223,0.049162), False),
        ('R1Wrist', 6, True, (-0.515075,0.782223,0.049162), (-0.596854,0.959717,-0.360509), False),
        ('R2Shldr', 0, False, (-0.047663,0.182787,0.001752), (-0.120007,0.417445,0.258687), False),
        ('R2Elbow', 8, True, (-0.120007,0.417445,0.258687), (-0.239859,0.883941,0.049162), False),
        ('R2Wrist', 9, True, (-0.239859,0.883941,0.049162), (-0.261831,1.113326,-0.360509), False),
        ('R3Shldr', 0, False, (0.018523,0.188022,0.001752), (0.043248,0.428883,0.258687), False),
        ('R3Elbow', 11, True, (0.043248,0.428883,0.258687), (0.044989,0.910624,0.049162), False),
        ('R3Wrist', 12, True, (0.044989,0.910624,0.049162), (0.043528,1.122054,-0.360509), False),
        ('R4Shldr', 0, False, (0.089543,0.184942,0.001752), (0.220665,0.386944,0.258687), False),
        ('R4Elbow', 14, True, (0.220665,0.386944,0.258687), (0.365833,0.847701,0.049162), False),
        ('R4Wrist', 15, True, (0.365833,0.847701,0.049162), (0.430898,1.053977,-0.360509), False),
        ('L1Shldr', 0, False, (-0.115617,-0.177080,0.001752), (-0.256934,-0.377898,0.258687), False),
        ('L1Elbow', 17, True, (-0.256934,-0.377898,0.258687), (-0.515075,-0.784525,0.049162), False),
        ('L1Wrist', 18, True, (-0.515075,-0.784525,0.049162), (-0.613342,-1.011288,-0.360509), False),
        ('L2Shldr', 0, False, (-0.047663,-0.185089,0.001752), (-0.120007,-0.419748,0.258687), False),
        ('L2Elbow', 20, True, (-0.120007,-0.419748,0.258687), (-0.239859,-0.886243,0.049162), False),
        ('L2Wrist', 21, True, (-0.239859,-0.886243,0.049162), (-0.257006,-1.098944,-0.360509), False),
        ('L3Shldr', 0, False, (0.018523,-0.190325,0.001752), (0.043248,-0.431185,0.258687), False),
        ('L3Elbow', 23, True, (0.043248,-0.431185,0.258687), (0.044989,-0.912926,0.049162), False),
        ('L3Wrist', 24, True, (0.044989,-0.912926,0.049162), (0.042862,-1.118423,-0.360509), False),
        ('L4Shldr', 0, False, (0.089543,-0.187244,0.001752), (0.220665,-0.389246,0.258687), False),
        ('L4Elbow', 26, True, (0.220665,-0.389246,0.258687), (0.365833,-0.850003,0.049162), False),
        ('L4Wrist', 27, True, (0.365833,-0.850003,0.049162), (0.428407,-1.048298,-0.360509), False),
        ('R1Finger', 7, True, (-0.596854,0.959717,-0.360509), (-0.641896,1.057479,-0.586150), True),
        ('R2Finger', 10, True, (-0.261831,1.113326,-0.360509), (-0.273517,1.235330,-0.578404), True),
        ('R3Finger', 13, True, (0.043528,1.122054,-0.360509), (0.042735,1.236708,-0.582666), True),
        ('R4Finger', 16, True, (0.430898,1.053977,-0.360509), (0.466009,1.165294,-0.581588), True),
        ('L1Finger', 19, True, (-0.613342,-1.011288,-0.360509), (-0.664690,-1.129778,-0.574574), True),
        ('L2Finger', 22, True, (-0.257006,-1.098944,-0.360509), (-0.266286,-1.214063,-0.582233), True),
        ('L3Finger', 25, True, (0.042862,-1.118423,-0.360509), (0.041701,-1.230514,-0.583969), True),
        ('L4Finger', 28, True, (0.428407,-1.048298,-0.360509), (0.462457,-1.156203,-0.583437), True),
        ('LTip', 2, True, (-0.307323,-0.030092,-0.054541), (-0.470000,0.100617,-0.192205), True),
        ('RTip', 4, True, (-0.309252,0.031410,-0.054541), (-0.473267,-0.104576,-0.185336), True),
        ],

    # TODO: merge with BURRICK
    "FROG": [
        ('LToe', 2, True, (0.378453,0.275477,-0.851730), (0.612707,0.256408,-0.936944), True),
        ('RToe', 3, True, (0.382258,-0.334437,-0.847035), (0.617618,-0.325476,-0.930853), True),
        ('LAnkle', 4, True, (0.226059,0.287881,-0.796295), (0.378453,0.275477,-0.851730), False),
        ('RAnkle', 5, True, (0.196966,-0.341492,-0.781047), (0.382258,-0.334437,-0.847035), False),
        ('LKnee', 6, True, (0.322197,0.264945,-0.381022), (0.226059,0.287881,-0.796295), False),
        ('RKnee', 7, True, (0.266464,-0.329761,-0.346667), (0.196966,-0.341492,-0.781047), False),
        ('LHip', 8, False, (0.165430,0.203643,-0.091792), (0.322197,0.264945,-0.381022), False),
        ('RHip', 8, False, (0.096097,-0.266501,-0.042480), (0.266464,-0.329761,-0.346667), False),
        ('Butt', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Neck', 18, False, (0.681372,0.018598,0.079116), (1.457583,-0.007542,0.365093), False),
        ('LShldr', 18, False, (0.578048,0.166082,-0.035857), (0.535669,0.247205,-0.294291), False),
        ('RShldr', 18, False, (0.569139,-0.135317,0.006070), (0.512768,-0.317255,-0.189417), False),
        ('LElbow', 10, True, (0.535669,0.247205,-0.294291), (0.673150,0.136649,-0.219067), False),
        ('RElbow', 11, True, (0.512768,-0.317255,-0.189417), (0.693010,-0.205419,-0.189114), False),
        ('LWrist', 12, True, (0.673150,0.136649,-0.219067), (0.822563,-0.011196,-0.254306), False),
        ('RWrist', 13, True, (0.693010,-0.205419,-0.189114), (0.850867,-0.167598,-0.296089), False),
        ('LFinger', 14, True, (0.822563,-0.011196,-0.254306), (0.997824,-0.184618,-0.295641), True),
        ('RFinger', 15, True, (0.850867,-0.167598,-0.296089), (1.053868,-0.118960,-0.433658), True),
        ('Abdomen', 8, False, (0.257191,0.007464,0.090013), (1.257191,0.007464,0.090013), False),
        ('Head', 9, True, (1.457583,-0.007542,0.365093), (1.692051,-0.015438,0.451477), True),
        ('Tail', 8, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), True),
        ],

    # TODO: merge with HUMANOID
    "CUTTY": [
        ('LToe', 2, True, (0.985982,0.255872,-3.125115), (1.234358,0.259832,-3.153284), True),
        ('RToe', 3, True, (0.991246,-0.389966,-3.109309), (1.239688,-0.409548,-3.129140), True),
        ('LAnkle', 4, True, (0.101181,0.241765,-3.024767), (0.985982,0.255872,-3.125115), False),
        ('RAnkle', 5, True, (0.077351,-0.317936,-3.036361), (0.991246,-0.389966,-3.109309), False),
        ('LKnee', 6, True, (0.175275,0.252371,-1.690359), (0.101181,0.241765,-3.024767), False),
        ('RKnee', 7, True, (0.243810,-0.312764,-1.657924), (0.077351,-0.317936,-3.036361), False),
        ('LHip', 8, False, (0.275117,0.208105,-0.591912), (0.175275,0.252371,-1.690359), False),
        ('RHip', 8, False, (0.272262,-0.291568,-0.550734), (0.243810,-0.312764,-1.657924), False),
        ('Butt', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Neck', 18, False, (0.079978,0.091229,1.704981), (0.240219,-0.016559,2.583091), False),
        ('LShldr', 18, False, (0.297153,0.593310,1.346768), (0.209014,1.538203,1.199824), False),
        ('RShldr', 18, False, (0.257775,-0.544262,1.372642), (0.024736,-1.389339,1.200346), False),
        ('LElbow', 10, True, (0.209014,1.538203,1.199824), (0.536215,2.257986,1.253451), False),
        ('RElbow', 11, True, (0.024736,-1.389339,1.200346), (0.339006,-2.223876,1.132223), False),
        ('LWrist', 12, True, (0.536215,2.257986,1.253451), (0.596936,2.828849,1.168378), False),
        ('RWrist', 13, True, (0.339006,-2.223876,1.132223), (0.378735,-2.777946,1.050287), False),
        ('LFinger', 14, True, (0.596936,2.828849,1.168378), (0.623092,3.074761,1.131731), True),
        ('RFinger', 15, True, (0.378735,-2.777946,1.050287), (0.396423,-3.024637,1.013806), True),
        ('Abdomen', 8, False, (0.084522,0.038631,0.454895), (1.084522,0.038631,0.454895), False),
        ('Head', 9, True, (0.240219,-0.016559,2.583091), (0.284775,-0.046531,2.827256), True),
        ],

    # TODO: "AVATAR": - exact same skeleton definition as HUMANOID

    "ROBOT": [
        ('LToe', 2, True, (0.839878,1.231792,-3.288423), (1.023777,1.207722,-3.456060), True),
        ('RToe', 3, True, (0.946882,-1.287747,-3.284290), (1.136831,-1.268632,-3.445703), True),
        ('LAnkle', 4, True, (-0.878642,1.456727,-1.721873), (0.839878,1.231792,-3.288423), False),
        ('RAnkle', 5, True, (-0.879178,-1.471508,-1.732564), (0.946882,-1.287747,-3.284290), False),
        ('LKnee', 6, True, (0.624675,1.449698,-0.773398), (-0.878642,1.456727,-1.721873), False),
        ('RKnee', 7, True, (0.563531,-1.454847,-0.802951), (-0.879178,-1.471508,-1.732564), False),
        ('LHip', 8, False, (-0.007961,1.459840,-0.005152), (0.624675,1.449698,-0.773398), False),
        ('RHip', 8, False, (-0.004058,-1.474143,-0.013808), (0.563531,-1.454847,-0.802951), False),
        ('Butt', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Abdomen', 8, False, (-0.010491,0.000552,0.550898), (0.989509,0.000552,0.550898), False),
        ('Neck', 9, False, (-0.021752,0.015704,2.448725), (-0.021752,0.015703,4.358021), False),
        ('LShldr', 9, False, (-0.022204,1.979230,2.454675), (-0.718890,1.966193,1.304038), False),
        ('RShldr', 9, False, (-0.019511,-1.882130,2.448167), (-0.676323,-1.878588,1.251288), False),
        ('LElbow', 11, True, (-0.718890,1.966193,1.304038), (2.044902,1.977960,1.276250), False),
        ('RElbow', 12, True, (-0.676323,-1.878588,1.251288), (2.002214,-1.905006,1.282021), False),
        ('LWrist', 13, True, (2.044902,1.977960,1.276250), (2.294887,1.979025,1.273736), True),
        ('RWrist', 14, True, (2.002214,-1.905006,1.282021), (2.252186,-1.907471,1.284889), True),
        ('Head', 10, True, (-0.021752,0.015703,4.358021), (-0.021752,0.015703,4.608021), True),
        ],

    # TODO: merge with ROBOT
    "SMALL_ROBOT": [
        ('LToe', 2, True, (0.476817,0.699314,-1.866906), (0.660715,0.675244,-2.034542), True),
        ('RToe', 3, True, (0.537565,-0.731081,-1.864560), (0.727514,-0.711966,-2.025972), True),
        ('LAnkle', 4, True, (-0.498823,0.827014,-0.977543), (0.476817,0.699314,-1.866906), False),
        ('RAnkle', 5, True, (-0.499128,-0.835406,-0.983612), (0.537565,-0.731081,-1.864560), False),
        ('LKnee', 6, True, (0.354642,0.823023,-0.439074), (-0.498823,0.827014,-0.977543), False),
        ('RKnee', 7, True, (0.319929,-0.825947,-0.455852), (-0.499128,-0.835406,-0.983612), False),
        ('LHip', 8, False, (-0.004519,0.828782,-0.002925), (0.354642,0.823023,-0.439074), False),
        ('RHip', 8, False, (-0.002303,-0.836902,-0.007839), (0.319929,-0.825947,-0.455852), False),
        ('Butt', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('Abdomen', 8, False, (-0.005956,0.000313,0.312757), (0.994044,0.000313,0.312757), False),
        ('Neck', 9, False, (-0.012349,0.008915,1.390192), (-0.012349,0.008915,2.474140), False),
        ('LShldr', 9, False, (-0.012606,1.123650,1.393570), (-0.408129,1.116248,0.740330), False),
        ('RShldr', 9, False, (-0.011077,-1.068525,1.389876), (-0.383962,-1.066514,0.710383), False),
        ('LElbow', 11, True, (-0.408129,1.116248,0.740330), (1.160934,1.122930,0.724554), False),
        ('RElbow', 12, True, (-0.383962,-1.066514,0.710383), (1.136699,-1.081512,0.727830), False),
        ('LWrist', 13, True, (1.160934,1.122930,0.724554), (1.410919,1.123994,0.722040), True),
        ('RWrist', 14, True, (1.136699,-1.081512,0.727830), (1.386670,-1.083977,0.730698), True),
        ('Head', 10, True, (-0.012349,0.008915,2.474140), (-0.012349,0.008915,2.724140), True),
        ],

    "SPIDER_BOT": [
        ('Base', -1, False, (0.000000,0.000000,0.000000), (1.000000,0.000000,0.000000), False),
        ('LMand', 0, False, (-0.966854,-0.234567,0.003028), (-1.375582,-0.418715,-0.110450), False),
        ('LMElbow', 1, True, (-1.375582,-0.418715,-0.110450), (-1.598338,-0.288490,-0.382587), False),
        ('RMand', 0, False, (-0.962055,0.098541,0.005063), (-1.382013,0.247584,-0.111316), False),
        ('RMElbow', 3, True, (-1.382013,0.247584,-0.111316), (-1.594714,0.146513,-0.380262), False),
        ('R1Shldr', 0, False, (-0.278199,0.423761,0.113229), (-0.618348,0.910729,0.734757), False),
        ('R1Elbow', 5, True, (-0.618348,0.910729,0.734757), (-1.239353,1.896535,0.227916), False),
        ('R1Wrist', 6, True, (-1.239353,1.896535,0.227916), (-1.435676,2.326582,-0.763078), False),
        ('R2Shldr', 0, False, (-0.113752,0.442562,0.113229), (-0.286771,1.010808,0.734757), False),
        ('R2Elbow', 8, True, (-0.286771,1.010808,0.734757), (-0.572752,2.140266,0.227916), False),
        ('R2Wrist', 9, True, (-0.572752,2.140266,0.227916), (-0.623965,2.695329,-0.763078), False),
        ('R3Shldr', 0, False, (0.046396,0.454667,0.113229), (0.108238,1.037096,0.734757), False),
        ('R3Elbow', 11, True, (0.108238,1.037096,0.734757), (0.116518,2.202405,0.227916), False),
        ('R3Wrist', 12, True, (0.116518,2.202405,0.227916), (0.114767,2.713863,-0.763078), False),
        ('R4Shldr', 0, False, (0.218165,0.446615,0.113229), (0.537052,0.934149,0.734757), False),
        ('R4Elbow', 14, True, (0.537052,0.934149,0.734757), (0.892103,2.047487,0.227916), False),
        ('R4Wrist', 15, True, (0.892103,2.047487,0.227916), (1.051234,2.545916,-0.763078), False),
        ('L1Shldr', 0, False, (-0.281171,-0.427378,0.113229), (-0.624710,-0.911959,0.734757), False),
        ('L1Elbow', 17, True, (-0.624710,-0.911959,0.734757), (-1.252583,-1.893405,0.227916), False),
        ('L1Wrist', 18, True, (-1.252583,-1.893405,0.227916), (-1.492205,-2.441111,-0.763078), False),
        ('L2Shldr', 0, False, (-0.116859,-0.447326,0.113229), (-0.293840,-1.014350,0.734757), False),
        ('L2Elbow', 20, True, (-0.293840,-1.014350,0.734757), (-0.587699,-2.141784,0.227916), False),
        ('L2Wrist', 21, True, (-0.587699,-2.141784,0.227916), (-0.630973,-2.656159,-0.763078), False),
        ('L3Shldr', 0, False, (0.043201,-0.460549,0.113229), (0.100975,-1.043396,0.734757), False),
        ('L3Elbow', 23, True, (0.100975,-1.043396,0.734757), (0.101120,-2.208734,0.227916), False),
        ('L3Wrist', 24, True, (0.101120,-2.208734,0.227916), (0.094238,-2.705811,-0.763078), False),
        ('L4Shldr', 0, False, (0.215023,-0.453696,0.113229), (0.530498,-0.943444,0.734757), False),
        ('L4Elbow', 26, True, (0.530498,-0.943444,0.734757), (0.877768,-2.059234,0.227916), False),
        ('L4Wrist', 27, True, (0.877768,-2.059234,0.227916), (1.027456,-2.539435,-0.763078), False),
        ('R1Finger', 7, True, (-1.435676,2.326582,-0.763078), (-1.480377,2.424500,-0.988720), True),
        ('R2Finger', 10, True, (-0.623965,2.695329,-0.763078), (-0.635225,2.817374,-0.980973), True),
        ('R3Finger', 13, True, (0.114767,2.713863,-0.763078), (0.114375,2.828519,-0.985235), True),
        ('R4Finger', 16, True, (1.051234,2.545916,-0.763078), (1.086734,2.657109,-0.984157), True),
        ('L1Finger', 19, True, (-1.492205,-2.441111,-0.763078), (-1.543965,-2.559421,-0.977143), True),
        ('L2Finger', 22, True, (-0.630973,-2.656159,-0.763078), (-0.640655,-2.771244,-0.984802), True),
        ('L3Finger', 25, True, (0.094238,-2.705811,-0.763078), (0.092687,-2.817897,-0.986538), True),
        ('L4Finger', 28, True, (1.027456,-2.539435,-0.763078), (1.061129,-2.647458,-0.986007), True),
        ('LTip', 2, True, (-1.598338,-0.288490,-0.382587), (-1.746836,-0.201677,-0.564004), True),
        ('RTip', 4, True, (-1.594714,0.146513,-0.380262), (-1.743467,0.075830,-0.568349), True),
        ('Sac', 0, False, (0.205546,-0.004777,2.322020), (1.205546,-0.004777,2.322020), True),
        ],

    }

def add_creature_armature(creature_type, context):
    try:
        skeleton = SKELETONS[creature_type]
    except KeyError:
        raise ValueError(f"Unsupported Creature Type '{creature_type}'")

    arm_obj = create_armature("Armature", Vector((0,0,0)), context=context)
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    edit_bones = arm_obj.data.edit_bones
    bones = {}
    # store actual bone names (i.e. with ".001" suffix if added) to look up
    # pose bones with:
    bone_names = {}
    def add_bone(j, name, pj, connected, head_pos, tail_pos, limb_end):
        b = edit_bones.new(name)
        b.head = head_pos
        b.tail = tail_pos
        b.use_deform = (not limb_end)
        print(j, b)
        bones[j] = b
        bone_names[j] = b.name
    def connect_bone(j, name, pj, connected, head_pos, tail_pos, limb_end):
        b = bones[j]
        if pj != -1:
            b.use_connect = connected
            b.parent = bones[pj]
    def hide_bone(j, name, pj, connected, head_pos, tail_pos, limb_end):
        print(arm_obj.data.items())
        b = arm_obj.data.bones[bone_names[j]]
        b.hide = limb_end

    for j, args in enumerate(skeleton):
        add_bone(j, *args)
    for j, args in enumerate(skeleton):
        connect_bone(j, *args)
    bpy.ops.object.mode_set(mode='POSE')
    for j, args in enumerate(skeleton):
        hide_bone(j, *args)
    bpy.ops.object.mode_set(mode='OBJECT')

#---------------------------------------------------------------------------#
# Operators

class TTDebugImportMeshOperator(Operator):
    bl_idname = "object.tt_debug_import_mesh"
    bl_label = "Import mesh"
    bl_options = {'REGISTER', 'UNDO'}

    filename : StringProperty()

    def execute(self, context):
        if context.mode != "OBJECT":
            self.report({'WARNING'}, f"{self.bl_label}: must be in Object mode.")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        print(f"filename: {self.filename}")
        do_import_mesh(context, self.filename)
        # context.view_layer.objects.active = o
        # o.select_set(True)
        return {'FINISHED'}

class TTDebugExportMeshOperator(Operator):
    bl_idname = "object.tt_debug_export_mesh"
    bl_label = "Export mesh"
    bl_options = {'REGISTER', 'UNDO'}

    filename : StringProperty()

    def execute(self, context):
        if context.mode != "OBJECT":
            self.report({'WARNING'}, f"{self.bl_label}: must be in Object mode.")
            return {'CANCELLED'}

        o = context.view_layer.objects.active
        if o.type != 'MESH':
            self.report({'WARNING'}, f"{self.bl_label}: active object is not a mesh.")
            return {'CANCELLED'}

        do_export_mesh(context, o, self.filename)
        return {'FINISHED'}

TT_CREATURE_TYPE_ENUM=[
    ("HUMANOID", "Humanoid", ""),
    ("PLAYER_ARM", "PlayerArm", ""),
    ("PLAYER_BOWARM", "PlayerBowArm", ""),
    ("BURRICK", "Burrick", ""),
    ("SPIDER", "Spider", ""),
    ("BUGBEAST", "BugBeast", ""),
    ("CRAYMAN", "Crayman", ""),
    ("CONSTANTINE", "Constantine", ""),
    ("APPARITION", "Apparition", ""),
    ("SWEEL", "Sweel", ""),
    ("ROPE", "Rope", ""),
    ("ZOMBIE", "Zombie", ""),
    ("SMALL_SPIDER", "Small Spider", ""),
    ("FROG", "Frog", ""),
    ("CUTTY", "Cutty", ""),
    ("AVATAR", "Avatar", ""),
    ("ROBOT", "Robot (T2)", ""),
    ("SMALL_ROBOT", "Small Robot (T2)", ""),
    ("SPIDER_BOT", "Spider Bot (T2)", ""),
    ]

class TTAddArmatureOperator(Operator):
    bl_idname = "object.tt_add_armature"
    bl_label = "Thief Armature"
    bl_options = {'REGISTER', 'UNDO'}

    creature_type : EnumProperty(
        items=TT_CREATURE_TYPE_ENUM,
        name="Creature Type",
        default=TT_CREATURE_TYPE_ENUM[0][0] )

    def execute(self, context):
        if context.mode != "OBJECT":
            self.report({'WARNING'}, f"{self.bl_label}: must be in Object mode.")
            return {'CANCELLED'}

        print(f"You chose {self.creature_type}!")
        add_creature_armature(self.creature_type, context)
        return {'FINISHED'}
