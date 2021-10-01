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
from bpy.props import IntProperty, PointerProperty, StringProperty
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
    weight_start: uint16 # number of weights = num vertices in segment
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
    caps: uint32
    alpha: float32
    self_illum: float32
    for_rent: uint32
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
    weight_start: uint16 # number of weights = num vertices in segment
    pad: uint16

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
    bbox: uint32
    joint_id: uint8
    smatsegs: uint8 # number of smatsegs in segment
    map_start: uint8
    flags: uint8        # 1 = stretchy segment
    pgons: uint16       # from here, only for segment order
    pgon_start: uint16
    verts: uint16
    vert_start: uint16
    weight_start: uint16 # number of weights = num vertices in segment
    pad: uint16

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

def create_armature(name, location, context=None, link=True):
    arm = bpy.data.armatures.new(name)
    arm.display_type = 'WIRE'
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
    arm_obj = create_armature("Human", Vector((0,0,0)), context=context)
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    edit_bones = arm_obj.data.edit_bones
    bones_by_joint_id = {}
    for j in sorted(head_by_joint_id.keys()):
        bone_name = HUMAN_JOINTS[j]
        b = edit_bones.new(bone_name)
        b.head = head_by_joint_id[j]
        b.tail = tail_by_joint_id[j]
        # TODO -- roll is all set to zero, but some of the bones are
        # getting created _not_ actually flat in world space??
        # BUT: check if that matches the imported skeleton anyway? with motions
        # etc, and see if they seem to match e.g. sword roll angle
        # ...maybe .align_roll()?
        b.roll = 0.0
        bones_by_joint_id[j] = b
        # TODO -- should we not create bones for FINGER, TOE?? they dont get
        # their own transforms... or do they??
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

    # TODO: do this as an array also, indexed by joint. We want to get
    #       to joint_ids as soon as possible!
    bone_head_pos = {}
    bone_directions = {}
    bone_lengths = {}
    for b in arm.bones:
        bone_head_pos[b.name] = Vector(arm_obj.location)+Vector(b.head_local)
        d = (Vector(b.tail_local)-Vector(b.head_local)).normalized()
        bone_directions[b.name] = d
        bone_lengths[b.name] = b.length

    vertex_group_joint_ids = [
        SPIDER_JOINT_INDICES.get(g.name.lower(), 0)
        for g in mesh_obj.vertex_groups]

    # TODO: ensure one uv layer

    # Vectors are all kinds of weird (only sometimes hashable), so
    # just keep everything as tuples unless we are doing maths on it!

    vertex_pos = []
    vertex_normal = []
    vertex_uv = []
    vertex_joint_id = []
    vertex_weight = []
    vertex_tuples = {}
    loop_vertex_id = []
    for li, (loop, uvloop) in enumerate(zip(mesh.loops, mesh.uv_layers[0].data)):
        mesh_vert = mesh.vertices[loop.vertex_index]
        pos = tuple(mesh_vert.co)
        normal = tuple(loop.normal)
        uv = tuple(uvloop.uv)
        # Split a vertex if it has multiple uvs or split normals.
        tup = (loop.vertex_index, normal, uv)
        vi = vertex_tuples.get(tup, -1)
        if vi==-1:
            vi = len(vertex_pos)
            vertex_pos.append(pos)
            vertex_normal.append(normal)
            vertex_uv.append(uv)
            vertex_tuples[tup] = vi
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
                raise ValueError(f"Vertex {loop.vertex_index} is in more than 2 vertex groups")
            vertex_joint_id.append(joint_id)
            vertex_weight.append(weight)
        loop_vertex_id.append(vi)

    poly_vertex_ids = []
    poly_material_id = []
    poly_normal = []
    poly_distance = []
    for tri in mesh.loop_triangles:
        vertex_ids = tuple(loop_vertex_id[li] for li in tri.loops)
        material_id = tri.material_index # TODO: handle materials!
        normal = tuple(tri.normal)
        distance = tri.center.length
        poly_vertex_ids.append(vertex_ids)
        poly_material_id.append(material_id)
        poly_normal.append(normal)
        poly_distance.append(distance)

    # From here on out, we are working with our arrays of data, and don't use
    # any more Blender objects (except for Vectors that we construct).


    print()
    print("VERTICES:")
    for vi, (pos, normal, uv, joint_id, weight) \
    in enumerate(zip(vertex_pos, vertex_normal, vertex_uv, vertex_joint_id, vertex_weight)):
        print(f"  {vi}: pos {pos}; normal {normal}; uv {uv}; joint_id {joint_id}; weight {weight}")
    print("TRIANGLES:")
    for pi, (vertex_ids, material_id, normal, distance) \
    in enumerate(zip(poly_vertex_ids, poly_material_id, poly_normal, poly_distance)):
        print(f"  {pi}: vertex_ids {vertex_ids}; material_id {material_id}; normal {normal}; distance {distance}")
    print()

    raise NotImplementedError("stop here")

    # okay, this is not good enough. we need to sort the vertices [by material,
    # once we stop hardcoding the material, then] by group, so that each
    # [material and each] smatseg can have contiguous vertices.

    # TODO: to support stretchy segments, we will need to support verts in
    # two groups, with weights derived from parent joint and child joint.
    # but for now, only nonstretchy groups!
    for vi, mesh_vert in enumerate(mesh.vertices):
        group_count = len(mesh_vert.groups)
        if group_count != 1:
            raise ValueError(f"Vertex {vi} is in {group_count} vertex groups! Should be only 1.")

    # TODO: what do we do about vertex groups that dont match any known
    # joint name? we should at least check for them!

    # for this first effort, we go with one smatr (all hard-coded); and for
    # each vertex group, one seg and one smatseg.

    # sort vertices by material and joint
    class SourceVert:
        def __init__(self, mesh_vert_id, mesh_vert):
            self.mesh_vert_id = mesh_vert_id
            self.mesh_vert = mesh_vert
            self.vert_id = -1
        @staticmethod
        def sort_key(source_vert):
            # TODO: insert real material sort key here
            temp_mat_id = 0
            # TODO: this will have to change when supporting stretchy segs (i.e. 2 groups)
            group_id = source_vert.mesh_vert.groups[0].group
            return (temp_mat_id, group_id)
    source_verts = sorted([
        SourceVert(mesh_vert_id, mesh_vert)
        for mesh_vert_id, mesh_vert
        in enumerate(mesh.vertices)
        ], key=SourceVert.sort_key)
    # update each source vert with its vert_id for the .bin
    for vi, source_vert in enumerate(source_verts):
        source_vert.vert_id = vi

    # TODO: vert positions must be local to the head of the bone they belong to
    verts = [
        LGVector(values=(v.mesh_vert.co.x,v.mesh_vert.co.y,v.mesh_vert.co.z))
        for v in source_verts
        ]
    print("VERTS:", file=dumpf)
    for i, v in enumerate(verts):
        print(f"  {i}: {v.x},{v.y},{v.z}", file=dumpf)

    vert_remap = [-1]*len(verts)
    for vi, source_vert in enumerate(source_verts):
        vert_remap[source_vert.mesh_vert_id] = source_vert.vert_id

    # TODO: actually generate souce_verts from loops, not from mesh.vertices!
    #       that will allow split normals, split uvs, and so on.
    mesh_uvs = mesh.uv_layers[0].data
    source_vert_uvs = [(0.0,0.0) for _ in source_verts]
    for li, loop in enumerate(mesh.loops):
        vi = vert_remap[loop.vertex_index]
        if source_vert_uvs[vi] == (0.0,0.0):
            source_vert_uvs[vi] = mesh_uvs[li].uv
        else:
            print(f"multiple uvs for vertex {loop.vertex_index}, not yet supported")
    uvnorms = [
        LGMMUVNorm(values=(
            source_vert_uvs[vi][0],
            source_vert_uvs[vi][1],
            pack_normal(v.mesh_vert.normal)))
        for vi, v in enumerate(source_verts)
        ]

    pgons = []
    norms = []
    for i, tri in enumerate(mesh.loop_triangles):
        pgon = LGMMPolygon()
        pgon.vert = Array(uint16, 3)([
            vert_remap[mesh.loops[tri.loops[0]].vertex_index],
            vert_remap[mesh.loops[tri.loops[1]].vertex_index],
            vert_remap[mesh.loops[tri.loops[2]].vertex_index],
            ])
        pgon.smatr_id = uint16(0) # TODO: multiple materials
        pgon.d = float32(Vector(tri.center).length)
        pgon.norm = uint16(i)
        pgons.append(pgon)
        norm = LGVector(values=tri.normal)
        norms.append(norm)

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
