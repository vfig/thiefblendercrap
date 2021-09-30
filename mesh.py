import bpy
import bmesh
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
    TypedArrayInstance.__name__ = f"{typeref_.__name__}x{count_}"
    return TypedArrayInstance

# TODO: can i turn Struct into a decorate that uses @dataclass and adds the
# read/write/size methods??
# TODO: also rename this because it clashes with struct.Struct haha
class Struct:
    def __init__(self, values):
        if self.__class__==Struct:
            raise TypeError("Cannot instantiate Struct itself, only subclasses")
        hints = get_type_hints(self.__class__)
        if len(hints)==0:
            raise TypeError(f"{self.__class__.__name__} has no fields defined")
        if len(values)!=len(hints):
            raise ValueError(f"Expected {len(self.hints)} values")
        for (name, typeref), value in zip(hints.items(), values):
            setattr(self, name, typeref(value))

    # @classmethod
    # def format_string(cls):
    #     fmt = []
    #     hints = get_type_hints(cls)
    #     for name, typeref in hints.items():
    #         fmt.append(typeref.format_string())
    #     return ''.join(fmt)

    # @classmethod
    # def size(cls):
    #     return struct.calcsize(cls.format_string())

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

class LGMMUVNorm(Struct):
    u: float32
    v: float32
    norm: uint32    # compacted normal

class LGMMPolygon(Struct):
    vert: Array(uint16, 3)
    smatr_id: uint16
    d: float32
    norm: uint16
    pad: uint16

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

class LGMMSmatSeg(Struct):
    pgons: uint16
    pgon_start: uint16
    verts: uint16
    vert_start: uint16
    weight_start: uint16 # number of weights = num vertices in segment
    pad: uint16
    smatr_id: uint16
    seg_id: uint16

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
