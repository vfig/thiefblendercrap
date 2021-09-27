import bpy
import bmesh
import math
import mathutils
import struct
import zlib

from array import array
from dataclasses import dataclass
from bpy.props import IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from typing import NewType, get_type_hints

int8 = NewType('int8', int)
int16 = NewType('int16', int)
int32 = NewType('int32', int)
uint8 = NewType('uint8', int)
uint16 = NewType('uint16', int)
uint32 = NewType('uint32', int)
float32 = NewType('float32', float)
bytes4 = NewType('bytes4', bytes)
bytes16 = NewType('bytes16', bytes)

TYPE_FORMAT_STRINGS = {
    int8: 'b',
    int16: 'h',
    int32: 'l',
    uint8: 'B',
    uint16: 'H',
    uint32: 'L',
    float32: 'f',
    bytes4: '4s',
    }

# TODO: can i turn Struct into a decorate that uses @dataclass and adds the
# read/write/size methods??
# TODO: also rename this because it clashes with struct.Struct haha
class Struct:
    def __init__(self, **kw):
        if self.__class__==Struct:
            raise TypeError("Cannot instantiate Struct itself, only subclasses")
        hints = get_type_hints(self.__class__)
        if len(hints)==0:
            raise TypeError(f"{self.__class__.__name__} has no fields defined")
        for name, typeref in hints.items():
            if name not in kw:
                raise KeyError(name)
            # default = typeref.__supertype__()
            # setattr(self, name, default)
        for name, value in kw.items():
            if name not in hints:
                raise KeyError(name)
            setattr(self, name, typeref(value))

    @classmethod
    def format_string(cls):
        fmt = ['@']
        hints = get_type_hints(cls)
        for name, typeref in hints.items():
            ch = TYPE_FORMAT_STRINGS[typeref]
            fmt.append(ch)
        return ''.join(fmt)

    @classmethod
    def size(cls):
        fmt = cls.format_string()
        return struct.calcsize(fmt)

    @classmethod
    def read(cls, data, offset=0):
        fmt = cls.format_string()
        hints = get_type_hints(cls)
        values = struct.unpack_from(fmt, data, offset=offset)
        args = dict(zip(hints.keys(), values))
        return cls(**args)

    def write(self):
        fmt = self.format_string()
        hints = get_type_hints(self)
        values = (getattr(self, name) for name in hints.keys())
        return struct.pack(fmt, *values)

class StructView:
    def __init__(self, view, struct_cls, *, offset=0, count=0, size=0):
        self.view = view
        self.struct_cls = struct_cls
        self.format_string = struct_cls.format_string()
        self.stride = struct_cls.size()
        self.offset = offset
        if count and size:
            raise ValueError("Must provide either count, or size, or neither (to use entire view)")
        if count:
            self.count = count
        elif size:
            self.count = size//self.stride
        else:
            self.count = len(view)//self.stride

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        if not (0<=i<self.count):
            raise IndexError(i)
        offset = self.offset+i*self.stride
        return self.struct_cls.read(self.view, offset=offset)

class LGVector(Struct):
    x: float32
    y: float32
    z: float32

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
    v0: uint16
    v1: uint16
    v2: uint16
    smatr_id: uint16
    d: float32
    norm: uint16
    pad: uint16

class MM_SMatrV1(Struct):
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

class MM_SMatrV2(Struct):
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
    pgons: uint16       # from here, only for segment order
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
    joint_id: int32[4]
    pts: LGVector[4]

class LGCALLimb(Struct):
    torso_id: int32
    bend: int32
    segments: int32
    joint_id: int16[5]
    seg: LGVector[4]
    seg_len: float32[4]

class LGCALFooter(Struct):
    scale: float32


def do_import_mesh(context, filename):
    with open(filename, 'rb') as f:
        data = f.read()
    view = memoryview(data)
    header = LGMMHeader.read(view)
    print(f"magic: {header.magic}")
    print(f"version: {header.version}")
    print(f"radius: {header.radius:f}")
    print(f"flags: {header.flags:04x}")
    print(f"app_data: {header.app_data:04x}")
    print(f"layout: {header.layout}")
    print(f"segs: {header.segs}")
    print(f"smatrs: {header.smatrs}")
    print(f"smatsegs: {header.smatsegs}")
    print(f"pgons: {header.pgons}")
    print(f"verts: {header.verts}")
    print(f"weights: {header.weights}")
    print(f"map_off: {header.map_off:08x}")
    print(f"seg_off: {header.seg_off:08x}")
    print(f"smatr_off: {header.smatr_off:08x}")
    print(f"smatseg_off: {header.smatseg_off:08x}")
    print(f"pgon_off: {header.pgon_off:08x}")
    print(f"norm_off: {header.norm_off:08x}")
    print(f"vert_vec_off: {header.vert_vec_off:08x}")
    print(f"vert_uvn_off: {header.vert_uvn_off:08x}")
    print(f"weight_off: {header.weight_off:08x}")
    print()

    # VERTEXES:

    p_verts = StructView(view, LGVector, offset=header.vert_vec_off, count=header.verts)
    print("VERTS:")
    print(f"count: {len(p_verts)}")
    vertices = []
    # bb_min = LGVector(x=9999,y=9999,z=9999)
    # bb_max = LGVector(x=-9999,y=-9999,z=-9999)
    for i, vert in enumerate(p_verts):
        # if vert.x>bb_max.x: bb_max.x = vert.x
        # if vert.y>bb_max.y: bb_max.y = vert.y
        # if vert.z>bb_max.z: bb_max.z = vert.z
        # if vert.x<bb_min.x: bb_min.x = vert.x
        # if vert.y<bb_min.y: bb_min.y = vert.y
        # if vert.z<bb_min.z: bb_min.z = vert.z
        v = (vert.x,vert.y,vert.z)
        vertices.append(v)
        if i<20: print(v)
    # print(f"BBox min: {bb_min.x}, {bb_min.y}, {bb_min.z}")
    # print(f"     max: {bb_max.x}, {bb_max.y}, {bb_max.z}")

    # POLYGONS:
    p_pgons = StructView(view, LGMMPolygon, offset=header.pgon_off, count=header.pgons)
    print("PGONS:")
    print(f"count: {len(p_pgons)}")
    faces = []
    for i, pgon in enumerate(p_pgons):
        f = (pgon.v0, pgon.v1, pgon.v2)
        faces.append(f)
        if i<20: print(f)

    name = "TEST"
    mesh = bpy.data.meshes.new(f"{name} mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=True)
    o = bpy.data.objects.new(name, mesh)
    #o.display_type = 'WIRE'
    return o

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
        o = do_import_mesh(context, self.filename)
        context.scene.collection.objects.link(o)
        context.view_layer.objects.active = o
        o.select_set(True)
        return {'FINISHED'}
