import bpy
import bmesh
import math
import mathutils
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
    bytes16: '16s',
    }

class Array:
    def __init__(self, typeref, count):
        self.typeref = typeref
        self.count = count

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
        for name, typeref_or_arr in hints.items():
            if isinstance(typeref_or_arr, Array):
                arr = typeref_or_arr
                fmt.append(str(arr.count))
                ch = TYPE_FORMAT_STRINGS[arr.typeref]
                fmt.append(ch)
            else:
                typeref = typeref_or_arr
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
        i = 0
        args = {}
        for name, typeref_or_arr in hints.items():
            if isinstance(typeref_or_arr, Array):
                arr = typeref_or_arr
                args[name] = values[i:i+arr.count]
                i += arr.count
            else:
                args[name] = values[i]
                i += 1
        return cls(**args)

    def write(self):
        fmt = self.format_string()
        hints = get_type_hints(self)
        values = (getattr(self, name) for name in hints.keys())
        return struct.pack(fmt, *values)

class StructView:
    def __init__(self, view, struct_cls, *, offset=0, count=0, size=0):
        if struct_cls is None: raise ValueError("wtf?")
        self.view = view
        if isinstance(struct_cls, type) and issubclass(struct_cls, Struct):
            self.typeref = None
            self.struct_cls = struct_cls
            self.format_string = struct_cls.format_string()
            self.stride = struct_cls.size()
        else:
            self.typeref = struct_cls
            self.struct_cls = None
            self.format_string = TYPE_FORMAT_STRINGS[self.typeref]
            self.stride = struct.calcsize(self.format_string)
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
        if isinstance(i, slice):
            if i.step is not None and i.step != 1:
                raise ValueError("Slices with step size other than 1 are not supported.")
            if i.start >= 0:
                start = min(max(0, i.start), self.count-1)
            else:
                start = min(max(0, self.count+i.start), self.count-1)
            offset = self.offset+start*self.stride
            if i.stop >= 0:
                count = min(self.count, i.stop)
            else:
                count = min(self.count, self.count+i.stop)
            return self.__class__(self.view, (self.struct_cls or self.typeref),
                offset=offset, count=count)
        else:
            if not (0<=i<self.count):
                raise IndexError(i)
            offset = self.offset+i*self.stride
            if self.struct_cls:
                return self.struct_cls.read(self.view, offset=offset)
            else:
                return self.typeref(struct.unpack_from(self.format_string, self.view, offset=offset)[0])

class LGVector(Struct):
    x: float32
    y: float32
    z: float32

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return getattr(self, ['x','y','z'][i])

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
    joint_id: Array(int32, 4)
    pts: Array(LGVector, 4)

class LGCALLimb(Struct):
    torso_id: int32
    bend: int32
    segments: int32
    joint_id: Array(int16, 5)
    seg: Array(LGVector, 4)
    seg_len: Array(float32, 4)

class LGCALFooter(Struct):
    scale: float32

def random_color(alpha=1.0):
    return [random.uniform(0.0, 1.0) for c in "rgb"] + [alpha]
ID_COLOR_TABLE = [random_color() for i in range(1024)]
def id_color(id):
    return ID_COLOR_TABLE[abs(id)%len(ID_COLOR_TABLE)]

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

    # TODO: does this match number of segs? smatrs? anything?
    map_count = header.seg_off-header.map_off
    print(f"maps: {map_count}")
    p_maps = StructView(view, uint8, offset=header.map_off, count=map_count)
    p_segs = StructView(view, LGMMSegment, offset=header.seg_off, count=header.segs)
    p_smatrs = StructView(view, (LGMMSMatrV2 if header.version==2 else LGMMSMatrV1),
        offset=header.smatr_off, count=header.smatrs)
    p_smatsegs = StructView(view, LGMMSmatSeg, offset=header.smatseg_off, count=header.smatsegs)
    p_pgons = StructView(view, LGMMPolygon, offset=header.pgon_off, count=header.pgons)
    p_norms = StructView(view, LGVector, offset=header.norm_off, count=header.pgons) # TODO: is count correct??
    p_verts = StructView(view, LGVector, offset=header.vert_vec_off, count=header.verts)
    p_uvnorms = StructView(view, LGMMUVNorm, offset=header.vert_uvn_off, count=header.verts)
    p_weights = StructView(view, float32, offset=header.weight_off, count=header.verts)

    if header.layout!=0:
        raise NotImplementedError("Not implemented segment-ordered (layout=1) meshes!")

    # Build segment/material/joint tables for later lookup
    segment_by_vert_id = {}
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
                segment_by_vert_id[vi] = smatseg.seg_id
                material_by_vert_id[vi] = smatseg.smatr_id
                joint_by_vert_id[vi] = seg.joint_id

    # Build the bare mesh
    vertices = [tuple(v) for v in p_verts]
    faces = [tuple(p.vert) for p in p_pgons]
    name = f"TEST"
    mesh = bpy.data.meshes.new(f"{name} mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=True)
    print(f"mesh vertices: {len(mesh.vertices)}, loops: {len(mesh.loops)}, polygons: {len(mesh.polygons)}")

    vert_colors = mesh.vertex_colors.new(name="Col")
    seg_colors = mesh.vertex_colors.new(name="SegCol")
    mat_colors = mesh.vertex_colors.new(name="MatCol")
    joint_colors = mesh.vertex_colors.new(name="JointCol")

    for vi, vert in enumerate(mesh.vertices):
        vert_colors.data[vi].color = id_color(vi)
        seg_colors.data[vi].color = id_color( segment_by_vert_id[vi] )
        mat_colors.data[vi].color = id_color( material_by_vert_id[vi] )
        joint_colors.data[vi].color = id_color( joint_by_vert_id[vi] )

    # # Add vertex color layers of info
    # bm = bmesh.new()
    # bm.from_mesh(mesh)
    # print(f"bmesh vertices: {len(bm.verts)}, polygons: {len(bm.faces)}")
    # seg_colors = bm.verts.layers.color.new("SegCol")
    # mat_colors = bm.verts.layers.color.new("MatCol")
    # joint_colors = bm.verts.layers.color.new("JointCol")
    # for vi, vert in enumerate(bm.verts):
    #     seg_colors.color[vi] = id_color( segment_by_vert_id[vi] )
    #     mat_colors.color[vi] = id_color( material_by_vert_id[vi] )
    #     joint_colors.color[vi] = id_color( joint_by_vert_id[vi] )
    # bm.to_mesh(mesh)
    # bm.free()

    # Create the object
    o = bpy.data.objects.new(name, mesh)
    context.scene.collection.objects.link(o)
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
        do_import_mesh(context, self.filename)
        # context.view_layer.objects.active = o
        # o.select_set(True)
        return {'FINISHED'}
