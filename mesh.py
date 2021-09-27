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
    def __init__(self, view, typeref, *, offset=0, count=0, size=0):
        self.view = view
        self.typeref = typeref
        self.stride = typeref.size()
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

def random_color(alpha=1.0):
    return [random.uniform(0.0, 1.0) for c in "rgb"] + [alpha]
ID_COLOR_TABLE = [random_color() for i in range(1024)]
def id_color(id):
    return ID_COLOR_TABLE[abs(id)%len(ID_COLOR_TABLE)]

def do_import_mesh(context, bin_filename):
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
    p_maps = StructView(bin_view, uint8, offset=header.map_off, count=map_count)
    p_segs = StructView(bin_view, LGMMSegment, offset=header.seg_off, count=header.segs)
    p_smatrs = StructView(bin_view, (LGMMSMatrV2 if header.version==2 else LGMMSMatrV1),
        offset=header.smatr_off, count=header.smatrs)
    p_smatsegs = StructView(bin_view, LGMMSmatSeg, offset=header.smatseg_off, count=header.smatsegs)
    p_pgons = StructView(bin_view, LGMMPolygon, offset=header.pgon_off, count=header.pgons)
    p_norms = StructView(bin_view, LGVector, offset=header.norm_off, count=header.pgons) # TODO: is count correct??
    p_verts = StructView(bin_view, LGVector, offset=header.vert_vec_off, count=header.verts)
    p_uvnorms = StructView(bin_view, LGMMUVNorm, offset=header.vert_uvn_off, count=header.verts)
    p_weights = StructView(bin_view, float32, offset=header.weight_off, count=header.verts)

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

    print("CAL:")
    print(f"  version: {cal_header.version}")
    print(f"  torsos: {cal_header.torsos}")
    print(f"  limbs: {cal_header.limbs}")
    for i, torso in enumerate(p_torsos):
        print(f"torso {i}:")
        print(f"  joint: {torso.joint}")
        print(f"  parent: {torso.parent}")
        print(f"  fixed_points: {torso.fixed_points}")
        print(f"  joint_id:")
        k = torso.fixed_points
        for joint_id in torso.joint_id[:k]:
            print(f"    {joint_id}")
        print(f"  pts:")
        for pt in torso.pts[:k]:
            print(f"    {pt.x}, {pt.y}, {pt.z}")
    for i, limb in enumerate(p_limbs):
        print(f"limb {i}:")
        print(f"  torso_id: {limb.torso_id}")
        print(f"  bend: {limb.bend}")
        print(f"  segments: {limb.segments}")
        print(f"  joint_id:")
        k = limb.segments
        for joint_id in limb.joint_id[:k+1]:
            print(f"    {joint_id}")
        print(f"  seg:")
        for seg in limb.seg[:k]:
            print(f"    {seg}")
        print(f"  seg_len:")
        for seg_len in limb.seg_len[:k]:
            print(f"    {seg_len}")
    print(f"scale: {cal_footer.scale}")
    print()

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
