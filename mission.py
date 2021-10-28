import bpy
import math
import mathutils
import os
import sys

from bpy.props import IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from collections import OrderedDict
from mathutils import Vector
from typing import Mapping, Sequence
from .binstruct import *
from .lgtypes import *

def create_object(name, mesh, location, context=None, link=True):
    o = bpy.data.objects.new(name, mesh)
    o.location = location
    if context and link:
        coll = context.view_layer.active_layer_collection.collection
        coll.objects.link(o)
    return o

#---------------------------------------------------------------------------#
# Operators

class TTDebugImportMissionOperator(Operator):
    bl_idname = "object.tt_debug_import_mission"
    bl_label = "Import mission"
    bl_options = {'REGISTER'}

    filename : StringProperty()

    def execute(self, context):
        if context.mode != "OBJECT":
            self.report({'WARNING'}, f"{self.bl_label}: must be in Object mode.")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')

        print(f"filename: {self.filename}")

        do_mission(self.filename, context)

        #context.view_layer.objects.active = o
        #o.select_set(True)

        return {'FINISHED'}

def byte_str(b):
    return b.partition(b'\x00')[0].decode('ascii')

class LGDBVersion(Struct):
    major: uint32
    minor: uint32

class LGDBFileHeader(Struct):
    table_offset: uint32
    version: LGDBVersion
    pad: Array(uint8, 256)
    deadbeef: ByteString(4)

class LGDBTOCEntry(Struct):
    name: ByteString(12)
    offset: uint32
    data_size: uint32

class LGDBChunkHeader(Struct):
    name: ByteString(12)
    version: LGDBVersion
    pad: uint32

class LGWRHeader(Struct):
    chunk: LGDBChunkHeader
    data_size: uint32
    cell_count: uint32

class LGWRCellHeader(Struct):
    num_vertices: uint8
    num_polys: uint8
    num_render_polys: uint8
    num_portal_polys: uint8
    num_planes: uint8
    medium: uint8
    flags: uint8
    portal_vertex_list: int32
    num_vlist: uint16
    num_anim_lights: uint8
    motion_index: uint8
    sphere_center: LGVector
    sphere_radius: float32

class LGWRPoly(Struct):
    flags: uint8
    num_vertices: uint8
    planeid: uint8
    clut_id: uint8
    destination: uint16
    motion_index: uint8
    padding: uint8

class LGWRRenderPoly(Struct):
    uv: Array(LGVector, 2)
    uv_base: Array(uint16, 2)
    texture_id: uint8
    texture_anchor: uint8
    cached_surface: uint16
    texture_mag: float32
    center: LGVector

class LGWRPlane(Struct):
    normal: LGVector
    distance: float32

class LGWRCell:
    # Note: this is _not_ a Struct subclass, because its array sizes are
    #       dynamic based on its other field values. So these type hints
    #       exist only for your benefit.
    header: LGWRCellHeader
    p_vertices: Sequence[LGVector]
    p_polys: Sequence[LGWRPoly]
    p_render_polys: Sequence[LGWRRenderPoly]
    vertex_offset: uint32
    p_vertex_list: Sequence[uint8]
    p_plane_list: Sequence[LGWRPlane]
    p_anim_lights: Sequence[LGWRPlane]
    # TODO: the rest!

    @classmethod
    def read(cls, view, offset=0):
        cell = cls()
        initial_offset = offset
        cell.header = LGWRCellHeader.read(view, offset=offset)
        offset += cell.header.size()
        #cell.vertices = Array(LGVector, cell.header.num_vertices).read(view, offset=offset)
        #offset += cell.vertices.size()
        cell.p_vertices = StructView(view, LGVector, offset=offset, count=cell.header.num_vertices)
        offset += cell.p_vertices.size()
        cell.p_polys = StructView(view, LGWRPoly, offset=offset, count=cell.header.num_polys)
        offset += cell.p_polys.size()
        cell.p_render_polys = StructView(view, LGWRRenderPoly, offset=offset, count=cell.header.num_render_polys)
        offset += cell.p_render_polys.size()
        cell.index_count = uint32.read(view, offset=offset)
        offset += uint32.size()
        cell.p_index_list = StructView(view, uint8, offset=offset, count=cell.index_count)
        offset += cell.p_index_list.size()
        cell.p_plane_list = StructView(view, LGWRPlane, offset=offset, count=cell.header.num_planes)
        offset += cell.p_plane_list.size()
        cell.p_anim_lights = StructView(view, LGWRPlane, offset=offset, count=cell.header.num_anim_lights)
        offset += cell.p_anim_lights.size()
        # TODO: light list and such... phew! this is a messy structure!
        cell._calculated_size = offset-initial_offset
        return cell

    def size(self):
        return self._calculated_size

class LGDBFile:
    header: LGDBFileHeader

    def __init__(self, filename='', data=None):
        if data is None:
            with open(filename, 'rb') as f:
                data = f.read()
        view = memoryview(data)
        # Read the header.
        offset = 0
        header = LGDBFileHeader.read(view)
        if header.deadbeef != b'\xDE\xAD\xBE\xEF':
            raise ValueError("File is not a .mis/.cow/.gam/.vbr")
        if (header.version.major, header.version.minor) not in [(0, 1)]:
            raise ValueError("Only version 0.1 .mis/.cow/.gam/.vbr files are supported")
        # Read the table of contents.
        offset = header.table_offset
        toc_count = uint32.read(view, offset=offset)
        offset += uint32.size()
        p_entries = StructView(view, LGDBTOCEntry, offset=offset, count=toc_count)
        toc = OrderedDict()
        for entry in p_entries:
            key = byte_str(entry.name)
            toc[key] = entry

        self.header = header
        self.toc = toc
        self.view = view

    def __len__(self):
        return len(self.toc)

    def __getitem__(self, name):
        entry = self.toc[name]
        return self.view[entry.offset:entry.offset+entry.data_size]

    def __iter__(self):
        return iter(self.toc.keys())

    def get(self, name, default=None):
        try:
            return self.__getitem__(name)
        except KeyError:
            return default

    def keys(self):
        return self.toc.keys()

    def values(self):
        for name in self.toc.keys():
            yield self[name]

    def items(self):
        return zip(self.keys(), self.values())

def hex_str(bytestr):
    return " ".join(format(b, "02x") for b in bytestr)

def do_mission(filename, context):
    #dump_filename = os.path.splitext(mis_filename)[0]+'.dump'
    #dumpf = open(dump_filename, 'w')
    dumpf = sys.stdout

    # Parse the .bin file
    mis = LGDBFile(filename)
    print(f"table_offset: {mis.header.table_offset}", file=dumpf)
    print(f"version: {mis.header.version.major}.{mis.header.version.minor}", file=dumpf)
    print(f"deadbeef: {mis.header.deadbeef!r}", file=dumpf)
    print("Chunks:")
    for i, name in enumerate(mis):
        print(f"  {i}: {name}", file=dumpf)

    foo = mis['FILE_TYPE']
    print(hex_str(foo))

    # TODO: WRRGB with t2?
    worldrep = mis['WR']
    # other useful chunks maybe: FAMILY, AMBIENT, DARKMISS
    do_worldrep(worldrep, context)

def do_worldrep(view, context):
    dumpf = sys.stdout

    offset = 0
    header = LGWRHeader.read(view, offset=offset)
    offset += header.size()
    if byte_str(header.chunk.name) != 'WR':
        raise ValueError("WR chunk name is invalid")
    if (header.chunk.version.major, header.chunk.version.minor) \
    not in [(0, 23), (0, 24)]:
        raise ValueError("Only version 0.23 and 0.24 WR chunk is supported")
    print(f"WR chunk:", file=dumpf)
    print(f"  version: {header.chunk.version.major}.{header.chunk.version.minor}", file=dumpf)
    print(f"  size: {header.data_size}", file=dumpf)
    print(f"  cell_count: {header.cell_count}", file=dumpf)

    cells = []
    for i in range(1): # TODO: header.cell_count
        cell = LGWRCell.read(view, offset)
        offset += cell.size()
        print(f"  Cell {i}:", file=dumpf)
        print(f"    num_vertices: {cell.header.num_vertices}", file=dumpf)
        print(f"    num_polys: {cell.header.num_polys}", file=dumpf)
        print(f"    num_render_polys: {cell.header.num_render_polys}", file=dumpf)
        print(f"    num_portal_polys: {cell.header.num_portal_polys}", file=dumpf)
        print(f"    num_planes: {cell.header.num_planes}", file=dumpf)
        print(f"    medium: {cell.header.medium}", file=dumpf)
        print(f"    flags: {cell.header.flags}", file=dumpf)
        print(f"    portal_vertex_list: {cell.header.portal_vertex_list}", file=dumpf)
        print(f"    num_vlist: {cell.header.num_vlist}", file=dumpf)
        print(f"    num_anim_lights: {cell.header.num_anim_lights}", file=dumpf)
        print(f"    motion_index: {cell.header.motion_index}", file=dumpf)
        print(f"    sphere_center: {cell.header.sphere_center}", file=dumpf)
        print(f"    sphere_radius: {cell.header.sphere_radius}", file=dumpf)
        print(f"    p_vertices: {cell.p_vertices}", file=dumpf)
        for i, v in enumerate(cell.p_vertices):
            print(f"      {i}: {v.x:06f},{v.y:06f},{v.z:06f}", file=dumpf)
        print(f"    p_polys: {cell.p_polys}", file=dumpf)
        print(f"    p_render_polys: {cell.p_render_polys}", file=dumpf)
        print(f"    index_count: {cell.index_count}", file=dumpf)
        print(f"    p_index_list: {cell.p_index_list}", file=dumpf)
        poly_start_index = 0
        for i, poly in enumerate(cell.p_polys):
            is_render = (i<cell.header.num_render_polys)
            is_portal = (i>=(cell.header.num_polys-cell.header.num_portal_polys))
            # Hmm, not sure about this. This comes from DarkUtils writeup,
            # but it claims "surface" (here "render") polys are only for
            # translucent polys like water surfaces. BUT its only the render
            # poly struct that has uvs? i dunno.
            if is_render: poly_type = 'render'
            elif is_portal: poly_type = 'portal'
            else: poly_type = 'wall'
            print(f"      {i}: ({poly_type})", end='', file=dumpf)
            for j in range(poly.num_vertices):
                k = poly_start_index+j
                vi = cell.p_index_list[k]
                print(f" {vi}", end='', file=dumpf)
            print(file=dumpf)
            poly_start_index += poly.num_vertices
        print(f"    p_plane_list: {cell.p_plane_list}", file=dumpf)
        print(f"    p_anim_lights: {cell.p_anim_lights}", file=dumpf)

        # TEMP: hack together a mesh
        vertices = [Vector(v) for v in cell.p_vertices]
        faces = []
        poly_start_index = 0
        for i, poly in enumerate(cell.p_polys):
            is_render = (i<cell.header.num_render_polys)
            is_portal = (i>=(cell.header.num_polys-cell.header.num_portal_polys))
            face = []
            for j in range(poly.num_vertices):
                k = poly_start_index+j
                vi = cell.p_index_list[k]
                face.append(vi)
            faces.append(face)
            poly_start_index += poly.num_vertices
        name = "Cell"
        mesh = bpy.data.meshes.new(f"{name} mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.validate(verbose=True)
        create_object(name, mesh, (0,0,0), context=context, link=True)
