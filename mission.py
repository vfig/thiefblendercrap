import bpy
import math
import mathutils
import os
import sys

from bpy.props import IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from bpy_extras.node_shader_utils import PrincipledBSDFWrapper
from bpy_extras.image_utils import load_image
from collections import OrderedDict
from mathutils import Vector
from typing import Mapping, Sequence
from .binstruct import *
from .images import load_gif, load_pcx
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

class LGTXLISTHeader(Struct):
    length: uint32
    tex_count: uint32
    fam_count: uint32

class LGTXLISTTex(Struct):
    flags: uint8
    fam_id: uint8
    pad: uint16
    name: ByteString(16)

class LGTXLISTFam(Struct):
    name: ByteString(16)

class LGWRHeader(Struct):
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
    tex_u: LGVector
    tex_v: LGVector
    u_base: uint16
    v_base: uint16
    texture_id: uint8
    texture_anchor: uint8
    cached_surface: uint16
    texture_mag: float32
    center: LGVector

class LGWRPlane(Struct):
    normal: LGVector
    distance: float32

class LGWRLightMapInfo(Struct):
    u_base: int16
    v_base: int16
    pixel_width: int16
    height: uint8
    width: uint8
    data_ptr: uint32            # Always zero on disk
    dynamic_light_ptr: uint32   # Always zero on disk
    anim_light_bitmask: uint32

    def entry_count(self):
        light_count = 1
        mask = self.anim_light_bitmask
        while mask!=0:
            if mask&1: light_count += 1
            mask >>= 1
        return self.height*self.pixel_width*light_count


LGWRLightmapEntry = uint16
LGWRRGBLightmapEntry = uint32

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
    p_anim_lights: Sequence[uint16]
    p_light_list: Sequence[LGWRLightMapInfo]
    lightmap_data: Sequence[bytes]
    num_light_indices: int32
    p_light_indices: Sequence[uint16]

    @classmethod
    def read(cls, view, offset=0):
        cell = cls()
        initial_offset = offset
        cell.header = LGWRCellHeader.read(view, offset=offset)
        offset += cell.header.size()
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
        cell.p_anim_lights = StructView(view, uint16, offset=offset, count=cell.header.num_anim_lights)
        offset += cell.p_anim_lights.size()
        cell.p_light_list = StructView(view, LGWRLightMapInfo, offset=offset, count=cell.header.num_render_polys)
        offset += cell.p_light_list.size()
        cell.lightmap_data = []
        for info in cell.p_light_list:
            # WR lightmap data is uint8; WRRGB is uint16 (xR5G5B5)
            entry_type = uint8
            lightmap_size = entry_type.size()*info.entry_count()
            lightmap_data = view[offset:offset+lightmap_size]
            offset += lightmap_size
            cell.lightmap_data.append(lightmap_data)
        cell.num_light_indices = int32.read(view, offset=offset)
        offset += int32.size()
        cell.p_light_indices = StructView(view, uint16, offset=offset, count=cell.num_light_indices)
        offset += cell.p_light_indices.size()
        # Done reading!
        cell._calculated_size = offset-initial_offset
        # Oh, but for sanity, lets build a table of polygon vertices, so
        # we dont have to deal with vertex-indices or vertex-index-indices
        # anywhere else. Except maybe for dumping.
        poly_indices = []
        poly_vertices = []
        start_index = 0
        for pi, poly in enumerate(cell.p_polys):
            indices = []
            vertices = []
            for j in range(poly.num_vertices):
                k = start_index+j
                vi = cell.p_index_list[k]
                indices.append(vi)
                vertices.append(cell.p_vertices[vi])
            start_index += poly.num_vertices
            poly_indices.append(indices)
            poly_vertices.append(vertices)
        cell.poly_indices = poly_indices
        cell.poly_vertices = poly_vertices
        return cell

    def size(self):
        return self._calculated_size

class LGDBChunk:
    header: LGDBChunkHeader
    data: Sequence[bytes]

    def __init__(self, view, offset, data_size):
        self.header = LGDBChunkHeader.read(view, offset=offset)
        offset += self.header.size()
        self.data = view[offset:offset+data_size]

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
        chunk = LGDBChunk(self.view, entry.offset, entry.data_size)
        if byte_str(chunk.header.name) != name:
            raise ValueError(f"{name} chunk name is invalid")
        return chunk

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
    dump_filename = os.path.splitext(filename)[0]+'.dump'
    dumpf = open(dump_filename, 'w')
    #dumpf = sys.stdout

    # Parse the .bin file
    mis = LGDBFile(filename)
    print(f"table_offset: {mis.header.table_offset}", file=dumpf)
    print(f"version: {mis.header.version.major}.{mis.header.version.minor}", file=dumpf)
    print(f"deadbeef: {mis.header.deadbeef!r}", file=dumpf)
    print("Chunks:", file=dumpf)
    for i, name in enumerate(mis):
        print(f"  {i}: {name}", file=dumpf)

    txlist = mis['TXLIST']
    do_txlist(txlist, context, dumpf)
    return # TODO: only go back to worldrep once we can load textures!

    # TODO: WRRGB with t2? what about newdark 32-bit lighting?
    worldrep = mis['WR']
    do_worldrep(worldrep, context, dumpf)

def do_txlist(chunk, context, dumpf):
    if (chunk.header.version.major, chunk.header.version.minor) \
    not in [(1, 0)]:
        raise ValueError("Only version 1.0 TXLIST chunk is supported")
    view = chunk.data
    offset = 0
    header = LGTXLISTHeader.read(view, offset=offset)
    offset += header.size()
    print(f"TXLIST:", file=dumpf)
    print(f"  length: {header.length}", file=dumpf)
    p_fams = StructView(view, LGTXLISTFam, offset=offset, count=header.fam_count)
    offset += p_fams.size()
    print(f"  fam_count: {header.fam_count}", file=dumpf)
    for i, fam in enumerate(p_fams):
        name = byte_str(fam.name)
        print(f"    {i}: {name}", file=dumpf)
    p_texs = StructView(view, LGTXLISTTex, offset=offset, count=header.tex_count)
    offset += p_texs.size()
    print(f"  tex_count: {header.tex_count}", file=dumpf)
    for i, tex in enumerate(p_texs):
        name = byte_str(tex.name)
        print(f"    {i}: fam {tex.fam_id}, {name}, flags 0x{tex.flags:02x}, pad 0x{tex.pad:04x}", file=dumpf)

    # Load all the textures into Blender images (except poor Jorge, who always
    # gets left out):
    tex_search_paths = ['e:/dev/thief/TG1.26/__unpacked/res/fam']
    tex_extensions = ['.dds', '.png', '.tga', '.bmp', '.pcx', '.gif', '.cel']
    ext_sort_order = {ext: i for (i, ext) in enumerate(tex_extensions)}
    def load_tex(fam_name, tex_name):
        fam_name = fam_name.lower()
        tex_name = tex_name.lower()
        # Don't load the image if it has already been loaded.
        image_name = f"fam_{fam_name}_{tex_name}"
        image = bpy.data.images.get(image_name, None)
        if image: return image
        # Find the candidate files (all matching types in all search paths)
        print(f"Searching for fam/{fam_name}/{tex_name}...")
        candidates = [] # (sort_key, full_path) tuples
        for path in tex_search_paths:
            fam_path = os.path.join(path, fam_name)
            print(f"  in path: {fam_path}")
            for entry in os.scandir(fam_path):
                if not entry.is_file(): continue
                name, ext = os.path.splitext(entry.name.lower())
                if name != tex_name: continue
                sort_key = ext_sort_order.get(ext, None)
                if sort_key is None: continue
                print(f"    Candidate: {entry.name}")
                candidates.append((sort_key, entry.path))
        if not candidates:
            raise ValueError(f"Cannot find texture {fam_name}/{tex_name}")
        candidates.sort()
        filename = candidates[0][1]
        # Load the winning file
        print(f"Loading: {filename}...")
        ext = os.path.splitext(filename.lower())[1]
        if ext in ('.png', '.tga', '.bmp'):
            image = bpy.data.images.load(filename)
        elif ext == '.pcx':
            image = load_pcx(filename)
        elif ext == '.gif':
            image = load_gif(filename)
        else:
            raise NotImplementedError(f"{ext} images not yet supported!")
        image.name = image_name
        return image

    textures = []
    for i, tex in enumerate(p_texs):
        if tex.fam_id==0:
            textures.append(None) # TODO: Jorge, is that you?
        else:
            fam = p_fams[tex.fam_id-1]
            fam_name = byte_str(fam.name)
            tex_name = byte_str(tex.name)
            image = load_tex(fam_name, tex_name)
            textures.append(image)
    return textures

def create_lightmap_for_cell_poly(cell, cell_index, pi):
    name = f"Lightmap.Cell{cell_index}.Poly{pi}"
    # TEMP: dont recreate the image if it already exists. this is so
    #       that i can re-run this over and over for getting uv mapping
    #       right, without adding to image/material spam
    #       of course for the actual import we want new images, but then
    #       we should be atlasing already
    image = bpy.data.images.get(name)
    if image is not None: return image
    # Blat the lightmap pixels into an image, very inefficiently.
    info = cell.p_light_list[pi]
    lightmap_data = cell.lightmap_data[pi]
    width = info.width
    height = info.height
    pixels = []
    for y in range(height): # TODO: reversed?
        # TODO: this is all only cromulent for uint8, ofc.
        ofs = y*info.pixel_width
        row = lightmap_data[ofs:ofs+width]
        for b in row:
            i = b/255.0
            pixels.extend([i, i, i, 1.0])
    image = bpy.data.images.new(name, width, height, alpha=True)
    image.pixels = pixels
    return image

def create_material_for_cell_poly(cell, cell_index, pi, lightmap_image):
    # Create a material
    name = f"Lightmap.Cell{cell_index}.Poly{pi}"
    # TEMP: dont recreate the material if it already exists. for the actual
    #       import, we want to be managing materials a little better than
    #       one per poly!
    mat = bpy.data.materials.get(name)
    if mat is not None: return mat
    # Create a material that uses the 'UV' uv_layer for its texcoords
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    principled = PrincipledBSDFWrapper(mat, is_readonly=False)
    principled.base_color = (1.0, 0.5, 0.0) # TODO: temp orange
    principled.base_color_texture.image = lightmap_image
    # TODO: roughness and stuff.
    # could maybe instead wrangle our own DiffuseBSDFWrapper?
    return mat

def poly_calculate_uvs(cell, pi):
    poly = cell.p_polys[pi]
    render = cell.p_render_polys[pi]
    vertices = cell.poly_vertices[pi]
    info = cell.p_light_list[pi]

    # TODO: we dont know the texture dimensions! so hardcode here
    # something that is right in some of the sample:
    tex_width = 64
    tex_height = 64

    # TODO: figure out the uv scales here
    #mxs_real u_scale, v_scale;
    #u_scale = two_to_n(6 - hw.tex->wlog);
    #v_scale = two_to_n(6 - hw.tex->hlog);
    # TODO:
    # wlog, hlog are log2 of the *texture* width/height (possibly scale included)
    # i think.
    # since most of what we have in front of us is 64x64 textures at scale 16,
    # i suppose we try using that...
    # ??? try to find 'correct' values manually
    u_scale = 0.25
    v_scale = 0.5
    # yeah, these are roughly the right scale for these windows
    # (but the offset is wrong, so maybe the scale is slightly off due to
    #  the log2(row) vs log2(w) thing? or i just have the offset calculation
    #  wrong, idk!!)
    #
    # TODO: in any case, because the *texture size* and tex_u/tex_v vectors
    #       are the basis for lightmap uvs as well, it makes sense to get
    #       texture loading happening first!
    #
    p_uvec = Vector(render.tex_u)
    p_vvec = Vector(render.tex_v)
    anchor = Vector(vertices[render.texture_anchor])
    uv = p_uvec.dot(p_vvec)
    u_base = float(render.u_base)*u_scale/(16.0*256.0) # u translation
    v_base = float(render.v_base)*v_scale/(16.0*256.0) # v translation
    u2 = p_uvec.length_squared
    v2 = p_vvec.length_squared
    uv_list = []
    if uv == 0.0:
        uvec = p_uvec*u_scale/u2;
        vvec = p_vvec*v_scale/v2;
        for i in range(poly.num_vertices):
            wvec = Vector(vertices[i])
            delta = wvec-anchor
            u = delta.dot(uvec)+u_base
            v = delta.dot(vvec)+v_base
            uv_list.append((u,v))
    else:
        denom = 1.0/(u2*v2 - (uv*uv));
        u2 *= v_scale*denom
        v2 *= u_scale*denom
        uvu = u_scale*denom*uv
        uvv = v_scale*denom*uv
        for i in range(poly.num_vertices):
            wvec = Vector(vertices[i])
            delta = wvec-anchor
            du = delta.dot(p_uvec)
            dv = delta.dot(p_vvec)
            u = u_base+v2*du-uvu*dv
            v = v_base+u2*dv-uvv*du
            uv_list.append((u,v))
    return uv_list

def do_worldrep(chunk, context, dumpf):
    if (chunk.version.major, chunk.version.minor) \
    not in [(0, 23), (0, 24)]:
        raise ValueError("Only version 0.23 and 0.24 WR chunk is supported")
    view = chunk.data
    offset = 0
    header = LGWRHeader.read(view, offset=offset)
    offset += header.size()
    print(f"WR chunk:", file=dumpf)
    print(f"  version: {chunk.header.version.major}.{chunk.header.version.minor}", file=dumpf)
    print(f"  size: {header.data_size}", file=dumpf)
    print(f"  cell_count: {header.cell_count}", file=dumpf)

    # TODO: import the entire worldrep into one mesh (with options to
    #       skip jorge & sky polys); as we read each cell, append its
    #       vertices and indices (adjusted by a global index total) to
    #       the list.
    #
    #       uv_layer 0 will be for texture coordinate; uv_layer 1 will
    #       be for lightmap coordinates.
    #       this means we will need to construct each lightmap material
    #       with nodes: UV Map -> Image Texture -> BSDF.
    #
    #       as we read each cell, we should pack its lightmap (skipping
    #       the animlight portions for now) into an image texture, and
    #       write out the uv scale+offset that needed. this probably
    #       means managing _multiple_ lightmap textures/materials, if
    #       the lightmaps are too big to fit in a 4Kx4K texture (the max
    #       that newdark permits; i might want to start smaller, too).
    #       feels like this might need to be a post pass (or passes).
    #
    #       however, to get started more easily: begin by just creating
    #       one material per poly (lol), uv_layer 0, and creating an
    #       image texture for its lightmap. this will ensure i can actually
    #       interpret the data correctly, before i try to do it efficiently.

    cells = []
    for cell_index in range(100): # TODO: was range(header.cell_count):
        print(f"Reading cell {cell_index} at offset 0x{offset:08x}")
        cell = LGWRCell.read(view, offset)
        offset += cell.size()
        print(f"  Cell {cell_index}:", file=dumpf)
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
        for i, rpoly in enumerate(cell.p_render_polys):
            print(f"      render_poly {i}:", file=dumpf)
            print(f"        tex_u: {rpoly.tex_u.x:06f},{rpoly.tex_u.y:06f},{rpoly.tex_u.z:06f}", file=dumpf)
            print(f"        tex_v: {rpoly.tex_v.x:06f},{rpoly.tex_v.y:06f},{rpoly.tex_v.z:06f}", file=dumpf)
            print(f"        u_base: {rpoly.u_base} (0x{rpoly.u_base:04x})", file=dumpf)
            print(f"        v_base: {rpoly.v_base} (0x{rpoly.v_base:04x})", file=dumpf)
            print(f"        texture_id: {rpoly.texture_id}", file=dumpf)
            print(f"        texture_anchor: {rpoly.texture_anchor}", file=dumpf)
            # Skip printing  cached_surface, texture_mag, center.
        print(f"    index_count: {cell.index_count}", file=dumpf)
        print(f"    p_index_list: {cell.p_index_list}", file=dumpf)
        for pi, poly in enumerate(cell.p_polys):
            is_render = (pi<cell.header.num_render_polys)
            is_portal = (pi>=(cell.header.num_polys-cell.header.num_portal_polys))
            if is_render and not is_portal: poly_type = 'render'
            elif is_portal and not is_render: poly_type = 'portal'
            else: poly_type = 'render,portal' # typically a water surface
            print(f"      {pi}: ({poly_type})", end='', file=dumpf)
            for j in range(poly.num_vertices):
                vi = cell.poly_indices[pi][j]
                print(f" {vi}", end='', file=dumpf)
            print(file=dumpf)
            print(f"      {pi} uvs:", end='', file=dumpf)
            if is_render:
                uv_list = poly_calculate_uvs(cell, pi)
                for j in range(poly.num_vertices):
                    u, v = uv_list[j]
                    print(f" ({u:0.2f},{v:0.2f}),", end='', file=dumpf)
                print(file=dumpf)
        print(f"    p_plane_list: {cell.p_plane_list}", file=dumpf)
        print(f"    p_anim_lights: {cell.p_anim_lights}", file=dumpf)

        print(f"    p_light_list: {cell.p_light_list}", file=dumpf)
        for i, info in enumerate(cell.p_light_list):
            print(f"      lightmapinfo {i}:", file=dumpf)
            print(f"        u_base: {info.u_base} (0x{info.u_base:04x})", file=dumpf)
            print(f"        v_base: {info.v_base} (0x{info.v_base:04x})", file=dumpf)
            print(f"        pixel_width: {info.pixel_width}", file=dumpf)
            print(f"        height: {info.height}", file=dumpf)
            print(f"        width: {info.width}", file=dumpf)
            print(f"        anim_light_bitmask: 0x{info.anim_light_bitmask:08x}", file=dumpf)
        print(f"    num_light_indices: {cell.num_light_indices}", file=dumpf)
        print(f"    p_light_indices: {cell.p_light_indices}", file=dumpf)

        # TEMP: hack together a mesh
        # TODO: okay, we need to do something about the uvs. first, we are not
        #       yet splitting the vertices. thats... kinda fine, kinda not.
        #       its fine because blender worries about that by putting things
        #       into loops and whatever. but its not fine for the uvs, because
        #       we need to somehow build uvs-by-loop-index, so we need a unique
        #       way of referencing each poly-vertex, because itll have unique
        #       lightmap uvs!
        #       maaaybe (poly_index, vertex_index) is okay, because each loop
        #       references those?? but are those actually unique? idk...
        vertices = [Vector(v) for v in cell.p_vertices]
        uvs = []
        edges = []
        faces = []
        images = []
        materials = []
        for pi, poly in enumerate(cell.p_polys):
            is_render = (pi<cell.header.num_render_polys)
            is_portal = (pi>=(cell.header.num_polys-cell.header.num_portal_polys))
            if not is_render: continue
            poly_uvs = poly_calculate_uvs(cell, pi)
            face = []
            poly_indices = cell.poly_indices[pi]
            for j in range(poly.num_vertices):
                vi = poly_indices[j]
                vi2 = (poly_indices[j+1] if j<(poly.num_vertices-1)
                       else poly_indices[0])
                face.append(vi)
                edges.append((vi, vi2))
                uvs.append(poly_uvs[j])
                # TODO: normals too! the plane normal (or we can 'shade flat' i guess)
            faces.append(face)

            # Hack together a lightmap texture + material for this poly
            image = create_lightmap_for_cell_poly(cell, cell_index, pi)
            images.append(image)
            mat = create_material_for_cell_poly(cell, cell_index, pi, image)
            materials.append(mat)

        name = f"Cell {cell_index}"
        mesh = bpy.data.meshes.new(name=f"{name} mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.validate(verbose=True)
        uv_layer = (mesh.uv_layers.get('UV')
            or mesh.uv_layers.new(name='UV'))
        for loop, uvloop, uv in zip(mesh.loops, uv_layer.data, uvs):
            uvloop.uv = uv
        for i, mat in enumerate(materials):
            mesh.materials.append(mat)
        for i, polygon in enumerate(mesh.polygons):
            polygon.material_index = i
        o = create_object(name, mesh, (0,0,0), context=context, link=True)
