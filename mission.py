import bpy
import math
import mathutils
import numpy as np
import os
import sys

from bpy.props import IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from bpy_extras.node_shader_utils import PrincipledBSDFWrapper
from bpy_extras.image_utils import load_image
from collections import OrderedDict
from dataclasses import dataclass
from mathutils import Vector
from typing import Mapping, Sequence, Tuple
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

        PROFILE = False
        if PROFILE:
            import cProfile
            cProfile.runctx("do_mission(self.filename, context)",
                globals(), locals(), "e:/temp/do_mission.prof")
        else:
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

LGWRPoly_dtype = np.dtype({
    'names':    ['flags',   'num_vertices', 'planeid',  'clut_id',  'destination',  'motion_index', 'padding'],
    'formats':  [ np.uint8,  np.uint8,       np.uint8,   np.uint8,   np.uint16,      np.uint8,       np.uint8],
    'aligned': False,
    })

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
    byte_width: int16
    height: uint8
    width: uint8
    data_ptr: uint32            # Always zero on disk
    dynamic_light_ptr: uint32   # Always zero on disk
    anim_light_bitmask: uint32

    def lightmap_count(self):
        light_count = 1
        mask = self.anim_light_bitmask
        while mask!=0:
            if mask&1: light_count += 1
            mask >>= 1
        return light_count

    def lightmap_size(self):
        return self.height*self.byte_width*self.lightmap_count()


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
    lightmaps: Sequence # of numpy arrays (lightmap_count, height, width, rgba floats)
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

        # hmm... thinking about whether to use numpy to read the raw data or not
        # probably best not! for vertex info, lightmaps, stuff like that, we can
        # use frombuffer and then slap that onto a Big Numpy Array of That Stuff.
        # but for the cell header, theres not much value in it.
        #
        # certainly we dont need all the data in a cell etc. for our purposes.
        # it should suffice to keep only what we need, perhaps in a recarray,
        # perhaps just in a bunch of individual np.arrays in a class (for the
        # convenience of aggregating)...
        #
        # stuff that is variable length we can either pad out to a fixed size
        # (storing the correct count ofc) for operating on; or we can put them
        # all in one big flat-ish array, and store start,end indices for slicing.
        #
        # notes:
        #    max vertices per poly = 32
        #

        old_p_polys = StructView(view, LGWRPoly, offset=offset, count=cell.header.num_polys)
        old_size = old_p_polys.size()
        old_itemsize = LGWRPoly.size()

        raw = np.frombuffer(view, dtype=np.uint8, count=cell.header.num_polys*LGWRPoly_dtype.itemsize, offset=offset)
        new_p_polys = raw.view(dtype=LGWRPoly_dtype, type=np.recarray)
        #new_p_polys = np.frombuffer(view, dtype=LGWRPoly_dtype, count=cell.header.num_polys, offset=offset)
        new_size = new_p_polys.nbytes
        new_itemsize = LGWRPoly_dtype.itemsize

        # print(f"old_p_polys ({old_size} bytes/{old_itemsize} each):\n  ", old_p_polys)
        # print(f"new_p_polys ({new_size} bytes/{new_itemsize} each):\n  ", new_p_polys)
        cell.p_polys = new_p_polys
        offset += new_size

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
        cell.lightmaps = []
        for info in cell.p_light_list:
            # WR lightmap data is uint8; WRRGB is uint16 (xR5G5B5)
            entry_type = uint8
            entry_numpy_type = np.uint8
            width = info.width
            height = info.height
            count = info.lightmap_count()
            lightmap_size = entry_type.size()*info.lightmap_size()
            assert info.byte_width==(info.width*entry_type.size()), "lightmap byte_width is wrong!"
            w = np.frombuffer(view, dtype=entry_numpy_type,
                count=count*height*width, offset=offset)
            offset += lightmap_size
            # Expand the lightmap into rgba floats
            w.shape = (count, height, width, 1)
            w = np.flip(w, axis=1)
            wf = np.array(w, dtype=float)/255.0
            rgbf = np.repeat(wf, repeats=3, axis=3)
            rgbaf = np.insert(rgbf, 3, 1.0, axis=3)
            # TODO: unify the lightmap data types
            cell.lightmaps.append(rgbaf)
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
    textures = do_txlist(txlist, context, dumpf)

    # TODO: WRRGB with t2? what about newdark 32-bit lighting?
    worldrep = mis['WR']
    do_worldrep(worldrep, textures, context, dumpf)

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

def texture_material_for_cell_poly(cell, cell_index, pi, texture_image):
    # Create a material
    name = f"Texture.Cell{cell_index}.Poly{pi}"
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
    principled.base_color_texture.image = texture_image
    principled.base_color_texture.texcoords = 'UV'
    principled.metallic = 0.0
    principled.specular = 0.0
    principled.roughness = 1.0
    return mat

def lightmap_material_for_image(lightmap_image):
    # Create a material that uses the 'Lightmap' uv_layer for its texcoords
    name = f"Lightmap"
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    principled = PrincipledBSDFWrapper(mat, is_readonly=False)
    principled.base_color = (1.0, 0.5, 0.0) # TODO: temp orange
    principled.base_color_texture.image = lightmap_image
    # TODO: this isnt working. it causes a KeyError when the wrapper tries
    #       to set up the uvmap node, or something. i think maybe it can
    #       only work after the mesh/object has been fully set up maybe?
    #principled.base_color_texture.texcoords = 'Lightmap'
    principled.metallic = 0.0
    principled.specular = 0.0
    principled.roughness = 1.0
    return mat

def poly_calculate_texture_uvs(cell, pi, texture_size):
    poly = cell.p_polys[pi]
    render = cell.p_render_polys[pi]
    vertices = cell.poly_vertices[pi]
    u_scale = 64.0/texture_size[0]
    v_scale = 64.0/texture_size[1]
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
            v = 1.0-v # Blender's V coordinate is bottom-up
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
            v = 1.0-v # Blender's V coordinate is bottom-up
            uv_list.append((u,v))
    return uv_list

def poly_calculate_lightmap_uvs(cell, pi, texture_size):
    poly = cell.p_polys[pi]
    render = cell.p_render_polys[pi]
    vertices = cell.poly_vertices[pi]
    info = cell.p_light_list[pi]
    p_uvec = Vector(render.tex_u)
    p_vvec = Vector(render.tex_v)
    anchor = Vector(vertices[render.texture_anchor])
    uv = p_uvec.dot(p_vvec)
    u_scale = 4.0/info.width
    v_scale = 4.0/info.height
    atlas_u0 = 0.5
    atlas_v0 = 0.5
    u_base = u_scale*(float(render.u_base)/(16.0*256.0)+(atlas_u0-float(info.u_base))/4.0) # u translation
    v_base = v_scale*(float(render.v_base)/(16.0*256.0)+(atlas_v0-float(info.v_base))/4.0) # v translation
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
            v = 1.0-v # Blender's V coordinate is bottom-up
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
            v = 1.0-v # Blender's V coordinate is bottom-up
            uv_list.append((u,v))
    return uv_list

def do_worldrep(chunk, textures, context, dumpf):
    if (chunk.header.version.major, chunk.header.version.minor) \
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
    #
    #       So.... because lightmaps are unique, and (i think) aligned with
    #       textures, we could during import merge both lightmaps and texture
    #       samples into one big atlas? that way we end up with unique texels
    #       for each poly, and the combined result will be ready for export
    #       as a single model, without the blender baking shenanigans? dunno.

    cells = []
    for cell_index in range(header.cell_count):
        print(f"Reading cell {cell_index} at offset 0x{offset:08x}")
        cell = LGWRCell.read(view, offset)
        cells.append(cell)
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
        print(f"    p_plane_list: {cell.p_plane_list}", file=dumpf)
        print(f"    p_anim_lights: {cell.p_anim_lights}", file=dumpf)
        print(f"    p_light_list: {cell.p_light_list}", file=dumpf)
        for i, info in enumerate(cell.p_light_list):
            print(f"      lightmapinfo {i}:", file=dumpf)
            print(f"        u_base: {info.u_base} (0x{info.u_base:04x})", file=dumpf)
            print(f"        v_base: {info.v_base} (0x{info.v_base:04x})", file=dumpf)
            print(f"        byte_width: {info.byte_width}", file=dumpf)
            print(f"        height: {info.height}", file=dumpf)
            print(f"        width: {info.width}", file=dumpf)
            print(f"        anim_light_bitmask: 0x{info.anim_light_bitmask:08x}", file=dumpf)
        print(f"    num_light_indices: {cell.num_light_indices}", file=dumpf)
        print(f"    p_light_indices: {cell.p_light_indices}", file=dumpf)

    # We need to build up a single mesh. We need:
    #
    # - Vertex positions: (float, float, float)
    # - Faces: (v-index, v-index, v-index, ...)
    # - Texture UVs: (float, float) per loop (belonging to first vertex)
    # - Lightmap UVs: (float, float) per loop (belonging to first vertex)
    # - Materials: [texture_mat, ..., lightmap_atlas_mat, ...]
    # - Material indices: (int) per face
    #
    # Note on materials:
    #
    #   The maximal set of texture materials is known in advance, as it is
    #   limited to those in the TXLIST (in olddark, a maximum of 256). For
    #   lightmaps, you'd hope they fit in a single atlas, but until you try
    #   to pack them all in, you don't know for sure. And if we bake the
    #   terrain textures and lightmaps together, on a whole map that would
    #   almost certainly require multiple atlas (even though that's an approach
    #   that only makes sense for worldrep->model workflows on not-whole-maps).
    #
    #   Regardless, atlassing lightmaps means we cannot know UVs as we walk
    #   through the worldrep for the first time; instead we need to grab a
    #   handle for the UVs, and give the atlas builder the offset:size of the
    #   lightmap data in the view. Then once the worldrep has been walked, we
    #   can pack the atlas(es), and get the actual material and UVs from the
    #   builder.
    #
    #   And if we are baking terrain textures and lightmaps together, the same
    #   situation applies to terrain textures. This suggests that the builder
    #   should be responsible for collating both terrain textures and lightmaps,
    #   and that we use handles for all UVs and for material indices.

    atlas_builder = AtlasBuilder()
    TEMP_LIMIT = 100 # TODO: use the full range
    # These lists have one entry per cell:
    meshes_name = []
    meshes_vertices = []
    meshes_faces = []
    meshes_texture_materials = []
    meshes_loop_texture_uvs = []
    meshes_loop_lightmap_uvs = []
    meshes_faces_lightmap_handles = []
    meshes_faces_material_indices = []

    for cell_index, cell in enumerate(cells[:TEMP_LIMIT]):
        # TEMP: hack together a mesh
        vertices = [Vector(v) for v in cell.p_vertices]
        texture_uvs = []
        lightmap_uvs = []
        faces = []
        materials = []
        material_indices = []
        lightmap_handles = []
        for pi, poly in enumerate(cell.p_polys):
            is_render = (pi<cell.header.num_render_polys)
            is_portal = (pi>=(cell.header.num_polys-cell.header.num_portal_polys))
            if not is_render: continue

            # TODO: yeah we should reuse materials, but too bad!!
            texture_id = cell.p_render_polys[pi].texture_id
            # Assume Jorge and SKY_HACK (and any invalid texture ids) are 64x64:
            in_range = (0<=texture_id<len(textures))
            special = texture_id in (0,249)
            ok = (in_range and not special)
            texture_image = textures[texture_id] if ok else None
            texture_size = texture_image.size if ok else (64, 64)

            poly_texture_uvs = poly_calculate_texture_uvs(cell, pi, texture_size)
            poly_lightmap_uvs = poly_calculate_lightmap_uvs(cell, pi, texture_size)
            face = []
            poly_indices = cell.poly_indices[pi]
            for j in range(poly.num_vertices):
                vi = poly_indices[j]
                vi2 = (poly_indices[j+1] if j<(poly.num_vertices-1)
                       else poly_indices[0])
                face.append(vi)
            # Reverse face winding order
            face = face[::-1]
            poly_texture_uvs = poly_texture_uvs[::-1]
            poly_lightmap_uvs = poly_lightmap_uvs[::-1]
            texture_uvs.extend(poly_texture_uvs)
            lightmap_uvs.extend(poly_lightmap_uvs)
                # TODO: normals too! the plane normal (or we can 'shade flat' i guess)
            faces.append(face)

            # Hack together a texture material for this poly
            mat_index = pi
            mat = texture_material_for_cell_poly(cell, cell_index, pi, texture_image)
            materials.append(mat)
            material_indices.append(mat_index)

            # Add the primary lightmap for this poly to the atlas.
            info = cell.p_light_list[pi]
            lm_handle = atlas_builder.add(info.width, info.height, cell.lightmaps[pi][0])
            # TODO: handles are one per poly, but uvs are just a flat list for
            #       the cell! so for now, just expand the handles to be one per
            #       poly too.
            lightmap_handles.extend([lm_handle]*len(poly_lightmap_uvs))

        # Store all this cell's data.
        name = f"Cell {cell_index}"
        meshes_name.append(name)
        meshes_vertices.append(vertices)
        meshes_faces.append(faces)
        meshes_texture_materials.append(materials)
        meshes_loop_texture_uvs.append(texture_uvs)
        meshes_loop_lightmap_uvs.append(lightmap_uvs)
        meshes_faces_lightmap_handles.append(lightmap_handles)
        meshes_faces_material_indices.append(material_indices)

    # After all cells complete:
    atlas_builder.close()
    mat_lightmap = lightmap_material_for_image(atlas_builder.image)

    for (name, vertices, faces, materials,
         texture_uvs, lightmap_uvs, lightmap_handles, material_indices) \
    in zip(meshes_name, meshes_vertices, meshes_faces,
           meshes_texture_materials, meshes_loop_texture_uvs,
           meshes_loop_lightmap_uvs, meshes_faces_lightmap_handles,
           meshes_faces_material_indices):

        # Hack together one mesh per cell.
        mesh = bpy.data.meshes.new(name=f"{name} mesh")
        mesh.from_pydata(vertices, [], faces)
        modified = mesh.validate(verbose=True)
        if modified:
            print("Vertices:")
            for i, v in enumerate(vertices):
                print(f"  {i}: {v[0]:0.2f},{v[1]:0.2f},{v[2]:0.2f}")
            print("Faces:")
            for i, f in enumerate(faces):
                print(f"  {i}: {f}")
        assert not modified, f"Mesh {name} pydata was invalid."

        # Transform this poly's lightmap uvs
        lightmap_atlas_uvs = []
        for (u,v), handle in zip(lightmap_uvs, lightmap_handles):
            u, _ = math.modf(u)
            v, _ = math.modf(v)
            if u<0: u = 1.0-u
            if v<0: v = 1.0-v
            scale, translate = atlas_builder.get_uv_transform(handle)
            u = scale[0]*u+translate[0]
            v = scale[1]*v+translate[1]
            lightmap_atlas_uvs.append((u,v))

        # TODO: because i cant figure out how to set up the shader inputs
        #       properly yet, lets just put the lightmap uvs into the
        #       'UV' layer, okay? at least then we can see if the
        #       lightmap uvs are okay or not!
        texture_uvs = lightmap_atlas_uvs[:]

        texture_uv_layer = (mesh.uv_layers.get('UV')
            or mesh.uv_layers.new(name='UV'))
        lightmap_uv_layer = (mesh.uv_layers.get('Lightmap')
            or mesh.uv_layers.new(name='Lightmap'))
        for (loop, tx_uvloop, tx_uv, lm_uvloop, lm_uv) \
        in zip(mesh.loops, texture_uv_layer.data, texture_uvs,
               lightmap_uv_layer.data, lightmap_uvs):
            tx_uvloop.uv = tx_uv
            lm_uvloop.uv = lm_uv

        mesh.materials.append(mat_lightmap)
        for i, mat in enumerate(materials):
            mesh.materials.append(mat)
        for i, (polygon, mat_index) \
        in enumerate(zip(mesh.polygons, material_indices)):
            polygon.material_index = mat_index+1 # 0 is the lightmap
            # TODO: because i cant figure out how to set up the shader inputs
            #       properly yet, lets just give every face the lightmap
            #       material!
            polygon.material_index = 0
        o = create_object(name, mesh, (0,0,0), context=context, link=True)

"""
    # blitting (y, x; remember y is bottom up in blender, which is fine)
    atlas = bpy.data.images.new(name='Atlas', width=256, height=256, alpha=False, float_buffer=False)
    px0 = np.array(im0.pixels)
    px1 = np.array(im1.pixels)
    pxa = np.array(atlas.pixels)
    px0.shape = (64,64,4)
    px1.shape = (64,64,4)
    pxa.shape = (256,256,4)
    pxa[ 0:64, 0:64, : ] = px0
    pxa[ 0:64, 64:128, : ] = px1
    atlas.pixels = pxa.reshape((-1,))
    # equivalent to: pxa.flatten(), but i think .flatten() always copies?

    # reading raw rgb bytes into an array
    raw = open('e:/temp/rails.raw', 'rb').read()
    rgb = np.frombuffer(raw, dtype=np.uint8) # can take count, offset kw
    rgb.shape = (256,256,3)
    # rgb[0][0] is: array([158, 151, 141], dtype=uint8)

    # expanding to rgba
    rgba = np.insert(rgb, 3, 255, axis=2)
    # rgba.shape is: (256, 256, 4)
    # rgba[0][0] is: array([158, 151, 141, 255], dtype=uint8)

    # expanding paletted data to rgb(a):
    # here using rgb ega half-palette with uint8, but this could be rgba floats
    pal = np.array([[0,0,0],[0,0,128],[0,128,0],[0,128,128],[128,0,0],[128,0,128],[128,64,0],[128,128,128]], dtype='uint8')
    # paletted 2x11 image:
    imp = np.array([
        [0,1,2,3,4,5,4,3,2,1,0],
        [7,6,5,4,3,2,3,4,5,6,7]], dtype='uint8')
    # imp.shape is: (2, 11)
    rgb = pal[imp]
    # rgb.shape is: (2, 11, 3)
    # rgb is:
    #     array([[[  0,   0,   0],
    #             [  0,   0, 128],
    #             [  0, 128,   0],
    #             [  0, 128, 128],
    #             [128,   0,   0],
    #             [128,   0, 128],
    #             [128,   0,   0],
    #             [  0, 128, 128],
    #             [  0, 128,   0],
    #             [  0,   0, 128],
    #             [  0,   0,   0]],
    #
    #            [[128, 128, 128],
    #             [128,  64,   0],
    #             [128,   0, 128],
    #             [128,   0,   0],
    #             [  0, 128, 128],
    #             [  0, 128,   0],
    #             [  0, 128, 128],
    #             [128,   0,   0],
    #             [128,   0, 128],
    #             [128,  64,   0],
    #             [128, 128, 128]]], dtype=uint8)

    # for 16-bit 1555 rgb -> 24-bit 888 rgb, probably check out:
    # np.bitwise_and() and np.right_shift()

    # repeating greyscale data into rgb channels:
    a = np.array([1, 2, 3, 4, 5])
    a.shape = (-1, 1)
    # a is:
    #   array([[1],
    #          [2],
    #          [3],
    #          [4],
    #          [5]])
    b = np.repeat(a, repeats=3, axis=1)
    # b is:
    #   array([[1, 1, 1],
    #          [2, 2, 2],
    #          [3, 3, 3],
    #          [4, 4, 4],
    #          [5, 5, 5]])
"""

class AtlasBuilder:
    def __init__(self):
        self.images = []
        # After close, these will be set:
        self.image = None
        self.placements = None

    def add(self, width, height, image):
        # image must be a (height, width, 4) rgbaf array
        rotated = False
        ## TODO: actually rotate the rgbaf data if we want to support rotation!!
        # if height>width:
        #     width,height = height,width
        #     rotated = True
        handle = len(self.images)
        assert image.shape==(height,width,4), f"image {handle} shape is {image.shape}, not {width}x{height}x4!"
        self.images.append((width, height, handle, image, rotated))
        return handle

    def close(self):
        # Build the atlas
        self.images = sorted(self.images, reverse=True)
        atlas_w, atlas_h = (1024, 1024)
        x = y = 0
        row_height = 0
        placements = [None]*len(self.images)
        USE_DEBUG_COLORS = False
        debug_colors = [(r/255.0,g/255.0,b/255.0,1.0) for (r,g,b) in
            [(237,20,91), (246,142,86), (60,184,120), (166,124,82),
            (255,245,104), (109,207,246), (168,100,168), (194,194,194)] ]
        # used_width is the amount of space used by placements in each scanline
        # (from the bottom up).
        #
        # To fit the images into the atlas, we place it at the cursor, if it
        # will fit. If not, we reset the cursor to the left edge, and move it
        # up by the row height; then reset the row height to zero. The row
        # height is the max height of the images placed on that row.
        #
        # The placements are the (x,y,w,h) tuples of the anchor where the
        # corresponding image was placed.
        #
        atlas_data = np.zeros((atlas_w,atlas_h,4), dtype=float)
        for image_index, (w, h, handle, image, rotated) in enumerate(self.images):
            # Find a place for the image.
            if w>atlas_w:
                raise ValueError(f"No space to fit image {handle} of {w}x{h}")
            available_w = atlas_w-x
            if w>available_w:
                x = 0
                y += row_height
                row_height = 0
            available_h = atlas_h-y
            if h>available_h:
                raise ValueError(f"No space to fit image {handle} of {w}x{h}")
            # Place and blit the image.
            print(f"Placing image {handle} of {w}x{h} at {x},{y}.")
            placements[handle] = (x,y,w,h)
            if USE_DEBUG_COLORS:
                source = colors[i%len(debug_colors)]
            else:
                source = image
            atlas_data[ y:y+h, x:x+w ] = source
            # Move the cursor along.
            x += w
            row_height = max(row_height, h)

        # Create the atlas image.
        atlas_image = bpy.data.images.new(name="Atlas", width=atlas_w, height=atlas_h)
        atlas_image.pixels = atlas_data.reshape((-1,))

        self.images = None # Done with the images now.
        self.image = atlas_image
        self.placements = placements

    def get_uv_transform(self, handle):
        """return (scale_x, scale_y), (offset_x, offset_y)"""
        x,y,w,h = self.placements[handle]
        atlas_w,atlas_h = self.image.size
        return ( (w/atlas_w, h/atlas_h), (x/atlas_w, y/atlas_h) )

@dataclass
class TerrainDef:
    tex_id: int

@dataclass
class LightmapDef:
    buffer: bytes   # Typically a memoryview slice.
    #format: LM_WHITE8, LM_RGB555, LM_RGB888

@dataclass
class BuiltMaterial:
    mat_index: int
    uv_offset: Tuple[float, float]
    uv_scale: Tuple[float, float]

class MaterialBuilder:
    """
    Give textures and lightmaps to me, and I will give you a handle. Once
    building is complete, you can then exchange that handle for a material
    index and a UV offset/scale transform.
    """
    # TODO: how do we deal with animlights?
    # TODO: how do we pass in terrain texture info? names/filenames/images/data
    # TODO: how do we pass in options (bake together / not)?
    # TODO: how do we handle Jorge and SKY_HACK?

    def __init__(self):
        self.open = True
        # Handles are indices into these lists:
        self.terrain_defs = [] # one per handle
        self.lightmap_defs = [] # one per handle
        self.terrain_bms = [] # one per handle
        self.lightmap_bms = [] # one per handle

    def add_poly_materials(terrain_def, lightmap_def):
        assert self.open, "Cannot add to MaterialBuilder after close()"
        handle = len(self.terrain_defs)
        self.terrain_defs.append(terrain_def)
        self.lightmap_defs.append(lightmap_def)
        self.terrain_bms.append(BuiltMaterial(0, (0.0,0.0), (1.0,1.0)))
        self.lightmap_bms.append(BuiltMaterial(0, (0.0,0.0), (1.0,1.0)))
        return handle

    def close():
        assert self.open, "Cannot close MaterialBuilder after close()"
        self.open = False
        # TODO: atlas lightmaps, load terrain textures, and so on
        ...

    def get_terrain_material(handle) -> BuiltMaterial:
        return self.terrain_bms[handle]

    def get_lightmap_material(handle) -> BuiltMaterial:
        return self.lightmap_bms[handle]
