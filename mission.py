import bpy
import math
import mathutils
import numpy as np
import os
import sys

from bpy.props import IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from bpy_extras.image_utils import load_image
from collections import OrderedDict
from dataclasses import dataclass
from mathutils import Vector
from numpy import (int8, int16, int32, uint8, uint16, uint32, float32)
from typing import Mapping, Sequence, Tuple
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

class StructuredReader:
    def __init__(self, filename='', mode='rb', buffer=None):
        if filename=='' and buffer is None:
            raise ValueError("One of 'filename' and 'buffer' must be given.")
        if filename!='' and buffer is not None:
            raise ValueError("Only one of 'filename' and 'buffer' must be given.")
        if filename!='':
            a = np.fromfile(filename, dtype=uint8)
        else:
            a = np.frombuffer(buffer, dtype=uint8)
        a.flags.writeable = False
        self.array = a
        self.offset = 0
        self.size = a.size

    def seek(self, offset):
        if offset<0:
            self.offset = self.size+self.offset
        else:
            self.offset = offset

    def read(self, dtype, count=None, peek=False):
        dtype = np.dtype(dtype)
        itemsize = dtype.itemsize
        if count is None:
            want_array = False
            count = 1
        else:
            want_array = True
        start = self.offset
        end = start+count*itemsize

        # TODO: all this is to guard against me accidentally making a foolish copy.
        def root_base(arr):
            if arr.base is None:
                return arr
            base = arr.base
            while base.base is not None:
                base = base.base
            return base
        root = root_base(self.array)

        a = self.array[start:end]
        assert (root_base(a) is root), "Oops, made a copy!"
        a = np.frombuffer(a, dtype=dtype, count=count)
        a = a.view(type=np.recarray)
        assert (root_base(a) is root), "Oops, made a copy!"
        if not peek:
            self.offset = end
        if not want_array:
            a = a[0]
        return a

def structure(cls, aligned=False):
    import sys
    dumpf = sys.stderr
    from typing import get_type_hints
    hints = get_type_hints(cls)
    if len(hints)==0:
        raise TypeError(f"{cls.__name__} has no fields defined")
    names = []
    formats = []
    for name, typeref in hints.items():
        names.append(name)
        formats.append(typeref)
    dtype = np.dtype({'names': names, 'formats': formats, 'aligned': aligned})
    for field_name, (field_dtype, field_shape) in dtype.fields.items():
        assert field_dtype.kind!='O', f"Field {cls.__name__}:{field_name} must not be an object type"
    # structure size is: dtype.itemsize
    return dtype

def ascii(b):
    return b.partition(b'\x00')[0].decode('ascii')

def bit_count(n):
    c = 0
    while n!=0:
        if n&1: c += 1
        n >>= 1
    return c

LGVector = (float32, 3)

@structure
class LGDBVersion:
    major: uint32
    minor: uint32

@structure
class LGDBFileHeader:
    table_offset: uint32
    version: LGDBVersion
    pad: (uint8, 256)
    deadbeef: (bytes, 4)

@structure
class LGDBTOCEntry:
    name: (bytes, 12)
    offset: uint32
    data_size: uint32

@structure
class LGDBChunkHeader:
    name: (bytes, 12)
    version: LGDBVersion
    pad: uint32

@structure
class LGTXLISTHeader:
    length: uint32
    tex_count: uint32
    fam_count: uint32

@structure
class LGTXLISTTex:
    flags_: uint8
    fam_id: uint8
    pad: uint16
    name: (bytes, 16)

@structure
class LGTXLISTFam:
    name: (bytes, 16)

@structure
class LGWRHeader:
    data_size: uint32
    cell_count: uint32

@structure
class LGWRCellHeader:
    num_vertices: uint8
    num_polys: uint8
    num_render_polys: uint8
    num_portal_polys: uint8
    num_planes: uint8
    medium: uint8
    flags_: uint8
    portal_vertex_list: int32
    num_vlist: uint16
    num_anim_lights: uint8
    motion_index: uint8
    sphere_center: LGVector
    sphere_radius: float32

@structure
class LGWRPoly:
    flags_: uint8
    num_vertices: uint8
    planeid: uint8
    clut_id: uint8
    destination: uint16
    motion_index: uint8
    padding: uint8

@structure
class LGWRRenderPoly:
    tex_u: LGVector
    tex_v: LGVector
    u_base: uint16
    v_base: uint16
    texture_id: uint8
    texture_anchor: uint8
    cached_surface: uint16
    texture_mag: float32
    center: LGVector

@structure
class LGWRPlane:
    normal: LGVector
    distance: float32

@structure
class LGWRLightMapInfo:
    u_base: int16
    v_base: int16
    byte_width: int16
    height: uint8
    width: uint8
    data_ptr: uint32            # Always zero on disk
    dynamic_light_ptr: uint32   # Always zero on disk
    anim_light_bitmask: uint32

# TODO: am i even using these right now?
LGWRLightmapEntry = uint16
LGWRRGBLightmapEntry = uint32

class LGWRCell:
    # # Note: this is _not_ a Struct subclass, because its array sizes are
    # #       dynamic based on its other field values. So these type hints
    # #       exist only for your benefit.
    # header: LGWRCellHeader
    # p_vertices: Sequence[LGVector]
    # p_polys: Sequence[LGWRPoly]
    # p_render_polys: Sequence[LGWRRenderPoly]
    # vertex_offset: uint32
    # p_vertex_list: Sequence[uint8]
    # p_plane_list: Sequence[LGWRPlane]
    # p_anim_lights: Sequence[uint16]
    # p_light_list: Sequence[LGWRLightMapInfo]
    # lightmaps: Sequence # of numpy arrays (lightmap_count, height, width, rgba floats)
    # num_light_indices: int32
    # p_light_indices: Sequence[uint16]

    @classmethod
    def read(cls, reader, cell_index):
        f = reader
        cell = cls()
        cell.header = f.read(LGWRCellHeader)
        cell.p_vertices = f.read(LGVector, count=cell.header.num_vertices)
        cell.p_polys = f.read(LGWRPoly, count=cell.header.num_polys)
        cell.p_render_polys = f.read(LGWRRenderPoly, count=cell.header.num_render_polys)
        cell.index_count = f.read(uint32)
        cell.p_index_list = f.read(uint8, count=cell.index_count)
        cell.p_plane_list = f.read(LGWRPlane, count=cell.header.num_planes)
        cell.p_anim_lights = f.read(uint16, count=cell.header.num_anim_lights)
        cell.p_light_list = f.read(LGWRLightMapInfo, count=cell.header.num_render_polys)
        cell.lightmaps = []
        for info in cell.p_light_list:
            # WR lightmap data is uint8; WRRGB is uint16 (xR5G5B5)
            entry_type = np.dtype(uint8)
            width = info.width
            height = info.height
            count = 1+bit_count(info.anim_light_bitmask) # 1 base lightmap, plus 1 per animlight
            assert info.byte_width==(info.width*entry_type.itemsize), "lightmap byte_width is wrong!"
            w = f.read(entry_type, count=count*height*width)
            # Expand the lightmap into rgba floats
            w.shape = (count, height, width, 1)
            w = np.flip(w, axis=1)
            wf = np.array(w, dtype=float)/255.0
            rgbf = np.repeat(wf, repeats=3, axis=3)
            rgbaf = np.insert(rgbf, 3, 1.0, axis=3)
            # TODO: unify the lightmap data types
            cell.lightmaps.append(rgbaf)
        cell.num_light_indices = f.read(int32)
        cell.p_light_indices = f.read(uint16, count=cell.num_light_indices)
        # Done reading!
        # Oh, but for sanity, lets build a table of polygon vertices, so
        # we dont have to deal with vertex-index-indices anywhere else.
        poly_indices = []
        poly_vertices = []
        start_index = 0
        for pi, poly in enumerate(cell.p_polys):
            assert poly.num_vertices<=32, "you fucked up, poly has >32 verts!"
            index_indices = np.arange(start_index,start_index+poly.num_vertices)
            indices = cell.p_index_list[index_indices]
            vertices = cell.p_vertices[indices]
            start_index += poly.num_vertices
            poly_indices.append(indices)
            poly_vertices.append(vertices)
        cell.poly_indices = poly_indices
        cell.poly_vertices = poly_vertices
        return cell

class LGDBChunk:
    def __init__(self, f, offset, data_size):
        f.seek(offset)
        self.header = f.read(LGDBChunkHeader)
        self.data = f.read(uint8, count=data_size)

class LGDBFile:
    def __init__(self, filename='', data=None):
        f = StructuredReader(filename=filename, buffer=data)
        # Read the header.
        header = f.read(LGDBFileHeader)
        if header.deadbeef != b'\xDE\xAD\xBE\xEF':
            raise ValueError("File is not a .mis/.cow/.gam/.vbr")
        if (header.version.major, header.version.minor) not in [(0, 1)]:
            raise ValueError("Only version 0.1 .mis/.cow/.gam/.vbr files are supported")
        # Read the table of contents.
        f.seek(header.table_offset)
        toc_count = f.read(uint32)
        p_entries = f.read(LGDBTOCEntry, count=toc_count)
        toc = OrderedDict()
        for entry in p_entries:
            key = ascii(entry.name)
            toc[key] = entry

        self.header = header
        self.toc = toc
        self.reader = f

    def __len__(self):
        return len(self.toc)

    def __getitem__(self, name):
        entry = self.toc[name]
        chunk = LGDBChunk(self.reader, entry.offset, entry.data_size)
        if ascii(chunk.header.name) != name:
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
    do_worldrep(worldrep, textures, context, options={
        'dump': False,
        'dump_file': sys.stdout,
        'cell_limit': 0,
        })

def do_txlist(chunk, context, dumpf):
    if (chunk.header.version.major, chunk.header.version.minor) \
    not in [(1, 0)]:
        raise ValueError("Only version 1.0 TXLIST chunk is supported")
    f = StructuredReader(buffer=chunk.data)
    header = f.read(LGTXLISTHeader)
    print(f"TXLIST:", file=dumpf)
    print(f"  length: {header.length}", file=dumpf)
    p_fams = f.read(LGTXLISTFam, count=header.fam_count)
    print(f"  fam_count: {header.fam_count}", file=dumpf)
    for i, fam in enumerate(p_fams):
        name = ascii(fam.name)
        print(f"    {i}: {name}", file=dumpf)
    p_texs = f.read(LGTXLISTTex, count=header.tex_count)
    print(f"  tex_count: {header.tex_count}", file=dumpf)
    for i, tex in enumerate(p_texs):
        name = ascii(tex.name)
        print(f"    {i}: fam {tex.fam_id}, {name}, flags_ 0x{tex.flags_:02x}, pad 0x{tex.pad:04x}", file=dumpf)

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
            fam_name = ascii(fam.name)
            tex_name = ascii(tex.name)
            image = load_tex(fam_name, tex_name)
            textures.append(image)
    return textures

def create_texture_material(name, texture_image, lightmap_image):
    is_textured = (texture_image is not None)
    is_lightmapped = (lightmap_image is not None)
    #
    # If lightmapped and textured:
    #
    #       UVMap('UVMap')              UVMap('UVLightmap')
    #             |                               |
    # ImageTexture(texture_image)     ImageTexture(lightmap_image)
    #             |                               |
    #             +----------------+--------------+
    #                              |
    #                     MixRGB('Multiply')
    #                              |
    #                        PrincipledBSDF
    #                              |
    #                        MaterialOutput
    #
    # If only textured (only lightmapped is a similar structure):
    #
    #                        UVMap('UVMap')
    #                              |
    #                  ImageTexture(texture_image)
    #                              |
    #                        PrincipledBSDF
    #                              |
    #                        MaterialOutput
    #
    # If neither textured nor lightmapped:
    #
    #                        PrincipledBSDF
    #                              |
    #                        MaterialOutput
    #
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    # Create all the nodes
    bsdf_node = None
    out_node = None
    for n in nodes:
        if n.bl_idname=='ShaderNodeBsdfPrincipled': bsdf_node = n
        elif n.bl_idname=='ShaderNodeOutputMaterial': out_node = n
    if out_node is None: out_node = nodes.new(type='ShaderNodeOutputMaterial')
    if bsdf_node is None: bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    if is_textured:
        tx_uv_node = nodes.new(type='ShaderNodeUVMap')
        tx_img_node = nodes.new(type='ShaderNodeTexImage')
    if is_lightmapped:
        lm_uv_node = nodes.new(type='ShaderNodeUVMap')
        lm_img_node = nodes.new(type='ShaderNodeTexImage')
    if is_textured and is_lightmapped:
        mix_node = nodes.new(type='ShaderNodeMixRGB')
    # Configure them
    bsdf_node.inputs['Base Color'].default_value = (1.0,0.0,1.0,1.0)
    bsdf_node.inputs['Metallic'].default_value = 0.0
    bsdf_node.inputs['Specular'].default_value = 0.0
    bsdf_node.inputs['Roughness'].default_value = 1.0
    if is_textured:
        tx_img_node.image = texture_image
        tx_uv_node.uv_map = 'UVMap'
    if is_lightmapped:
        lm_img_node.image = lightmap_image
        lm_uv_node.uv_map = 'UVLightmap'
    if is_textured and is_lightmapped:
        mix_node.blend_type = 'MULTIPLY'
        mix_node.inputs['Fac'].default_value = 1.0
        mix_node.use_clamp = True
    # Place them
    def grid(x,y):
        return (x*400.0,y*200.0)
    out_node.location = grid(1,0)
    bsdf_node.location = grid(0,0)
    if is_lightmapped and is_textured:
        mix_node.location = grid(-1,0)
        tx_img_node.location = grid(-2,1)
        tx_uv_node.location = grid(-3,1)
        lm_img_node.location = grid(-2,-1)
        lm_uv_node.location = grid(-3,-1)
    elif is_textured:
        tx_img_node.location = grid(-1,0)
        tx_uv_node.location = grid(-2,0)
    elif is_lightmapped:
        lm_img_node.location = grid(-1,0)
        lm_uv_node.location = grid(-2,0)
    # Link them
    links.new(out_node.inputs['Surface'], bsdf_node.outputs['BSDF'])
    if is_lightmapped and is_textured:
        links.new(bsdf_node.inputs['Base Color'], mix_node.outputs['Color'])
        links.new(mix_node.inputs['Color1'], tx_img_node.outputs['Color'])
        links.new(mix_node.inputs['Color2'], lm_img_node.outputs['Color'])
        links.new(tx_img_node.inputs['Vector'], tx_uv_node.outputs['UV'])
        links.new(lm_img_node.inputs['Vector'], lm_uv_node.outputs['UV'])
    elif is_textured:
        links.new(bsdf_node.inputs['Base Color'], tx_img_node.outputs['Color'])
        links.new(tx_img_node.inputs['Vector'], tx_uv_node.outputs['UV'])
    elif is_lightmapped:
        links.new(bsdf_node.inputs['Base Color'], lm_img_node.outputs['Color'])
        links.new(lm_img_node.inputs['Vector'], lm_uv_node.outputs['UV'])
    return mat

def poly_calculate_uvs(cell, pi, texture_size):
    poly = cell.p_polys[pi]
    render = cell.p_render_polys[pi]
    vertices = cell.poly_vertices[pi]
    info = cell.p_light_list[pi]
    tx_u_scale = 64.0/texture_size[0]
    tx_v_scale = 64.0/texture_size[1]
    lm_u_scale = 4.0/info.width
    lm_v_scale = 4.0/info.height
    p_uvec = Vector(render.tex_u)
    p_vvec = Vector(render.tex_v)
    u2 = p_uvec.dot(p_uvec)
    v2 = p_vvec.dot(p_vvec)
    uv = p_uvec.dot(p_vvec)
    anchor = Vector(vertices[render.texture_anchor])
    tx_u_base = float(render.u_base)*tx_u_scale/(16.0*256.0) # u translation
    tx_v_base = float(render.v_base)*tx_v_scale/(16.0*256.0) # v translation
    lm_u_base = lm_u_scale*(float(render.u_base)/(16.0*256.0)+(0.5-float(info.u_base))/4.0) # u translation
    lm_v_base = lm_v_scale*(float(render.v_base)/(16.0*256.0)+(0.5-float(info.v_base))/4.0) # v translation
    tx_uv_list = []
    lm_uv_list = []
    if uv == 0.0:
        tx_uvec = p_uvec*tx_u_scale/u2;
        tx_vvec = p_vvec*tx_v_scale/v2;
        lm_uvec = p_uvec*lm_u_scale/u2;
        lm_vvec = p_vvec*lm_v_scale/v2;
        for i in reversed(range(poly.num_vertices)):
            wvec = Vector(vertices[i])
            delta = wvec-anchor
            tx_u = delta.dot(tx_uvec)+tx_u_base
            tx_v = delta.dot(tx_vvec)+tx_v_base
            lm_u = delta.dot(lm_uvec)+lm_u_base
            lm_v = delta.dot(lm_vvec)+lm_v_base
            # Blender's V coordinate is bottom-up
            tx_v = 1.0-tx_v
            lm_v = 1.0-lm_v
            tx_uv_list.append((tx_u,tx_v))
            lm_uv_list.append((lm_u,lm_v))
    else:
        denom = 1.0/(u2*v2-(uv*uv));
        tx_u2 = u2*tx_v_scale*denom
        tx_v2 = v2*tx_u_scale*denom
        tx_uvu = tx_u_scale*denom*uv
        tx_uvv = tx_v_scale*denom*uv
        lm_u2 = u2*lm_v_scale*denom
        lm_v2 = v2*lm_u_scale*denom
        lm_uvu = lm_u_scale*denom*uv
        lm_uvv = lm_v_scale*denom*uv
        for i in reversed(range(poly.num_vertices)):
            wvec = Vector(vertices[i])
            delta = wvec-anchor
            du = delta.dot(p_uvec)
            dv = delta.dot(p_vvec)
            tx_u = tx_u_base+tx_v2*du-tx_uvu*dv
            tx_v = tx_v_base+tx_u2*dv-tx_uvv*du
            lm_u = lm_u_base+lm_v2*du-lm_uvu*dv
            lm_v = lm_v_base+lm_u2*dv-lm_uvv*du
            tx_v = 1.0-tx_v
            lm_v = 1.0-lm_v
            tx_uv_list.append((tx_u,tx_v))
            lm_uv_list.append((lm_u,lm_v))
    return (tx_uv_list, lm_uv_list)

def do_worldrep(chunk, textures, context, options=None):
    default_options = {
        'dump': False,
        'dump_file': sys.stdout,
        'cell_limit': 0,
        }
    options = {**default_options, **(options or {})}
    DUMP = options['dump']
    dumpf = options['dump_file']

    if (chunk.header.version.major, chunk.header.version.minor) \
    not in [(0, 23), (0, 24)]:
        raise ValueError("Only version 0.23 and 0.24 WR chunk is supported")
    f = StructuredReader(buffer=chunk.data)
    header = f.read(LGWRHeader)

    if DUMP:
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

    cells = np.zeros(header.cell_count, dtype=object)
    for cell_index in range(header.cell_count):
        print(f"Reading cell {cell_index} at offset 0x{f.offset:08x}")
        cells[cell_index] = LGWRCell.read(f, cell_index)

    if DUMP:
        limit = options['cell_limit']
        cells_to_dump = cells[:limit] if limit else cells
        for cell_index, cell in enumerate(cells_to_dump):
            print(f"  Cell {cell_index}:", file=dumpf)
            print(f"    num_vertices: {cell.header.num_vertices}", file=dumpf)
            print(f"    num_polys: {cell.header.num_polys}", file=dumpf)
            print(f"    num_render_polys: {cell.header.num_render_polys}", file=dumpf)
            print(f"    num_portal_polys: {cell.header.num_portal_polys}", file=dumpf)
            print(f"    num_planes: {cell.header.num_planes}", file=dumpf)
            print(f"    medium: {cell.header.medium}", file=dumpf)
            print(f"    flags_: {cell.header.flags_}", file=dumpf)
            print(f"    portal_vertex_list: {cell.header.portal_vertex_list}", file=dumpf)
            print(f"    num_vlist: {cell.header.num_vlist}", file=dumpf)
            print(f"    num_anim_lights: {cell.header.num_anim_lights}", file=dumpf)
            print(f"    motion_index: {cell.header.motion_index}", file=dumpf)
            print(f"    sphere_center: {cell.header.sphere_center}", file=dumpf)
            print(f"    sphere_radius: {cell.header.sphere_radius}", file=dumpf)
            print(f"    p_vertices: {cell.p_vertices.size}", file=dumpf)
            for i, v in enumerate(cell.p_vertices):
                print(f"      {i}: {v[0]:06f},{v[1]:06f},{v[2]:06f}", file=dumpf)
            print(f"    p_polys: {cell.p_polys.size}", file=dumpf)
            print(f"    p_render_polys: {cell.p_render_polys.size}", file=dumpf)
            for i, rpoly in enumerate(cell.p_render_polys):
                print(f"      render_poly {i}:", file=dumpf)
                print(f"        tex_u: {rpoly.tex_u[0]:06f},{rpoly.tex_u[1]:06f},{rpoly.tex_u[2]:06f}", file=dumpf)
                print(f"        tex_v: {rpoly.tex_v[0]:06f},{rpoly.tex_v[1]:06f},{rpoly.tex_v[2]:06f}", file=dumpf)
                print(f"        u_base: {rpoly.u_base} (0x{rpoly.u_base:04x})", file=dumpf)
                print(f"        v_base: {rpoly.v_base} (0x{rpoly.v_base:04x})", file=dumpf)
                print(f"        texture_id: {rpoly.texture_id}", file=dumpf)
                print(f"        texture_anchor: {rpoly.texture_anchor}", file=dumpf)
                # Skip printing  cached_surface, texture_mag, center.
            print(f"    index_count: {cell.index_count}", file=dumpf)
            print(f"    p_index_list: {cell.p_index_list.size}", file=dumpf)
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
            print(f"    p_plane_list: {cell.p_plane_list.size}", file=dumpf)
            print(f"    p_anim_lights: {cell.p_anim_lights.size}", file=dumpf)
            print(f"    p_light_list: {cell.p_light_list.size}", file=dumpf)
            for i, info in enumerate(cell.p_light_list):
                print(f"      lightmapinfo {i}:", file=dumpf)
                print(f"        u_base: {info.u_base} (0x{info.u_base:04x})", file=dumpf)
                print(f"        v_base: {info.v_base} (0x{info.v_base:04x})", file=dumpf)
                print(f"        byte_width: {info.byte_width}", file=dumpf)
                print(f"        height: {info.height}", file=dumpf)
                print(f"        width: {info.width}", file=dumpf)
                print(f"        anim_light_bitmask: 0x{info.anim_light_bitmask:08x}", file=dumpf)
            print(f"    num_light_indices: {cell.num_light_indices}", file=dumpf)
            print(f"    p_light_indices: {cell.p_light_indices.size}", file=dumpf)

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

    MAX_CELLS = 32678       # Imposed by Dromed
    MAX_VERTICES = 256*1024 # Imposed by Dromed
    MAX_FACES = 256*1024    # Rough guess
    MAX_FACE_INDICES = 32   # Imposed by Dromed
    MAX_INDICES = MAX_FACE_INDICES*MAX_FACES

    # Allocate a bunch of memory
    verts = np.zeros((MAX_VERTICES,3), dtype=float32)
    idxs = np.zeros(MAX_INDICES, dtype=int32)
    texture_uvs = np.zeros((MAX_INDICES,2), dtype=float32)
    lightmap_uvs = np.zeros((MAX_INDICES,2), dtype=float32)
    lightmap_handles = np.zeros(MAX_INDICES, dtype=int32)
    loop_starts = np.zeros(MAX_FACES, dtype=int32)
    loop_counts = np.zeros(MAX_FACES, dtype=int32)
    material_idxs = np.zeros(MAX_FACES, dtype=int32)
    material_textures = {} # texture_id: material_idx

    def lookup_texture_id(texture_id):
        # Assume Jorge and SKY_HACK (and any invalid texture ids) are 64x64:
        in_range = (0<=texture_id<len(textures))
        special = texture_id in (0,249)
        ok = (in_range and not special)
        texture_image = textures[texture_id] if ok else None
        texture_size = texture_image.size if ok else (64, 64)
        return (texture_image, texture_size)

    atlas_builder = AtlasBuilder()
    vert_ptr = idx_ptr = face_ptr = 0
    limit = options['cell_limit']
    cells_to_build = cells[:limit] if limit else cells
    for cell_index, cell in enumerate(cells_to_build):
        # Add the vertices from this cell.
        cell_vert_start = vert_ptr
        end = cell_vert_start+len(cell.p_vertices)
        verts[cell_vert_start:end] = cell.p_vertices
        vert_ptr = end
        # Add each poly.
        for pi, poly in enumerate(cell.p_polys):
            is_render = (pi<cell.header.num_render_polys)
            is_portal = (pi>=(cell.header.num_polys-cell.header.num_portal_polys))
            if not is_render: continue  # Skip air portals
            if is_portal: continue      # Skip water surfaces
            # Add the indices from this poly:
            # Reverse the indices, so faces point the right way in Blender.
            # Adjust indices to point into our vertex array.
            idx_start = idx_ptr
            idx_end = idx_start+poly.num_vertices
            poly_idxs = np.array(cell.poly_indices[pi][::-1], dtype=uint32)+cell_vert_start
            idxs[idx_start:idx_end] = poly_idxs
            idx_ptr = idx_end
            # Add the loop start/count for this poly.
            loop_starts[face_ptr] = idx_start
            loop_counts[face_ptr] = poly.num_vertices
            # Look up the texture.
            texture_id = cell.p_render_polys[pi].texture_id
            texture_image, texture_size = lookup_texture_id(texture_id)
            # Set the material index.
            mat_idx = material_textures.get(texture_id)
            if mat_idx is None:
                mat_idx = len(material_textures)
                material_textures[texture_id] = mat_idx
            material_idxs[face_ptr] = mat_idx
            # Calculate uvs.
            poly_tx_uvs, poly_lm_uvs = poly_calculate_uvs(cell, pi, texture_size)
            texture_uvs[idx_start:idx_end] = poly_tx_uvs
            lightmap_uvs[idx_start:idx_end] = poly_lm_uvs
            # Add the primary lightmap for this poly to the atlas.
            info = cell.p_light_list[pi]
            lm_handle = atlas_builder.add(info.width, info.height, cell.lightmaps[pi][0])
            lightmap_handles[idx_start:idx_end] = lm_handle
            # Time for the next polygon!
            face_ptr += 1
    # Count totals now that all cells have been processed:
    vert_total = vert_ptr
    idx_total = idx_ptr
    face_total = face_ptr
    # Build the lightmap
    atlas_builder.finish()
    lightmap_image = atlas_builder.image
    # Create the mesh geometry.
    name = "miss1" # TODO: get the mission name passed in!
    mesh = bpy.data.meshes.new(name=f"{name} mesh")
    mesh.vertices.add(vert_total)
    mesh.loops.add(idx_total)
    mesh.polygons.add(face_total)
    try:
        mesh.vertices.foreach_set("co", verts[:vert_total].reshape(-1))
        mesh.polygons.foreach_set("loop_total", loop_counts[:face_total])
        mesh.polygons.foreach_set("loop_start", loop_starts[:face_total])
        mesh.polygons.foreach_set("vertices", idxs[:idx_total])
        mesh.update(calc_edges=True, calc_edges_loose=False)
        modified = mesh.validate(verbose=True)
        assert not modified, f"Mesh {name} pydata was invalid."
    except:
        # The polygon was invalid for some reason! Dump out the data we are
        # building it from to help understand why.
        sys.stdout.flush()
        sys.stderr.flush()
        np.set_printoptions(threshold=100)
        print("vert_total: ", vert_total, file=sys.stderr)
        print("idx_total: ", idx_total, file=sys.stderr)
        print("face_total: ", face_total, file=sys.stderr)
        print("loop_total: ", loop_total, file=sys.stderr)
        print("verts: ", verts[:vert_total], file=sys.stderr)
        print("idxs: ", idxs[:idx_total], file=sys.stderr)
        print("loop_starts: ", loop_starts[:face_total], file=sys.stderr)
        print("loop_counts: ", loop_counts[:face_total], file=sys.stderr)
        raise
    # Set the remaining mesh attributes.
    mesh.polygons.foreach_set("material_index", material_idxs[:face_total])
    texture_uv_layer = (mesh.uv_layers.get('UVMap')
        or mesh.uv_layers.new(name='UVMap'))
    lightmap_uv_layer = (mesh.uv_layers.get('UVLightmap')
        or mesh.uv_layers.new(name='UVLightmap'))
    texture_uv_layer.data.foreach_set('uv', texture_uvs[:idx_total].reshape(-1))

    # Transform all lightmap uvs
    # TODO: i am sure can we do this in a better way with numpy!
    lightmap_atlas_uvs = np.zeros((MAX_INDICES,2), dtype=float32)
    for i in range(idx_total):
        u, v = lightmap_uvs[i]
        handle = lightmap_handles[i]
        # Wrap to 0,1 range.
        u, _ = math.modf(u)
        v, _ = math.modf(v)
        if u<0: u = 1.0-u
        if v<0: v = 1.0-v
        # Transform into atlassed location.
        scale, translate = atlas_builder.get_uv_transform(handle)
        u = scale[0]*u+translate[0]
        v = scale[1]*v+translate[1]
        lightmap_atlas_uvs[i] = (u, v)
    lightmap_uv_layer.data.foreach_set('uv', lightmap_atlas_uvs[:idx_total].reshape(-1))
    # Create the materials.
    mat_jorge = create_texture_material('JORGE', None, None)
    mat_sky = create_texture_material('SKY_HACK', None, None)
    mat_missing = create_texture_material('MISSING', None, None)
    texture_ids_needed = [tid
        for (tid, _) in sorted(material_textures.items(), key=(lambda item:item[1]))]
    for texture_id in texture_ids_needed:
        if texture_id==0:
            mat = mat_jorge
        elif texture_id==249:
            mat = mat_sky
        else:
            im, _ = lookup_texture_id(texture_id)
            if im is None:
                mat = mat_missing
            else:
                mat = create_texture_material(im.name, im, lightmap_image)
        mesh.materials.append(mat)
    # Create and link the object.
    o = create_object(name, mesh, (0,0,0), context=context, link=True)
    return o

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

    def finish(self):
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
