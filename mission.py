import bpy
import math
import mathutils
import numpy as np
import os
import sys
import time

from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from bpy_extras.image_utils import load_image
from bpy_extras.io_utils import ImportHelper
from bpy_extras.wm_utils.progress_report import ProgressReport
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
    padded_width: int16
    height: uint8
    width: uint8
    data_ptr: uint32            # Always zero on disk
    dynamic_light_ptr: uint32   # Always zero on disk
    anim_light_bitmask: uint32

# WR lightmap data is uint8; WRRGB is uint16 (xB5G5R5)
LGWRLightmap8Bit = uint8
LGWRRGBLightmap16Bit = uint16

class LGWRCell:
    @classmethod
    def read(cls, reader, cell_index, lightmap_format):
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
        entry_type = np.dtype(lightmap_format)
        for info in cell.p_light_list:
            width = info.width
            height = info.height
            count = 1+bit_count(info.anim_light_bitmask) # 1 base lightmap, plus 1 per animlight
            assert info.padded_width==info.width, f"lightmap padded_width {padded_width} is not the same as width {width}!"
            raw = f.read(entry_type, count=count*height*width)
            # Expand the lightmap into rgba floats
            if lightmap_format is LGWRLightmap8Bit:
                raw.shape = (count, height, width, 1)
                raw = np.flip(raw, axis=1)
                wf = np.array(raw, dtype=float)/255.0
                rgbf = np.repeat(wf, repeats=3, axis=3)
                rgbaf = np.insert(rgbf, 3, 1.0, axis=3)
            elif lightmap_format is LGWRRGBLightmap16Bit:
                raw.shape = -1
                r = np.bitwise_and(raw, 0x1f)
                g = np.bitwise_and(np.right_shift(raw, 5), 0x1f)
                b = np.bitwise_and(np.right_shift(raw, 10), 0x1f)
                rgbaf = np.ones((len(raw),4), dtype=np.float32)
                rgbaf[:,0] = r
                rgbaf[:,1] = g
                rgbaf[:,2] = b
                rgbaf.shape = (-1, 4)
                div = np.array([32.0,32.0,32.0,1.0], dtype=np.float32)
                rgbaf = rgbaf/div
                rgbaf.shape = (count, height, width, 4)
                rgbaf = np.flip(rgbaf, axis=1)
            else:
                raise ValueError("Unsupported lightmap_format")
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

    @property
    def name(self):
        return ascii(self.header.name)

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
        if chunk.name != name:
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

def import_mission(context, filepath):
    dirname, basename = os.path.split(filepath)
    miss_name = os.path.splitext(basename)[0]
    # TODO: make dumping an option?
    dump_filename = os.path.join(dirname, miss_name+'.dump')
    dumpf = open(dump_filename, 'w')
    #dumpf = sys.stdout

    with ProgressReport(context.window_manager) as progress:
        progress.enter_substeps(1, f"Importing .MIS {filepath!r}...")
        # Parse the .bin file
        mis = LGDBFile(filepath)
        # TODO: make dumping an option!
        print(f"table_offset: {mis.header.table_offset}", file=dumpf)
        print(f"version: {mis.header.version.major}.{mis.header.version.minor}", file=dumpf)
        print(f"deadbeef: {mis.header.deadbeef!r}", file=dumpf)
        print("Chunks:", file=dumpf)
        for i, name in enumerate(mis):
            print(f"  {i}: {name}", file=dumpf)

        start = time.process_time()
        txlist = mis['TXLIST']
        options={
            'dump': False,
            'dump_file': dumpf,
            }
        textures = do_txlist(txlist, context, progress=progress, **options)
        textures_time = time.process_time()-start

        start = time.process_time()
        if 'WR' in mis:
            worldrep = mis['WR']
        elif 'WRRGB' in mis:
            worldrep = mis['WRRGB']
        else:
            # TODO: what about newdark 32-bit lighting?
            raise ValueError(f"No WR or WRRGB worldrep chunk in {basename}.")
        options={
            'dump': False,
            'dump_file': dumpf,
            'cell_limit': 0,
            }
        obj = do_worldrep(worldrep, textures, context, name=miss_name, progress=progress, **options)
        worldrep_time = time.process_time()-start
        progress.leave_substeps(f"Finished importing: {filepath!r}")

    print(f"Load textures: {textures_time:0.1f}s")
    print(f"Load worldrep: {worldrep_time:0.1f}s")
    return obj

def do_txlist(chunk, context, progress=None,
    dump=False, dump_file=None):
    dumpf = dump_file or sys.stdout
    assert progress is not None
    if (chunk.header.version.major, chunk.header.version.minor) \
    not in [(1, 0)]:
        raise ValueError("Only version 1.0 TXLIST chunk is supported")
    f = StructuredReader(buffer=chunk.data)
    header = f.read(LGTXLISTHeader)
    p_fams = f.read(LGTXLISTFam, count=header.fam_count)
    p_texs = f.read(LGTXLISTTex, count=header.tex_count)
    if dump:
        print(f"TXLIST:", file=dumpf)
        print(f"  length: {header.length}", file=dumpf)
        print(f"  fam_count: {header.fam_count}", file=dumpf)
        for i, fam in enumerate(p_fams):
            name = ascii(fam.name)
            print(f"    {i}: {name}", file=dumpf)
        print(f"  tex_count: {header.tex_count}", file=dumpf)
        for i, tex in enumerate(p_texs):
            name = ascii(tex.name)
            print(f"    {i}: fam {tex.fam_id}, {name}, flags_ 0x{tex.flags_:02x}, pad 0x{tex.pad:04x}", file=dumpf)

    # Load all the textures into Blender images (except poor Jorge, who always
    # gets left out):
    tex_search_paths = ['e:/dev/thief/TMA1.27/__unpacked/res/fam']
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
        if dump:
            print(f"Searching for fam/{fam_name}/{tex_name}...", file=dumpf)
        candidates = [] # (sort_key, full_path) tuples
        for path in tex_search_paths:
            fam_path = os.path.join(path, fam_name)
            for lang in ['', 'english', 'french', 'german', 'russian', 'italian']:
                if lang:
                    lang_path = os.path.join(fam_path, lang)
                    if os.path.isdir(lang_path):
                        fam_path = lang_path
                    else:
                        continue
                if dump:
                    print(f"  in path: {fam_path}", file=dumpf)
                for entry in os.scandir(fam_path):
                    if not entry.is_file(): continue
                    name, ext = os.path.splitext(entry.name.lower())
                    if name != tex_name: continue
                    sort_key = ext_sort_order.get(ext, None)
                    if sort_key is None: continue
                    if dump:
                        print(f"    Candidate: {entry.name}", file=dumpf)
                    candidates.append((sort_key, entry.path))
        if not candidates:
            raise ValueError(f"Cannot find texture {fam_name}/{tex_name}")
        candidates.sort()
        filename = candidates[0][1]
        # Load the winning file
        if dump:
            print(f"Loading: {filename}...", file=dumpf)
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

    tex_count = len(p_texs)
    progress.enter_substeps(tex_count, "Loading textures...")
    textures = []
    for i, tex in enumerate(p_texs):
        if tex.fam_id==0:
            textures.append(None) # TODO: Jorge, is that you?
            progress.step()
        else:
            fam = p_fams[tex.fam_id-1]
            fam_name = ascii(fam.name)
            tex_name = ascii(tex.name)
            progress.step(f"Loading fam/{fam_name}/{tex_name}")
            image = load_tex(fam_name, tex_name)
            textures.append(image)
    progress.leave_substeps(f"{tex_count} textures loaded.")
    return textures

def create_settings_node_group(obj, base_name):
    name = f"{base_name}.Settings"
    tree = bpy.data.node_groups.new(name=name, type='ShaderNodeTree')
    nodes = tree.nodes
    links = tree.links
    def grid(x,y):
        return (x*400.0,y*200.0)
    # Group output
    n = nodes.new(type='NodeGroupOutput')
    n.location = grid(0,0)
    output_node = n
    # Ambient brightness
    n = nodes.new(type='ShaderNodeValue')
    n.location = grid(-1,0)
    n.name = 'AmbientBrightness'
    n.label = "Ambient Brightness"
    n.outputs['Value'].default_value = 0.0
    # Create a property on the object to drive this from
    obj["AmbientBrightness"] = 0.2
    d = n.outputs['Value'].driver_add('default_value').driver
    v = d.variables.new()
    v.name = 'var'
    v.type = 'SINGLE_PROP'
    t = v.targets[0]
    t.id = obj
    t.data_path = 'tt_mission.ambient_brightness'
    d.expression = "var"
    ambient_brightness_node = n
    # Connect everything
    output = tree.outputs.new('NodeSocketFloat', 'AmbientBrightness')
    links.new(output_node.inputs['AmbientBrightness'], ambient_brightness_node.outputs['Value'])
    return tree

def create_texture_material(name, texture_image, lightmap_image, settings_group):
    is_textured = (texture_image is not None)
    is_lightmapped = (lightmap_image is not None)
    #
    # If lightmapped and textured:
    #
    #       UVMap('UVMap')              UVMap('UVLightmap')
    #             |                               |
    # ImageTexture(texture_image)     ImageTexture(lightmap_image)
    #             |                               |
    #             +----------------+              |
    #                              |              |
    #                              |         MixRGB('Mix') <-- NodeGroup('Settings').Ambient
    #                              |              |
    #                              +--------------+
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
        tx_uv_node = nodes.new(type='ShaderNodeUVMap'); tx_uv_node.select = False
        tx_img_node = nodes.new(type='ShaderNodeTexImage'); tx_img_node.select = False
    if is_lightmapped:
        lm_uv_node = nodes.new(type='ShaderNodeUVMap'); lm_uv_node.select = False
        lm_img_node = nodes.new(type='ShaderNodeTexImage'); lm_img_node.select = False
        lm_mix_node = nodes.new(type='ShaderNodeMixRGB'); lm_mix_node.select = False
        settings_node = nodes.new(type='ShaderNodeGroup'); settings_node.select = False
    if is_textured and is_lightmapped:
        mix_node = nodes.new(type='ShaderNodeMixRGB'); mix_node.select = False
    # Configure them
    bsdf_node.inputs['Base Color'].default_value = (1.0,0.0,1.0,1.0)
    bsdf_node.inputs['Metallic'].default_value = 0.0
    bsdf_node.inputs['Specular'].default_value = 0.0
    bsdf_node.inputs['Roughness'].default_value = 1.0
    if is_textured:
        tx_img_node.name = 'TerrainTexture'
        tx_img_node.image = texture_image
        tx_uv_node.name = 'TerrainUV'
        tx_uv_node.uv_map = 'UVMap'
    if is_lightmapped:
        lm_img_node.name = 'LightmapTexture'
        lm_img_node.image = lightmap_image
        lm_uv_node.name = 'LightmapUV'
        lm_uv_node.uv_map = 'UVLightmap'
        lm_mix_node.name = 'MixAmbient'
        lm_mix_node.blend_type = 'MIX'
        lm_mix_node.inputs['Fac'].default_value = 0.0
        lm_mix_node.inputs['Color1'].default_value = (1.0,1.0,1.0,1.0)
        lm_mix_node.inputs['Color2'].default_value = (1.0,1.0,1.0,1.0)
        settings_node.node_tree = settings_group
    if is_textured and is_lightmapped:
        mix_node.blend_type = 'MULTIPLY'
        mix_node.inputs['Fac'].default_value = 1.0
        mix_node.inputs['Color1'].default_value = (1.0,1.0,1.0,1.0)
        mix_node.inputs['Color2'].default_value = (1.0,1.0,1.0,1.0)
        mix_node.use_clamp = True
    # Place them
    def grid(x,y):
        return (x*400.0,y*200.0)
    out_node.location = grid(1,0)
    bsdf_node.location = grid(0,0)
    if is_lightmapped and is_textured:
        mix_node.location = grid(-1,0)
        tx_img_node.location = grid(-3,1)
        tx_uv_node.location = grid(-4,1)
        lm_img_node.location = grid(-3,-1)
        lm_uv_node.location = grid(-4,-1)
        lm_mix_node.location = grid(-2,-1)
        settings_node.location = grid(-3,-3)
    elif is_textured:
        tx_img_node.location = grid(-1,0)
        tx_uv_node.location = grid(-2,0)
    elif is_lightmapped:
        lm_img_node.location = grid(-2,0)
        lm_uv_node.location = grid(-3,0)
        lm_mix_node.location = grid(-1,-0)
    # Link them
    links.new(out_node.inputs['Surface'], bsdf_node.outputs['BSDF'])
    if is_lightmapped and is_textured:
        links.new(bsdf_node.inputs['Base Color'], mix_node.outputs['Color'])
        links.new(mix_node.inputs['Color1'], tx_img_node.outputs['Color'])
        links.new(mix_node.inputs['Color2'], lm_mix_node.outputs['Color'])
        links.new(lm_mix_node.inputs['Fac'], settings_node.outputs['AmbientBrightness'])
        links.new(lm_mix_node.inputs['Color1'], lm_img_node.outputs['Color'])
        links.new(tx_img_node.inputs['Vector'], tx_uv_node.outputs['UV'])
        links.new(lm_img_node.inputs['Vector'], lm_uv_node.outputs['UV'])
    elif is_textured:
        links.new(bsdf_node.inputs['Base Color'], tx_img_node.outputs['Color'])
        links.new(tx_img_node.inputs['Vector'], tx_uv_node.outputs['UV'])
    elif is_lightmapped:
        links.new(bsdf_node.inputs['Base Color'], lm_mix_node.outputs['Color'])
        links.new(lm_mix_node.inputs['Color1'], lm_img_node.outputs['Color'])
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

def do_worldrep(chunk, textures, context, name="mission", progress=None,
    dump=False, dump_file=None, cell_limit=0):
    dumpf = dump_file or sys.stdout
    assert (progress is not None)
    progress.enter_substeps(5, "Loading worldrep...")

    if chunk.name=='WR':
        if (chunk.header.version.major, chunk.header.version.minor) \
        not in [(0, 23)]:
            raise ValueError("Only version 0.23 WR chunk is supported")
        lightmap_format = LGWRLightmap8Bit
    elif chunk.name=='WRRGB':
        if (chunk.header.version.major, chunk.header.version.minor) \
        not in [(0, 24)]:
            raise ValueError("Only version 0.24 WR chunk is supported")
        lightmap_format = LGWRRGBLightmap16Bit
    else:
        raise ValueError(f"Unsupported worldrep chunk {chunk.name}")

    f = StructuredReader(buffer=chunk.data)
    header = f.read(LGWRHeader)
    if dump:
        print(f"{chunk.name} chunk:", file=dumpf)
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

    progress.step("Loading cells...")
    cell_progress_step_size = 100
    cell_progress_step_count = (header.cell_count//cell_progress_step_size)+1
    progress.enter_substeps(cell_progress_step_count)
    cells = np.zeros(header.cell_count, dtype=object)
    for cell_index in range(header.cell_count):
        try:
            if cell_index%cell_progress_step_size==0: progress.step()
            cells[cell_index] = LGWRCell.read(f, cell_index, lightmap_format)
        except:
            print(f"Reading cell {cell_index} at offset 0x{f.offset:08x}...", file=sys.stderr)
            raise

    if dump:
        cells_to_dump = cells[:cell_limit] if cell_limit else cells
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
                print(f"        padded_width: {info.padded_width}", file=dumpf)
                print(f"        height: {info.height}", file=dumpf)
                print(f"        width: {info.width}", file=dumpf)
                print(f"        anim_light_bitmask: 0x{info.anim_light_bitmask:08x}", file=dumpf)
            print(f"    num_light_indices: {cell.num_light_indices}", file=dumpf)
            print(f"    p_light_indices: {cell.p_light_indices.size}", file=dumpf)
    progress.leave_substeps()

    progress.step("Building geometry...")
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
    cells_to_build = cells[:cell_limit] if cell_limit else cells
    cell_progress_step_size = 100
    cell_progress_step_count = (len(cells_to_build)//cell_progress_step_size)+1
    progress.enter_substeps(cell_progress_step_count)
    for cell_index, cell in enumerate(cells_to_build):
        if cell_index%cell_progress_step_size==0: progress.step()
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
    progress.leave_substeps()

    # Build the lightmap
    progress.step("Building lightmap atlas...")
    atlas_builder.finish()
    lightmap_image = atlas_builder.image

    # Create the mesh geometry.
    progress.step("Creating mesh...")
    mesh = bpy.data.meshes.new(name=f"{name} mesh")
    obj = create_object(name, mesh, (0,0,0), context=context, link=True)
    obj.tt_mission.is_mission = True
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
        assert not modified, f"Mesh {mesh.name} pydata was invalid."
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
    progress.step("Creating materials and shaders...")
    settings_group = create_settings_node_group(obj, name)
    mat_jorge = create_texture_material('JORGE', None, None, settings_group)
    mat_sky = create_texture_material('SKY_HACK', None, None, settings_group)
    mat_missing = create_texture_material('MISSING', None, None, settings_group)
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
                mat = create_texture_material(im.name, im, lightmap_image, settings_group)
        mesh.materials.append(mat)

    progress.leave_substeps("Done")
    return obj

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
        atlas_w, atlas_h = (256, 256)
        quadrant_w, quadrant_h = (atlas_w, atlas_h)
        quadrant_index = 0
        # x, y are abs coords of the image placement cursor
        x = y = 0
        # quadrant_x, quadrant_y are abs coords of the current quadrant
        quadrant_x = quadrant_y = 0
        row_height = 0
        placements = [None]*len(self.images)
        USE_DEBUG_COLORS = False
        DEBUG_QUADRANTS = False
        debug_colors = [(r/255.0,g/255.0,b/255.0,1.0) for (r,g,b) in
            [(237,20,91), (246,142,86), (60,184,120), (166,124,82),
            (255,245,104), (109,207,246), (168,100,168), (194,194,194)] ]
        #
        # To fit the images into the atlas, we place it at the cursor, if it
        # will fit. If not, we reset the cursor to the left edge, and move it
        # up by the row height; then reset the row height to zero. The row
        # height is the max height of the images placed on that row.
        #
        # The placements are the (x,y,w,h) tuples of the anchor where the
        # corresponding image was placed.
        #
        # 1. Start with a small atlas
        #     +-----+
        #     |     |
        #     |     |
        #     +-----+
        #
        # 2. Expand it when needed:
        #
        #     +-----------+
        #     ^           |
        #     |           |
        #     +.....+     |
        #     |     .     |
        #     |     .     |
        #     +-----+---->+
        #
        # 2. Expand again if needed:
        #
        #     +-----------------------+
        #     |                       |
        #     |                       |
        #     |                       |
        #     |                       |
        #     |                       |
        #     +-----------+           |
        #     ^           |           |
        #     |           |           |
        #     + . . .     |           |
        #     |     .     |           |
        #     |     .     |           |
        #     +-----+-----+---------->+

        atlas_data = np.zeros((atlas_w,atlas_h,4), dtype=float)

        for image_index, (w, h, handle, image, rotated) in enumerate(self.images):
            # Find a place for the image.
            if w>quadrant_w or h>quadrant_h:
                raise ValueError(f"No space to fit image {handle} of {w}x{h}")
            while True: # TODO: break if the atlas is now too big
                # quadrant's right edge:
                if (x+w)>(quadrant_x+quadrant_w):
                    # Wrap the cursor to the next row up:
                    x = quadrant_x
                    y += row_height
                    row_height = 0
                if (y+h)>(quadrant_y+quadrant_h):
                    # Wrap the cursor to the next quadrant across:
                    quadrant_x += quadrant_w
                    quadrant_index += 1
                    x = quadrant_x
                    y = quadrant_y
                if quadrant_x>=atlas_w:
                    # Wrap the cursor to the next quadrant above left:
                    quadrant_x = 0
                    quadrant_y += quadrant_h
                    x = quadrant_x
                    y = quadrant_y
                if quadrant_y>=atlas_h:
                    # Expand the atlas (and quadrant) size:
                    quadrant_w = atlas_w
                    quadrant_h = atlas_h
                    atlas_w *= 2
                    atlas_h *= 2
                    quadrant_x = quadrant_w
                    quadrant_y = 0
                    x = quadrant_x
                    y = quadrant_y
                    if atlas_w>=4096 or atlas_h>=4096:
                        raise ValueError(f"No space to fit image {handle} of {w}x{h} in 4k atlas")
                    # Resize the array with three new blank quadrants:
                    blank = np.zeros((quadrant_w,quadrant_h*4), dtype=float)
                    old_data = atlas_data.reshape((quadrant_w,quadrant_h*4))
                    new_data = np.block([[ old_data, blank ],
                                         [ blank,    blank ]])
                    atlas_data = new_data.reshape((atlas_w,atlas_h,4))
                # We have a valid cursor position
                break
            # Place and blit the image.
            placements[handle] = (x,y,w,h)
            if USE_DEBUG_COLORS:
                if DEBUG_QUADRANTS:
                    source = debug_colors[quadrant_index%len(debug_colors)]
                else:
                    source = debug_colors[i%len(debug_colors)]
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

#---------------------------------------------------------------------------#
# Post-import utilities

def get_texture_interpolation(obj):
    # The first material with a texture node determines what we consider the
    # interpolation setting.
    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        node = mat.node_tree.nodes.get('TerrainTexture')
        if node is None: continue
        return node.interpolation
    return 'Linear'

def set_texture_interpolation(obj, interpolation):
    # The first material with a texture node determines what we consider the
    # interpolation setting.
    assert interpolation in ('Closest', 'Linear')
    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        node = mat.node_tree.nodes.get('TerrainTexture')
        if node is None: continue
        node.interpolation = interpolation

def is_textures_enabled(obj):
    # The first material with a texture node determines if we consider the
    # textures enabled.
    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        node = mat.node_tree.nodes.get('TerrainTexture')
        if node is None: continue
        return (not node.mute)
    return False

def enable_textures(obj, enable=True):
    # Mute or unmute texture nodes in all the mesh's materials (if present).
    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        node = mat.node_tree.nodes.get('TerrainTexture')
        if node is None: continue
        node.mute = (not enable)

def is_lightmaps_enabled(obj):
    # The first material with a lightmap node determines if we consider the
    # lightmaps enabled.
    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        node = mat.node_tree.nodes.get('LightmapTexture')
        if node is None: continue
        return (not node.mute)
    return False

def enable_lightmaps(obj, enable=True):
    # Mute or unmute lightmap nodes in all the mesh's materials (if present).
    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        node = mat.node_tree.nodes.get('LightmapTexture')
        if node is None: continue
        node.mute = (not enable)

def get_lightmap_interpolation(obj):
    # The first material with a lightmap node determines what we consider the
    # interpolation setting.
    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        node = mat.node_tree.nodes.get('LightmapTexture')
        if node is None: continue
        return node.interpolation
    return 'Linear'

def set_lightmap_interpolation(obj, interpolation):
    # The first material with a lightmap node determines what we consider the
    # interpolation setting.
    assert interpolation in ('Closest', 'Linear')
    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        node = mat.node_tree.nodes.get('LightmapTexture')
        if node is None: continue
        node.interpolation = interpolation

#---------------------------------------------------------------------------#
# Properties

def _get_enable_textures(self):
    if not self.is_mission: return False
    o = self.id_data
    return is_textures_enabled(o)

def _set_enable_textures(self, value):
    if not self.is_mission: return
    o = self.id_data
    enable_textures(o, value)

def _get_texture_filtering(self):
    if not self.is_mission: return False
    o = self.id_data
    return (get_texture_interpolation(o)=='Linear')

def _set_texture_filtering(self, value):
    if not self.is_mission: return
    o = self.id_data
    interpolation = 'Linear' if value else 'Closest'
    set_texture_interpolation(o, interpolation)

def _get_enable_lightmaps(self):
    if not self.is_mission: return False
    o = self.id_data
    return is_lightmaps_enabled(o)

def _set_enable_lightmaps(self, value):
    if not self.is_mission: return
    o = self.id_data
    enable_lightmaps(o, value)

def _get_lightmap_filtering(self):
    if not self.is_mission: return False
    o = self.id_data
    return (get_lightmap_interpolation(o)=='Linear')

def _set_lightmap_filtering(self, value):
    if not self.is_mission: return
    o = self.id_data
    interpolation = 'Linear' if value else 'Closest'
    set_lightmap_interpolation(o, interpolation)

class TTMissionSettings(PropertyGroup):
    is_mission: BoolProperty(name="(is_mission)", default=False)
    enable_textures: BoolProperty(name="Textures", default=True,
        get=_get_enable_textures,
        set=_set_enable_textures)
    texture_filtering: BoolProperty(name="Texture Filtering", default=True,
        get=_get_texture_filtering,
        set=_set_texture_filtering)
    enable_lightmaps: BoolProperty(name="Lightmaps", default=True,
        get=_get_enable_lightmaps,
        set=_set_enable_lightmaps)
    lightmap_filtering: BoolProperty(name="Lightmap Filtering", default=True,
        get=_get_lightmap_filtering,
        set=_set_lightmap_filtering)
    ambient_brightness: FloatProperty(name="Brightness", default=0.0,
        min=0.0, max=1.0, step=1)

#---------------------------------------------------------------------------#
# Operators

class TTImportMISOperator(Operator, ImportHelper):
    bl_idname = "object.tt_import_mis"
    bl_label = "Import .MIS"
    bl_options = {'REGISTER'}

    filename : StringProperty()

    filter_glob: StringProperty(
            default="*.mis;*.cow;*.vbr",
            options={'HIDDEN'},
            )

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        PROFILE = False
        if PROFILE:
            import cProfile
            o = None
            cProfile.runctx("o = import_mission(context, self.filepath)",
                globals(), locals(), "e:/temp/import_mission.prof")
            o.select_set(True)
        else:
            o = import_mission(context, self.filepath)
            context.view_layer.objects.active = o
            o.select_set(True)
        return {'FINISHED'}

#---------------------------------------------------------------------------#
# Menus

def menu_func_import(self, context):
    self.layout.operator(TTImportMISOperator.bl_idname, text="Thief: Mission (.mis)")

#---------------------------------------------------------------------------#
# Panels

class TOOLS_PT_thieftools_mission(Panel):
    bl_label = "Thief: Mission Settings"
    bl_idname = "TOOLS_PT_thieftools_mission"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    @classmethod
    def poll(self, context):
        # Only show this panel for missions
        o = context.active_object
        if o is None: return False
        return o.tt_mission.is_mission

    def draw(self, context):
        layout = self.layout
        o = context.active_object
        mission_settings = o.tt_mission

        row = layout.row(align=False)
        row.prop(mission_settings, 'enable_textures')
        row.prop(mission_settings, 'texture_filtering')
        row = layout.row(align=False)
        row.prop(mission_settings, 'enable_lightmaps')
        row.prop(mission_settings, 'lightmap_filtering')
        layout.prop(mission_settings, 'ambient_brightness')
