import bpy
import gpu
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
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
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
class LGWREXTHeader:
    unknown0: uint32
    unknown1: uint32
    unknown2: uint32
    lightmap_format: uint32     # 0: 16 bit; 1: 32 bit; 2: 32 bit 2x
    lightmap_scale: int32       # 0: 1x; 2: 2x; 4: 4x; -2: 0.5x; -4: 0.25x
                                # non power of two values may be stored in
                                # here; just ignore all but the highest bit,
                                # and use the sign bit to determine if it
                                # is a multiply or a divide.
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
class LGWREXTRenderPoly:
    tex_u: LGVector
    tex_v: LGVector
    u_base: float32         # Changed in WREXT
    v_base: float32         # Changed in WREXT
    texture_id: uint16      # Changed in WREXT (texture_anchor removed)
    cached_surface: uint16  ## TODO: jk has this as texture_anchor!
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
LGWRRGBLightmap32Bit = uint32

class LGWRCell:
    @classmethod
    def read(cls, reader, cell_index, wr_version, lightmap_format):
        f = reader
        cell = cls()
        cell.header = f.read(LGWRCellHeader)
        cell.p_vertices = f.read(LGVector, count=cell.header.num_vertices)
        cell.p_polys = f.read(LGWRPoly, count=cell.header.num_polys)
        if wr_version>=(0,30):
            render_poly_type = LGWREXTRenderPoly
        else:
            render_poly_type = LGWRRenderPoly
        cell.p_render_polys = f.read(render_poly_type, count=cell.header.num_render_polys)
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
            assert info.padded_width==info.width, f"lightmap padded_width {info.padded_width} is not the same as width {width}!"
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
            elif lightmap_format is LGWRRGBLightmap32Bit:
                bgra = raw.view(dtype=uint8)
                # Swap channels to convert to RGBA. There might be a more
                # efficient way to do this in numpy, but hey, this works.
                bgra.shape = (-1, 4)
                rgba = np.zeros(bgra.shape, dtype=uint8)
                rgba[:,0] = bgra[:,2]
                rgba[:,1] = bgra[:,1]
                rgba[:,2] = bgra[:,0]
                rgba[:,3] = bgra[:,3]
                # Then flip Y etc. as usual.
                rgba.shape = (count, height, width, 4)
                rgba = np.flip(rgba, axis=1)
                rgbaf = np.array(rgba, dtype=float)/255.0
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
            # Note: could actually check if it is a .mis/.cow and not just a
            #       .vbr/.gam/.sav - but no need, as none of those have
            #       a worldrep anyway.
            raise ValueError("File is not a .mis/.cow")
        if (header.version.major, header.version.minor) not in [(0, 1)]:
            raise ValueError("Only version 0.1 .mis/.cow files are supported")
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

def import_mission(context, filepath='', search_paths=(), **options):
    # Make sure nothing is selected when we begin.
    bpy.ops.object.select_all(action='DESELECT')

    dirname, basename = os.path.split(filepath)
    miss_name = os.path.splitext(basename)[0]

    dump = options.get('dump', False)
    if dump:
        dump_filename = os.path.join(dirname, miss_name+'.dump')
        dumpf = open(dump_filename, 'w')
        options['dump_file'] = dumpf

    with ProgressReport(context.window_manager) as progress:
        progress.enter_substeps(1, f"Importing .MIS {filepath!r}...")
        # Parse the .bin file
        mis = LGDBFile(filepath)
        if dump:
            print(f"table_offset: {mis.header.table_offset}", file=dumpf)
            print(f"version: {mis.header.version.major}.{mis.header.version.minor}", file=dumpf)
            print(f"deadbeef: {mis.header.deadbeef!r}", file=dumpf)
            print("Chunks:", file=dumpf)
            for i, name in enumerate(mis):
                print(f"  {i}: {name}", file=dumpf)

        # Check there's a worldrep *before* we spend time loading textures.
        if ('WR' not in mis
        and 'WRRGB' not in mis
        and 'WREXT' not in mis):
            raise ValueError(f"No WR/WRRGB/WREXT worldrep chunk in {basename}.")

        start = time.process_time()
        txlist = mis['TXLIST']
        textures = do_txlist(txlist, context, search_paths=search_paths, progress=progress, **options)
        textures_time = time.process_time()-start

        start = time.process_time()
        if 'WREXT' in mis:
            worldrep = mis['WREXT']
        elif 'WRRGB' in mis:
            worldrep = mis['WRRGB']
        else:
            worldrep = mis['WR']
        obj = do_worldrep(worldrep, textures, context, name=miss_name, progress=progress, **options)
        worldrep_time = time.process_time()-start
        # Make the object active and selected
        context.view_layer.objects.active = obj
        obj.select_set(True)
        # Clean up loose vertices (from skipped polygons).
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose(use_verts=True, use_edges=False, use_faces=False)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        # I was tempted to do a limited dissolve here, to clean up the
        # mesh a little. But doing so would destroy the uvs, and to deal
        # with that would require cross-baking textures and lightmaps from
        # the imported mesh to the cleaned up one. That's way more
        # complexity than I'm prepared to deal with here and now.
        progress.leave_substeps(f"Finished importing: {filepath!r}")

    print(f"Load textures: {textures_time:0.1f}s")
    print(f"Load worldrep: {worldrep_time:0.1f}s")
    return obj

def do_txlist(chunk, context, search_paths=(), progress=None,
    dump=False, dump_file=None, **options):
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
    ## TODO: '.dds' should come first (per newdark priorities), but i cant
    ##       load them yet.
    tex_extensions = ['.png', '.tga', '.bmp', '.pcx', '.gif', '.cel']
    ext_sort_order = {ext: i for (i, ext) in enumerate(tex_extensions)}
    # This image cache is keyed by unique name (fam_foo_bar), not by Image name.
    # We keep the Image name the same as the filename (as far as we can) so
    # that NewDark exporter (when exporting unbaked objects) gets the right
    # name for images whenever possible.
    image_cache = {}
    def load_tex(fam_name, tex_name):
        fam_name = fam_name.lower()
        tex_name = tex_name.lower()
        # Don't load the image if it has already been loaded.
        cache_name = f"fam_{fam_name}_{tex_name}"
        image = image_cache.get(cache_name, None)
        if image: return image
        image_name = f"{tex_name}.png"
        # Find the candidate files (all matching types in all search paths)
        if dump:
            print(f"Searching for fam/{fam_name}/{tex_name}...", file=dumpf)
        candidates = [] # (sort_key, full_path) tuples
        for path in search_paths:
            base_fam_path = os.path.join(path, 'fam', fam_name)
            for lang in ['', 'english', 'french', 'german', 'russian', 'italian']:
                if lang:
                    fam_path = os.path.join(base_fam_path, lang)
                else:
                    fam_path = base_fam_path
                if (not os.path.exists(fam_path)
                or not os.path.isdir(fam_path)):
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
            ## TODO: need a better way of handling missing textures
            #raise ValueError(f"Cannot find texture {fam_name}/{tex_name}")
            return None
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
        image_cache[cache_name] = image
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
            if image is None:
                progress.step(f"Missing fam/{fam_name}/{tex_name}")
            else:
                progress.step(f"Loaded fam/{fam_name}/{tex_name} size {image.size[0]}x{image.size[1]}")
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
    # Power node (absolute hack so the input feels linearish)
    pow_node = nodes.new(type='ShaderNodeMath')
    pow_node.location = grid(-0.5,0)
    pow_node.label = 'Power'
    pow_node.operation = 'POWER'
    pow_node.inputs[1].default_value = 4.0
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
    links.new(output_node.inputs['AmbientBrightness'], pow_node.outputs[0])
    links.new(pow_node.inputs[0], ambient_brightness_node.outputs['Value'])
    return tree

def create_texture_material(name, texture_image, lightmap_image, lightmap_2x_modulation, settings_group):
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
    #                              |         MixRGB('Mix') <-- NodeGroup('Settings').AmbientBrightness
    #                              |              |
    #                              +--------------+
    #                              |
    #                     MixRGB('Multiply')
    #                              |
    #                          Multiply <-- 1.0 or 2.0 (lightmap_2x_modulation)
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
    mat.use_backface_culling = True
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
    if is_lightmapped:
        mul_node = nodes.new(type='ShaderNodeVectorMath'); mul_node.select = False
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
        if lightmap_2x_modulation:
            mul_node.inputs[1].default_value = (2.0,2.0,2.0)
        else:
            mul_node.inputs[1].default_value = (1.0,1.0,1.0)
        mul_node.operation = 'MULTIPLY'
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
        mul_node.location = grid(-1,0)
        mix_node.location = grid(-2,0)
        tx_img_node.location = grid(-4,1)
        tx_uv_node.location = grid(-5,1)
        lm_img_node.location = grid(-4,-1)
        lm_uv_node.location = grid(-5,-1)
        lm_mix_node.location = grid(-3,-1)
        settings_node.location = grid(-4,-3)
    elif is_textured:
        tx_img_node.location = grid(-1,0)
        tx_uv_node.location = grid(-2,0)
    elif is_lightmapped:
        mul_node.location = grid(-1,0)
        lm_img_node.location = grid(-3,0)
        lm_uv_node.location = grid(-4,0)
        lm_mix_node.location = grid(-2,-0)
    # Link them
    links.new(out_node.inputs['Surface'], bsdf_node.outputs['BSDF'])
    if is_lightmapped and is_textured:
        links.new(bsdf_node.inputs['Base Color'], mul_node.outputs[0])
        links.new(mul_node.inputs[0], mix_node.outputs['Color'])
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
        links.new(bsdf_node.inputs['Base Color'], mul_node.outputs[0])
        links.new(mul_node.inputs[0], lm_mix_node.outputs['Color'])
        links.new(lm_mix_node.inputs['Color1'], lm_img_node.outputs['Color'])
        links.new(lm_img_node.inputs['Vector'], lm_uv_node.outputs['UV'])
    return mat

def poly_calculate_uvs(cell, pi, texture_size, version, lightmap_scale):
    poly = cell.p_polys[pi]
    render = cell.p_render_polys[pi]
    vertices = cell.poly_vertices[pi]
    info = cell.p_light_list[pi]
    #
    # Despite the name, I apply lightmap_scale to the texture scale,
    # because it is affects the lightmap-texels-per-texture-texel factor;
    # the lightmap uvs remain constant (covering approximately the whole
    # polygon), and so as lightmap_scale increases, the texture uvs get
    # smaller to compensate.
    #
    # Note that texture uv scales will still be incorrect if the fm uses
    # terrain_scale in .mtl files, because I don't yet support .mtl files.
    #
    tx_u_scale = 64.0/lightmap_scale/texture_size[0]
    tx_v_scale = 64.0/lightmap_scale/texture_size[1]
    lm_u_scale = 4.0/info.width
    lm_v_scale = 4.0/info.height
    p_uvec = Vector(render.tex_u)
    p_vvec = Vector(render.tex_v)
    u2 = p_uvec.dot(p_uvec)
    v2 = p_vvec.dot(p_vvec)
    uv = p_uvec.dot(p_vvec)
    if version>=(0,30):
        ## TODO: is this a field in the WREXT or not? so far, just using
        ##       vertex 0 as the anchor seems to provide correct results.
        ##       should check across a bunch of missions to see if the
        ##       olddark texture_anchor is ever nonzero, then save them
        ##       in newdark format to see if there is any difference; also
        ##       see if any newdark missions have nonzero 'cached_surface',
        ##       and see if that corresponds with misaligned textures, thus
        ##       implying that field should actually be 'texture_anchor'.
        anchor_vid = 0
        u_base = render.u_base
        v_base = render.v_base
    else:
        anchor_vid = render.texture_anchor
        u_base = float(render.u_base)/(16.0*256.0)
        v_base = float(render.v_base)/(16.0*256.0)
    anchor = Vector(vertices[anchor_vid])
    tx_u_base = u_base*tx_u_scale # u translation
    tx_v_base = v_base*tx_v_scale # v translation
    lm_u_base = lm_u_scale*(u_base+(0.5-float(info.u_base))/4.0) # u translation
    lm_v_base = lm_v_scale*(v_base+(0.5-float(info.v_base))/4.0) # v translation
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
    dump=False, dump_file=None, cell_limit=0, skip_jorge=False, skip_skyhack=False, **options):
    dumpf = dump_file or sys.stdout
    assert (progress is not None)
    progress.enter_substeps(5, "Loading worldrep...")

    version = (chunk.header.version.major, chunk.header.version.minor)

    if chunk.name=='WR':
        if version!=(0,23):
            raise ValueError("Only version 0.23 WR chunk is supported")
    elif chunk.name=='WRRGB':
        if version!=(0,24):
            raise ValueError("Only version 0.24 WRRGB chunk is supported")
    elif chunk.name=='WREXT':
        if version!=(0,30):
            raise ValueError("Only version 0.30 WREXT chunk is supported")
    else:
        raise ValueError(f"Unsupported worldrep chunk {chunk.name}")

    f = StructuredReader(buffer=chunk.data)
    if version>=(0,30):
        header = f.read(LGWREXTHeader)
    else:
        header = f.read(LGWRHeader)
    if dump:
        print(f"{chunk.name} chunk:", file=dumpf)
        print(f"  version: {chunk.header.version.major}.{chunk.header.version.minor}", file=dumpf)
        if version>=(0,30):
            print(f"  unknown0: 0x{header.unknown0:08x}", file=dumpf)
            print(f"  unknown1: 0x{header.unknown1:08x}", file=dumpf)
            print(f"  unknown2: 0x{header.unknown2:08x}", file=dumpf)
            print(f"  lightmap_format: {header.lightmap_format}", file=dumpf)
            print(f"  lightmap_scale: 0x{header.lightmap_scale:08x}", file=dumpf)
        print(f"  data_size: {header.data_size}", file=dumpf)
        print(f"  cell_count: {header.cell_count}", file=dumpf)

    def calculate_lightmap_scale(value):
        value = int(value)
        sign = (1 if value>=0 else -1)
        if value==0: value = 1
        exponent = int(math.log2(abs(value)))
        return 2.0**(sign*exponent)

    lightmap_2x_modulation = False
    lightmap_scale = 1.0
    if version==(0,23):
        lightmap_format = LGWRLightmap8Bit
    elif version==(0,24):
        lightmap_format = LGWRRGBLightmap16Bit
    elif version==(0,30):
        lightmap_scale = calculate_lightmap_scale(header.lightmap_scale)
        if header.lightmap_format==0:
            lightmap_format = LGWRRGBLightmap16Bit
        elif header.lightmap_format==1:
            lightmap_format = LGWRRGBLightmap32Bit
        elif header.lightmap_format==2:
            lightmap_format = LGWRRGBLightmap32Bit
            lightmap_2x_modulation = True
        else:
            raise ValueError(f"Unrecognised lightmap_format {header.lightmap_format}")

    progress.step("Loading cells...")
    cell_progress_step_size = 100
    cell_progress_step_count = (header.cell_count//cell_progress_step_size)+1
    progress.enter_substeps(cell_progress_step_count)
    cells = np.zeros(header.cell_count, dtype=object)
    for cell_index in range(header.cell_count):
        cell_offset = f.offset
        try:
            if cell_index%cell_progress_step_size==0: progress.step()
            cells[cell_index] = LGWRCell.read(f, cell_index, version, lightmap_format)
        except:
            print(f"While reading cell {cell_index} at offset 0x{cell_offset:08x}:", file=sys.stderr)
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
                if version>=(0,30):
                    print(f"        u_base: {rpoly.u_base:0.2f}", file=dumpf)
                    print(f"        v_base: {rpoly.v_base:0.2f}", file=dumpf)
                    print(f"        texture_id: {rpoly.texture_id}", file=dumpf)
                    print(f"        texture_anchor: {getattr(rpoly, 'texture_anchor', None)}", file=dumpf)
                    print(f"        cached_surface: {getattr(rpoly, 'cached_surface', None)}", file=dumpf)
                    print(f"        texture_mag: {rpoly.texture_mag:0.2f}", file=dumpf)
                else:
                    print(f"        u_base: {rpoly.u_base} (0x{rpoly.u_base:04x})", file=dumpf)
                    print(f"        v_base: {rpoly.v_base} (0x{rpoly.v_base:04x})", file=dumpf)
                    print(f"        texture_id: {rpoly.texture_id}", file=dumpf)
                    print(f"        texture_anchor: {rpoly.texture_anchor}", file=dumpf)
                    print(f"        cached_surface: {rpoly.cached_surface}", file=dumpf)
                    print(f"        texture_mag: {rpoly.texture_mag:0.2f}", file=dumpf)
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

    MAX_CELLS = 32678       # Imposed by Dromed
    MAX_VERTICES = 256*1024 # Imposed by Dromed
    MAX_FACES = 256*1024    # Rough guess
    MAX_FACE_INDICES = 32   # Imposed by Dromed
    MAX_INDICES = MAX_FACE_INDICES*MAX_FACES

    JORGE_TEXTURE_ID = 0
    SKYHACK_TEXTURE_ID = 249

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

    def get_texture_image(texture_id):
        if texture_id in (JORGE_TEXTURE_ID, SKYHACK_TEXTURE_ID):
            # Special texture
            return None
        elif (0<=texture_id<len(textures)):
            # Plain old image texture
            return textures[texture_id]
        else:
            # Missing texture
            return None

    def get_texture_size(texture_id):
        image = get_texture_image(texture_id)
        if image is None:
            # Assume Jorge and SKY_HACK (and missing textures) are 64x64:
            return (64,64)
        else:
            return image.size

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
            # Look up the texture.
            texture_id = cell.p_render_polys[pi].texture_id
            # Skip Jorge and Sky Hack
            if skip_jorge and texture_id==JORGE_TEXTURE_ID: continue
            if skip_skyhack and texture_id==SKYHACK_TEXTURE_ID: continue
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
            # Set the material index.
            mat_idx = material_textures.get(texture_id)
            if mat_idx is None:
                mat_idx = len(material_textures)
                material_textures[texture_id] = mat_idx
            material_idxs[face_ptr] = mat_idx
            # Calculate uvs.
            texture_size = get_texture_size(texture_id)
            poly_tx_uvs, poly_lm_uvs = poly_calculate_uvs(cell, pi, texture_size, version, lightmap_scale)
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
    lightmap_image.name = f"{name}_Lightmap"

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
    mat_jorge = create_texture_material('JORGE', None, None, lightmap_2x_modulation, settings_group)
    mat_sky = create_texture_material('SKY_HACK', None, None, lightmap_2x_modulation, settings_group)
    mat_missing = create_texture_material('MISSING', None, None, lightmap_2x_modulation, settings_group)
    texture_ids_needed = [tid
        for (tid, _) in sorted(material_textures.items(), key=(lambda item:item[1]))]
    for texture_id in texture_ids_needed:
        if texture_id==JORGE_TEXTURE_ID:
            mat = mat_jorge
        elif texture_id==SKYHACK_TEXTURE_ID:
            mat = mat_sky
        else:
            im = get_texture_image(texture_id)
            if im is None:
                mat = mat_missing
            else:
                mat = create_texture_material(im.name, im, lightmap_image, lightmap_2x_modulation, settings_group)
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
        handle = len(self.images)
        assert image.shape==(height,width,4), f"image {handle} shape is {image.shape}, not {width}x{height}x4!"
        self.images.append((width, height, handle, image))
        return handle

    def finish(self):
        # Build the atlas
        self.images = sorted(self.images, reverse=True)
        atlas_w, atlas_h = (256, 256)
        atlas_max_w, atlas_max_h = (8192, 8192)
        quadrant_w, quadrant_h = (atlas_w, atlas_h)
        quadrant_index = 0
        # x, y are abs coords of the image placement cursor
        x = y = 0
        # quadrant_x, quadrant_y are abs coords of the current quadrant
        quadrant_x = quadrant_y = 0
        overflow = 0
        row_height = 0
        placements = [None]*len(self.images)
        texels_filled = 0
        DUMP = False
        CONTINUE_ON_OVERFLOW = False
        DEBUG_COLOR_LIGHTMAPS = False
        DEBUG_COLOR_QUADRANTS = False
        DEBUG_COLOR_BACKGROUND = False
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
        if DEBUG_COLOR_BACKGROUND:
            atlas_data[:,:] = debug_colors[quadrant_index%len(debug_colors)]

        if CONTINUE_ON_OVERFLOW:
            # Start with a debug color at 0,0, for atlas overflows
            x,y,w,h = 0,0,16,16
            atlas_data[ y:y+h, x:x+w ] = debug_colors[2]
            x,y = 16,0
            row_height = 16
            overflow = False

        if DUMP:
            dumpf = open('e:/dev/thief/blender/thieftools/atlas.dump', 'w')
            print(f"Placing {len(self.images)} images...", file=dumpf)

        # We get much better packing, even without rotation, if we sort
        # by height first.
        def by_height(entry): return entry[1]
        self.images.sort(key=by_height, reverse=True)

        for image_index, (w, h, handle, image) in enumerate(self.images):
            # Find a place for the image.
            if w>quadrant_w or h>quadrant_h:
                raise ValueError(f"No space to fit image {handle} of {w}x{h}")
            while True: # TODO: break if the atlas is now too big
                if overflow:
                    x = 0
                    y = 0
                    break
                # quadrant's right edge:
                if (x+w)>(quadrant_x+quadrant_w):
                    # Wrap the cursor to the next row up:
                    x = quadrant_x
                    y += row_height
                    row_height = 0
                    if DUMP: print(f"Cursor row-wrapped to {x},{y}", file=dumpf)
                if (y+h)>(quadrant_y+quadrant_h):
                    # Wrap the cursor to the next quadrant across:
                    quadrant_x += quadrant_w
                    quadrant_index += 1
                    x = quadrant_x
                    y = quadrant_y
                    row_height = 0
                    if DUMP: print(f"Cursor quadrant-wrapped right to {x},{y}", file=dumpf)
                if quadrant_x>=atlas_w:
                    # Wrap the cursor to the next quadrant above left:
                    quadrant_x = 0
                    quadrant_y += quadrant_h
                    x = quadrant_x
                    y = quadrant_y
                    row_height = 0
                    if DUMP: print(f"Cursor quadrant-wrapped up-left to {x},{y}", file=dumpf)
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
                    row_height = 0
                    if DUMP: print(f"Atlas expanded to {atlas_w}x{atlas_h}", file=dumpf)
                    if atlas_w>atlas_max_w or atlas_h>atlas_max_h:
                        # Newdark doesn't support >4k textures, so if we
                        # don't have enough space now, someone (me) is gonna
                        # have to write more code to do multiple atlases! D:
                        if CONTINUE_ON_OVERFLOW:
                            print("\nATLAS OVERFLOW\n", file=sys.stderr)
                            if DUMP: print(f"OVERFLOW", file=dumpf)
                            atlas_w = atlas_max_w
                            atlas_h = atlas_max_h
                            overflow = True
                            x = 0
                            y = 0
                            break
                        else:
                            raise ValueError(f"No space to fit image {handle} of {w}x{h} in 4k atlas")
                    # Resize the array with three new blank quadrants:
                    blank = np.zeros((quadrant_w,quadrant_h*4), dtype=float)
                    if DEBUG_COLOR_BACKGROUND:
                        blank.shape = (quadrant_w,quadrant_h,4)
                        blank[:,:] = debug_colors[quadrant_index%len(debug_colors)]
                        blank.shape = (quadrant_w,quadrant_h*4)
                    old_data = atlas_data.reshape((quadrant_w,quadrant_h*4))
                    new_data = np.block([[ old_data, blank ],
                                         [ blank,    blank ]])
                    atlas_data = new_data.reshape((atlas_w,atlas_h,4))
                # We have a valid cursor position
                break
            # Place and blit the image.
            if DUMP: print(f"{image_index} (handle handle): {x},{y} - {w}x{h}", file=dumpf)
            placements[handle] = (x,y,w,h)
            if DEBUG_COLOR_LIGHTMAPS:
                source = debug_colors[image_index%len(debug_colors)]
            elif DEBUG_COLOR_QUADRANTS:
                source = debug_colors[quadrant_index%len(debug_colors)]
            else:
                source = image
            if not overflow:
                atlas_data[ y:y+h, x:x+w ] = source
                # w and h are uint8, so we need to promote them before
                # multiplying:
                texels_filled += (int(w)*int(h))
            # Move the cursor along.
            x += w
            row_height = max(row_height, h)

        # Calculate how much space is available--not in the entire atlas,
        # but in the area of the atlas walked by the cursor so far:
        available_texels = (
            quadrant_y*2*quadrant_w                 # quadrants below
            + quadrant_x*quadrant_h                 # quadrant to the left
            + (y+row_height-quadrant_y)*quadrant_w  # portion of current quadrant
            )
        # If we have no lightmaps (maybe lighting was never built, or all
        # the polys were Jorge and we skipped Jorge polys), then
        # available_texels will be zero.
        if available_texels==0:
            efficiency = 0.0
        else:
            efficiency = (texels_filled/available_texels)*100.0
        percent_filled = (texels_filled/(atlas_w*atlas_h))*100.0
        if DUMP: print(f"Atlas efficiency: {efficiency:0.1f}% (atlas space filled: {percent_filled:0.1f}% of {atlas_w}x{atlas_h})", file=dumpf)
        if DUMP: dumpf.close()
        print(f"Atlas efficiency: {efficiency:0.1f}% (atlas space filled: {percent_filled:0.1f}% of {atlas_w}x{atlas_h})")

        # Create the atlas image.
        atlas_image = bpy.data.images.new(name="Atlas", width=atlas_w, height=atlas_h)
        atlas_image.pixels = atlas_data.reshape((-1,))
        # Pack the atlas, so it won't need to be saved (and won't go blank unexpectedly).
        atlas_image.pack()

        self.images = None # Done with the images now.
        self.image = atlas_image
        self.placements = placements

    def get_uv_transform(self, handle):
        """return (scale_x, scale_y), (offset_x, offset_y)"""
        x,y,w,h = self.placements[handle]
        atlas_w,atlas_h = self.image.size
        return ( (w/atlas_w, h/atlas_h), (x/atlas_w, y/atlas_h) )

#---------------------------------------------------------------------------#
# Baking textures and lightmaps together

BAKE_VERTEX_SHADER_SOURCE = """\
uniform mat4 ModelViewProjectionMatrix;

in vec2 texCoord;
in vec2 lmCoord;
in vec2 pos;
out vec2 texCoord_interp;
out vec2 lmCoord_interp;

void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos.xy, 0.0f, 1.0f);
    gl_Position.z = 1.0;
    texCoord_interp = texCoord;
    lmCoord_interp = lmCoord;
}
"""

BAKE_FRAGMENT_SHADER_SOURCE = """\
in vec2 texCoord_interp;
in vec2 lmCoord_interp;
out vec4 fragColor;

uniform vec4 color;
uniform sampler2D image;
uniform sampler2D lightmap;

void main()
{
    vec4 t = texture(image, texCoord_interp);
    vec4 l = texture(lightmap, lmCoord_interp);
    fragColor = color+t*l; // just random maths
}
"""

def bake_textures_and_lightmaps(context, obj):
    from math import ceil, floor
    # TODO: use the lightmap scale from the mission? but we havent
    # saved it. maybe we need to!
    lightmap_scale = 16.0
    mesh = obj.data
    num_polys = len(mesh.polygons)
    num_loops = len(mesh.loops)
    texture_uv_layer = mesh.uv_layers['UVMap']
    lightmap_uv_layer = mesh.uv_layers['UVLightmap']
    # Find the texture and lightmap images.
    mat_texture_images = [None]*len(mesh.materials)
    mat_lightmap_image = None
    for mat_index,mat in enumerate(mesh.materials):
        # TODO: what if the materials and/or shaders have been changed, or were
        #       imported without both TerrainTexture and LightmapTexture nodes?
        nodes = mat.node_tree.nodes
        mat_texture_images[mat_index] = nodes['TerrainTexture'].image
        if mat_lightmap_image is None:
            mat_lightmap_image = nodes['LightmapTexture'].image
    # Parameters that control the size.
    padding = (1,1) # TODO: decide on what this needs to be.
    lm_atlas_w = mat_lightmap_image.size[0]
    lm_atlas_h = mat_lightmap_image.size[1]
    scaled_pixel_size = np.array(
        [lm_atlas_w*lightmap_scale, lm_atlas_h*lightmap_scale],
        dtype=float32)
    # Copy relevant mesh data into numpy arrays
    loop_starts = np.zeros(num_polys, dtype=int32)
    loop_totals = np.zeros(num_polys, dtype=int32)
    mat_indexes = np.zeros(num_polys, dtype=int32)
    texture_uvs = np.zeros(2*num_loops, dtype=float32)
    lightmap_uvs = np.zeros(2*num_loops, dtype=float32)
    mesh.polygons.foreach_get("loop_start", loop_starts)
    mesh.polygons.foreach_get("loop_total", loop_totals)
    mesh.polygons.foreach_get("material_index", mat_indexes)
    texture_uv_layer.data.foreach_get('uv', texture_uvs)
    lightmap_uv_layer.data.foreach_get('uv', lightmap_uvs)
    texture_uvs.shape = (-1,2)
    lightmap_uvs.shape = (-1,2)
    baked_uvs = lightmap_uvs.copy()
    # Allocate arrays for new data.
    min_vert = np.zeros(2, dtype=float32) # bake atlas pixel space
    max_vert = np.zeros(2, dtype=float32)
    tx_min_uv = np.zeros(2, dtype=float32) # texture UV space
    tx_max_uv = np.zeros(2, dtype=float32)
    lm_min_uv = np.zeros(2, dtype=float32) # lightmap atlas UV space
    lm_max_uv = np.zeros(2, dtype=float32)
    target_verts = np.zeros((num_polys,2,2), dtype=float32) # bake atlas pixel space
    target_tx_uvs = np.zeros((num_polys,2,2), dtype=float32)  # texture UV space
    target_lm_uvs = np.zeros((num_polys,2,2), dtype=float32)  # lightmap atlas UV space
    # Conversions from bake atlas pixel space (by the origin) to
    # texture UV space and lightmap atlas UV space:
    tx_scale = np.zeros(2, dtype=float32)
    tx_origin = np.zeros(2, dtype=float32)
    lm_scale = np.zeros(2, dtype=float32)
    lm_origin = np.zeros(2, dtype=float32)

    for poly_idx,(loop_start,loop_total) \
    in enumerate(zip(loop_starts,loop_totals)):
        loop_end = loop_start+loop_total
        tx_uvs = texture_uvs[loop_start:loop_end]
        lm_uvs = lightmap_uvs[loop_start:loop_end]
        # Find the minimum and maximum uvs.
        np.amin(tx_uvs, axis=0, out=tx_min_uv) # texture UV space
        np.amax(tx_uvs, axis=0, out=tx_max_uv)
        np.amin(lm_uvs, axis=0, out=lm_min_uv) # lightmap atlas UV space
        np.amax(lm_uvs, axis=0, out=lm_max_uv)
        # Derive texture and lightmap scale and offset.
        size = (lm_max_uv-lm_min_uv)*scaled_pixel_size
        tx_scale[:] = (tx_max_uv-tx_min_uv)/size
        tx_origin[:] = tx_min_uv
        lm_scale[:] = (lm_max_uv-lm_min_uv)/size
        lm_origin[:] = lm_min_uv
        # Derive intermediate vertices by flooring/ceiling and padding.
        min_scaled = lm_min_uv*scaled_pixel_size
        min_floored = np.floor(min_scaled)
        min_vert[:] = min_scaled-min_floored
        max_vert[:] = np.ceil(lm_max_uv*scaled_pixel_size-min_scaled)
        min_vert -= padding
        max_vert += padding
        # Calculate baked uvs from lightmap uvs.
        baked_uvs[loop_start:loop_end] = (
            (lm_uvs-lm_min_uv)*scaled_pixel_size-min_vert
            )
        # Calculate texture and lightmap uvs for the vertices.
        target_tx_uvs[poly_idx][0] = min_vert*tx_scale+tx_origin
        target_tx_uvs[poly_idx][1] = max_vert*tx_scale+tx_origin
        target_lm_uvs[poly_idx][0] = min_vert*lm_scale+lm_origin
        target_lm_uvs[poly_idx][1] = max_vert*lm_scale+lm_origin
        # Final vertices are back at the origin again:
        target_verts[poly_idx][0] = 0.0
        # TODO: how come these coords aren't integers??
        target_verts[poly_idx][1] = max_vert-min_vert

    # Atlas all the rects.
    atlas_builder = BakeAtlasBuilder()
    handles = np.zeros(num_polys, dtype=int32)
    for poly_idx, verts in enumerate(target_verts):
        # TODO: these should have been integers, but for now they are not,
        #       so i ceil() them
        w = ceil(verts[1][0])
        h = ceil(verts[1][1])
        handle = atlas_builder.add(w,h)
        handles[poly_idx] = handle
    atlas_builder.finish()
    for poly_idx,(verts,loop_start,loop_total) \
    in enumerate(zip(target_verts,loop_starts,loop_totals)):
        loop_end = loop_start+loop_total
        handle = handles[poly_idx]
        x,y = atlas_builder.get_pos(handle)
        verts[0] += (x,y)
        verts[1] += (x,y)
        baked_uvs[loop_start:loop_end] += (x,y)

    baked_uvs /= atlas_builder.size

    # Set the triangle vertices (in target pixel coords).
    render_vertices = np.zeros((num_polys,6,2), dtype=float32)
    render_tx_uvs = np.zeros((num_polys,6,2), dtype=float32)
    render_lm_uvs = np.zeros((num_polys,6,2), dtype=float32)
    for poly_idx,pos \
    in enumerate(target_verts):
        render_vertices[poly_idx,0,0] = pos[0][0]
        render_vertices[poly_idx,0,1] = pos[0][1]
        render_vertices[poly_idx,1,0] = pos[1][0]
        render_vertices[poly_idx,1,1] = pos[0][1]
        render_vertices[poly_idx,2,0] = pos[1][0]
        render_vertices[poly_idx,2,1] = pos[1][1]
        render_vertices[poly_idx,3,0] = pos[1][0]
        render_vertices[poly_idx,3,1] = pos[1][1]
        render_vertices[poly_idx,4,0] = pos[0][0]
        render_vertices[poly_idx,4,1] = pos[1][1]
        render_vertices[poly_idx,5,0] = pos[0][0]
        render_vertices[poly_idx,5,1] = pos[0][1]
        uv = target_tx_uvs[poly_idx]
        render_tx_uvs[poly_idx,0,0] = uv[0][0]
        render_tx_uvs[poly_idx,0,1] = uv[0][1]
        render_tx_uvs[poly_idx,1,0] = uv[1][0]
        render_tx_uvs[poly_idx,1,1] = uv[0][1]
        render_tx_uvs[poly_idx,2,0] = uv[1][0]
        render_tx_uvs[poly_idx,2,1] = uv[1][1]
        render_tx_uvs[poly_idx,3,0] = uv[1][0]
        render_tx_uvs[poly_idx,3,1] = uv[1][1]
        render_tx_uvs[poly_idx,4,0] = uv[0][0]
        render_tx_uvs[poly_idx,4,1] = uv[1][1]
        render_tx_uvs[poly_idx,5,0] = uv[0][0]
        render_tx_uvs[poly_idx,5,1] = uv[0][1]
        uv = target_lm_uvs[poly_idx]
        render_lm_uvs[poly_idx,0,0] = uv[0][0]
        render_lm_uvs[poly_idx,0,1] = uv[0][1]
        render_lm_uvs[poly_idx,1,0] = uv[1][0]
        render_lm_uvs[poly_idx,1,1] = uv[0][1]
        render_lm_uvs[poly_idx,2,0] = uv[1][0]
        render_lm_uvs[poly_idx,2,1] = uv[1][1]
        render_lm_uvs[poly_idx,3,0] = uv[1][0]
        render_lm_uvs[poly_idx,3,1] = uv[1][1]
        render_lm_uvs[poly_idx,4,0] = uv[0][0]
        render_lm_uvs[poly_idx,4,1] = uv[1][1]
        render_lm_uvs[poly_idx,5,0] = uv[0][0]
        render_lm_uvs[poly_idx,5,1] = uv[0][1]

    DRAW_BAKED_POLYS = False
    baked_image = _bake_together(
        f"{obj.name}_b",
        atlas_builder.size,
        num_polys,
        render_vertices,
        render_tx_uvs,
        render_lm_uvs,
        mat_indexes,
        mat_texture_images,
        mat_lightmap_image,
        DRAW_BAKED_POLYS,
        loop_starts,
        loop_totals,
        baked_uvs,
        texture_uvs,
        lightmap_uvs)

    # TODO: decide if we want to duplicate first, or if we
    #       should just copy the mesh and create a new object
    #       (as here). right now i think the latter, because
    #       we don't want other level-specific datablocks like
    #       settings copied over!

    CREATE_OBJECT = True
    if CREATE_OBJECT:
        # Duplicate the mesh, and put it on a new object.
        old_mesh = mesh
        old_obj = obj
        mesh = mesh.copy()
        name = f"{old_obj.name}_b"
        obj = create_object(name, mesh, old_obj.location, context=context, link=True)
        # Create a new material with the baked atlas.
        mat = create_texture_material(name, baked_image, None, False, None)
        mesh.materials.clear()
        mesh.materials.append(mat)
        for poly in mesh.polygons:
            poly.material_index = 0
        # Update the uv layers.
        texture_uv_layer = mesh.uv_layers['UVMap']
        lightmap_uv_layer = mesh.uv_layers['UVLightmap']
        mesh.uv_layers.remove(lightmap_uv_layer)
        baked_uvs.shape = -1
        texture_uv_layer.data.foreach_set('uv', baked_uvs)
        # Make it selected and active.
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = obj
        obj.select_set(True)

    return obj

def _bake_together(name, size, num_polys, verts, tx_uvs, lm_uvs,
        mat_indexes, mat_texture_images, mat_lightmap_image,
        draw_baked_polys, loop_starts, loop_totals, baked_verts,
        baked_tx_uvs, baked_lm_uvs):
    from time import perf_counter
    shader = gpu.types.GPUShader(BAKE_VERTEX_SHADER_SOURCE,
        BAKE_FRAGMENT_SHADER_SOURCE)

    # TODO: these should be parameters
    IMAGE_NAME = name
    WIDTH = size[0]
    HEIGHT = size[1]
    offscreen = gpu.types.GPUOffScreen(WIDTH, HEIGHT)

    # Force images to use linear colorspace while we blend (because texture
    # creation doesn't have a parameter for this).
    texture_colorspaces = []
    lightmap_colorspace = mat_lightmap_image.colorspace_settings.name
    for image in mat_texture_images:
        texture_colorspaces.append(image.colorspace_settings.name)
        image.colorspace_settings.name = 'Linear'
    mat_lightmap_image.colorspace_settings.name = 'Linear'
    def restore_colorspaces():
        for image, name in zip(mat_texture_images, texture_colorspaces):
            image.colorspace_settings.name = name
        mat_lightmap_image.colorspace_settings.name = lightmap_colorspace

    def cleanup():
        offscreen.free()
        restore_colorspaces()

    try:
        # Get all these damn images on the damn gpu:
        tx_textures = [None] * len(mat_texture_images)
        tx_lightmap = None
        tx_lightmap = gpu.texture.from_image(mat_lightmap_image)
        for i,image in enumerate(mat_texture_images):
            tx_textures[i] = gpu.texture.from_image(image)

        with offscreen.bind():
            # Clear the render target
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0,0.0,0.0,1.0))
            # Set up the shader and uniforms
            t_start = perf_counter()
            shader.bind()
            shader.uniform_sampler("lightmap", tx_lightmap)
            # Draw the rects
            color = np.array([0.0,0.0,0.0,1.0], dtype=np.float32)
            shader.uniform_vector_float(shader.uniform_from_name("color"), color, 4, 1)
            with gpu.matrix.push_pop():
                # Map (0,0)-(WIDTH,HEIGHT) into NDC (-1,-1)-(1,1)
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.translate((-1.0,-1.0))
                gpu.matrix.scale((2.0/WIDTH,2.0/HEIGHT))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))
                for pi in range(num_polys):
                    mat = mat_indexes[pi]
                    batch = batch_for_shader(shader, 'TRIS',
                        {"pos": verts[pi], "texCoord": tx_uvs[pi], "lmCoord": lm_uvs[pi]})
                    shader.uniform_sampler("image", tx_textures[mat])
                    batch.draw(shader)
            # Draw the debug polys on top.
            if draw_baked_polys:
                color = np.array([0.0,0.05,0.2,1.0], dtype=np.float32)
                shader.uniform_vector_float(shader.uniform_from_name("color"), color, 4, 1)
                with gpu.matrix.push_pop():
                    # Map (0,0)-(1.0,1.0) into NDC (-1,-1)-(1,1)
                    gpu.matrix.load_matrix(Matrix.Identity(4))
                    gpu.matrix.translate((-1.0,-1.0))
                    gpu.matrix.scale((2.0,2.0))
                    gpu.matrix.load_projection_matrix(Matrix.Identity(4))
                    for pi,(loop_start,loop_total) in enumerate(zip(loop_starts,loop_totals)):
                        loop_end = loop_start+loop_total
                        mat = mat_indexes[pi]
                        batch = batch_for_shader(shader, 'TRI_FAN', {
                            "pos": baked_verts[loop_start:loop_end],
                            "texCoord": baked_tx_uvs[loop_start:loop_end],
                            "lmCoord": baked_lm_uvs[loop_start:loop_end],
                            })
                        shader.uniform_sampler("image", tx_textures[mat])
                        batch.draw(shader)
            t_render = perf_counter()-t_start
            # Read back the offscreen buffer
            t_start = perf_counter()
            pixels = np.empty((HEIGHT,WIDTH,4), dtype=np.uint8)
            buffer = gpu.types.Buffer('UBYTE', pixels.shape, pixels)
            fb.read_color(0, 0, WIDTH, HEIGHT, 4, 0, 'UBYTE', data=buffer)
            t_readback = perf_counter()-t_start
    except:
        raise
    else:
        # Create an image from it
        t_start = perf_counter()
        if not IMAGE_NAME in bpy.data.images:
            bpy.data.images.new(IMAGE_NAME, WIDTH, HEIGHT)
        image = bpy.data.images[IMAGE_NAME]
        image.scale(WIDTH, HEIGHT)
        t_create_image = perf_counter()-t_start
        t_start = perf_counter()
        pixels = pixels/255.0
        pixels.shape = -1
        t_tofloat = perf_counter()-t_start
        t_start = perf_counter()
        image.pixels[:] = pixels
        t_setpixels = perf_counter()-t_start
        print(f"render: {t_render:0.3f}\n"
              f"readback: {t_readback:0.3f}\n"
              f"create_image: {t_create_image:0.3f}\n"
              f"tofloat: {t_tofloat:0.3f}\n"
              f"setpixels: {t_setpixels:0.3f}\n")
        return image
    finally:
        cleanup()

class BakeAtlasBuilder:
    def __init__(self):
        self.rects = []
        # After close, these will be set:
        self.placements = None

    def add(self, width, height):
        handle = len(self.rects)
        self.rects.append((width, height, handle))
        return handle

    def finish(self):
        # Build the atlas
        # TODO: these should probably be parameters!
        atlas_w, atlas_h = (1024,1024)
        atlas_max_w, atlas_max_h = (8192, 8192)
        quadrant_w, quadrant_h = (atlas_w, atlas_h)
        quadrant_index = 0
        # x, y are abs coords of the placement cursor
        x = y = 0
        # quadrant_x, quadrant_y are abs coords of the current quadrant
        quadrant_x = quadrant_y = 0
        overflow = 0
        row_height = 0
        placements = [None]*len(self.rects)
        texels_filled = 0
        DUMP = False
        CONTINUE_ON_OVERFLOW = False
        #
        # To fit the rects into the atlas, we place it at the cursor, if it
        # will fit. If not, we reset the cursor to the left edge, and move it
        # up by the row height; then reset the row height to zero. The row
        # height is the max height of the rects placed on that row.
        #
        # The placements are the (x,y,w,h) tuples of the anchor where the
        # corresponding rect was placed.
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

        if CONTINUE_ON_OVERFLOW:
            # Start with a debug color at 0,0, for atlas overflows
            x,y,w,h = 0,0,16,16
            x,y = 16,0
            row_height = 16
            overflow = False

        if DUMP:
            dumpf = open('e:/dev/thief/blender/thieftools/atlas.dump', 'w')
            print(f"Placing {len(self.rects)} rects...", file=dumpf)

        # We get much better packing, even without rotation, if we sort
        # by height first.
        def by_height(entry): return entry[1]
        self.rects.sort(key=by_height, reverse=True)

        for rect_index, (w, h, handle) in enumerate(self.rects):
            # Find a place for the rect.
            if w>quadrant_w or h>quadrant_h:
                raise ValueError(f"No space to fit rect {handle} of {w}x{h}")
            while True: # TODO: break if the atlas is now too big
                if overflow:
                    x = 0
                    y = 0
                    break
                # quadrant's right edge:
                if (x+w)>(quadrant_x+quadrant_w):
                    # Wrap the cursor to the next row up:
                    x = quadrant_x
                    y += row_height
                    row_height = 0
                    if DUMP: print(f"Cursor row-wrapped to {x},{y}", file=dumpf)
                if (y+h)>(quadrant_y+quadrant_h):
                    # Wrap the cursor to the next quadrant across:
                    quadrant_x += quadrant_w
                    quadrant_index += 1
                    x = quadrant_x
                    y = quadrant_y
                    row_height = 0
                    if DUMP: print(f"Cursor quadrant-wrapped right to {x},{y}", file=dumpf)
                if quadrant_x>=atlas_w:
                    # Wrap the cursor to the next quadrant above left:
                    quadrant_x = 0
                    quadrant_y += quadrant_h
                    x = quadrant_x
                    y = quadrant_y
                    row_height = 0
                    if DUMP: print(f"Cursor quadrant-wrapped up-left to {x},{y}", file=dumpf)
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
                    row_height = 0
                    if DUMP: print(f"Atlas expanded to {atlas_w}x{atlas_h}", file=dumpf)
                    if atlas_w>atlas_max_w or atlas_h>atlas_max_h:
                        # Newdark doesn't support >4k textures, so if we
                        # don't have enough space now, someone (me) is gonna
                        # have to write more code to do multiple atlases! D:
                        if CONTINUE_ON_OVERFLOW:
                            print("\nATLAS OVERFLOW\n", file=sys.stderr)
                            if DUMP: print(f"OVERFLOW", file=dumpf)
                            atlas_w = atlas_max_w
                            atlas_h = atlas_max_h
                            overflow = True
                            x = 0
                            y = 0
                            break
                        else:
                            raise ValueError(f"No space to fit rect {handle} of {w}x{h} in 4k atlas")
                # We have a valid cursor position
                break
            # Place the rect.
            if DUMP: print(f"{rect_index} (handle handle): {x},{y} - {w}x{h}", file=dumpf)
            placements[handle] = (x,y,w,h) # TODO: id??
            if not overflow:
                # w and h are uint8, so we need to promote them before
                # multiplying:
                texels_filled += (int(w)*int(h))
            # Move the cursor along.
            x += w
            row_height = max(row_height, h)

        # Calculate how much space is available--not in the entire atlas,
        # but in the area of the atlas walked by the cursor so far:
        available_texels = (
            quadrant_y*2*quadrant_w                 # quadrants below
            + quadrant_x*quadrant_h                 # quadrant to the left
            + (y+row_height-quadrant_y)*quadrant_w  # portion of current quadrant
            )
        # If we have no lightmaps (maybe lighting was never built, or all
        # the polys were Jorge and we skipped Jorge polys), then
        # available_texels will be zero.
        if available_texels==0:
            efficiency = 0.0
        else:
            efficiency = (texels_filled/available_texels)*100.0
        percent_filled = (texels_filled/(atlas_w*atlas_h))*100.0
        if DUMP: print(f"Atlas efficiency: {efficiency:0.1f}% (atlas space filled: {percent_filled:0.1f}% of {atlas_w}x{atlas_h})", file=dumpf)
        if DUMP: dumpf.close()
        print(f"Atlas efficiency: {efficiency:0.1f}% (atlas space filled: {percent_filled:0.1f}% of {atlas_w}x{atlas_h})")

        self.rects = None # Done with the rects now.
        self.size = (atlas_w,atlas_h)
        self.placements = placements

    def get_pos(self, handle):
        """return (x, y)"""
        x,y,w,h = self.placements[handle]
        return (x,y)


# --------------------------------------------------------------------------#
# Removing lightmaps (for unbaked Blender NewDark Toolkit compatibility)

def remove_lightmaps(context, obj):
    # Remove lightmap (and supporting) nodes in all the mesh's materials (if present).
    def remove_lightmap_nodes(node_tree):
        tex_node = node_tree.nodes.get('TerrainTexture')
        if tex_node is None: return
        def is_bsdf(n):
            return n.type.startswith('BSDF')
        dead_nodes = []
        node = tex_node
        left_socket = None
        right_socket = None
        while (node is not None) and (not is_bsdf(node)):
            if len(node.outputs) == 0: break
            if node is not tex_node:
                dead_nodes.append(node)
            from_socket = node.outputs[0]
            if left_socket is None:
                left_socket = from_socket
            if len(from_socket.links) == 0: break
            to_socket = from_socket.links[0].to_socket
            right_socket = to_socket
            node = to_socket.node
        if not is_bsdf(node):
            return
        bsdf_node = node

        # Unlink everything in between        
        for link in list(left_socket.links):
            node_tree.links.remove(link)
        for link in list(right_socket.links):
            node_tree.links.remove(link)

        # Link texture directly to bsdf
        node_tree.links.new(left_socket, right_socket)

        # TODO: clean up dead nodes (barely matters ofc)

    mesh = obj.data
    for mat in mesh.materials:
        if not mat.use_nodes: continue
        remove_lightmap_nodes(mat.node_tree)
        # lm_node = mat.node_tree.nodes.get('LightmapTexture')
        # # Trace first output from LightmapTexture to the BSDF. Keep track of the
        # #
        # if lm_node is None: continue
        #     # mix_node = lm_node.outputs[0].links[0].to_node

    BSDF_NODE_TYPES = (
        bpy.types.ShaderNodeBsdfDiffuse,
        bpy.types.ShaderNodeBsdfGlass,
        bpy.types.ShaderNodeBsdfGlossy,
        bpy.types.ShaderNodeBsdfPrincipled,
        bpy.types.ShaderNodeBsdfToon,
        bpy.types.ShaderNodeBsdfTranslucent,
        bpy.types.ShaderNodeBsdfTransparent,
        )




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
    bl_idname = "import_scene.tt_mis"
    bl_label = "Import .MIS"
    bl_options = {'PRESET', 'UNDO'}

    filter_glob: StringProperty(
            default="*.mis;*.cow",
            options={'HIDDEN'},
            )

    skip_jorge: BoolProperty(
            name="Skip Jorge",
            description="Don't import polys with Jorge texture.",
            default=True,
            )

    skip_skyhack: BoolProperty(
            name="Skip SKY HACK",
            description="Don't import polys with SKY HACK texture.",
            default=True,
            )

    dump: BoolProperty(
            name="Write .dump file",
            description="Write debug info to <missname>.dump text file.",
            default=False,
            )

    cell_limit: IntProperty(
            name="Cell limit",
            description="Import only the first N cells of the mission.",
            default=0,
            min=0,
            )

    def invoke(self, context, event):
        from .prefs import get_preferences, show_preferences
        prefs = get_preferences(context)
        search_paths = prefs.texture_paths()
        # Check that search paths has been set (e.g. to the Thief directory)
        # and that all search paths actually exist.
        if not search_paths:
            show_preferences()
            self.report({'ERROR'}, f"Resource search paths not set.")
            return {'CANCELLED'}
        for path in search_paths:
            if not os.path.exists(path):
                show_preferences()
                self.report({'ERROR'}, f"Resource search path {path} does not exist.")
                return {'CANCELLED'}
        # Default to the same file last opened (so the dialog will show
        # the same directory).
        if prefs.last_filepath and not self.filepath:
            self.filepath = prefs.last_filepath
        return super().invoke(context, event)

    def execute(self, context):
        from .prefs import get_preferences, show_preferences
        # Save the last filepath (so next time we open, we will be in the
        # same folder.
        prefs = get_preferences(context)
        prefs.last_filepath = self.filepath
        # Always include the mission's directory in resource search paths.
        search_paths = list(prefs.texture_paths())
        mission_dir = os.path.dirname(self.filepath)
        search_paths.append(mission_dir)

        options = self.as_keywords(ignore=('filter_glob',))

        # TODO: remove the profile option at some point.
        PROFILE = False
        if PROFILE:
            import cProfile
            o = None
            cProfile.runctx("o = import_mission(context, search_paths=search_paths, **options)",
                globals(), locals(), "e:/temp/import_mission.prof")
        else:
            o = import_mission(context, search_paths=search_paths, **options)
        return {'FINISHED'}

    def draw(self, context):
        # Don't draw any properties here; the panels below will draw the
        # ui that we need.
        pass

class MIS_PT_import_options(bpy.types.Panel):
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_label = "Options"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "IMPORT_SCENE_OT_tt_mis"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.
        sfile = context.space_data
        operator = sfile.active_operator
        layout.prop(operator, "skip_jorge")
        layout.prop(operator, "skip_skyhack")

class MIS_PT_import_debug(bpy.types.Panel):
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_label = "Debug"
    bl_parent_id = "FILE_PT_operator"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "IMPORT_SCENE_OT_tt_mis"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.
        sfile = context.space_data
        operator = sfile.active_operator
        layout.prop(operator, "dump")
        layout.prop(operator, "cell_limit")


class TT_bake_together(Operator):
    bl_idname = "object.tt_bake_together"
    bl_label = "Bake together"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(self, context):
        if context.mode!="OBJECT": return False
        # Only run this operator for missions
        o = context.active_object
        if o is None: return False
        return o.tt_mission.is_mission

    def execute(self, context):
        obj_old = context.active_object
        DUPLICATE_FIRST = False
        if DUPLICATE_FIRST:
            # Duplicate this object
            bpy.ops.object.select_all(action='DESELECT')
            obj_old.select_set(True)
            bpy.ops.object.duplicate(linked=False, mode='TRANSLATION')
            obj_new = context.active_object
            # Hide the old object
            obj_old.hide_viewport = True
        else:
            obj_new = obj_old
        bake_textures_and_lightmaps(context, obj_new)
        return {'FINISHED'}

class TT_remove_lightmaps(Operator):
    bl_idname = "object.tt_remove_lightmaps"
    bl_label = "Remove lightmaps"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(self, context):
        if context.mode!="OBJECT": return False
        # Only run this operator for missions
        o = context.active_object
        if o is None: return False
        return o.tt_mission.is_mission

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        remove_lightmaps(context, context.active_object)
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
