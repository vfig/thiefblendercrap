import bpy
import numpy
import os
import sys

from bpy.props import StringProperty
from bpy.types import Operator
from .binstruct import *

#---------------------------------------------------------------------------#
# GIF

class GIFFileHeader(Struct):
    tag: ByteString(6)
    width: uint16
    height: uint16
    flags: uint8
    transparent_index: uint8
    pixel_aspect: uint8

class GIFColorTable:
    def __init__(self, count):
        self.count = count

    def read(self, view, offset=0):
        # read rgb bytes, and convert to array of (r,g,b,a) floats
        size = self.size()
        rgb = numpy.frombuffer(view, dtype=numpy.uint8, count=size, offset=offset)
        rgbf = numpy.array(rgb, dtype=float)/255.0
        rgbf.shape = (-1, 3)
        rgbaf = numpy.insert(rgbf, 3, 1.0, axis=1)
        return rgbaf

    def size(self):
        return 3*self.count

class GIFImageHeader(Struct):
    x: uint16
    y: uint16
    width: uint16
    height: uint16
    flags: uint8

GIF_IMAGE_BLOCK = 0x2C
GIF_EXTENSION_BLOCK = 0x21
GIF_EOF_BLOCK = 0x3B

def load_gif(filename):
    with open(filename, "rb") as f:
        data = f.read()
    view = memoryview(data)
    offset = 0
    file_header = GIFFileHeader.read(view, offset=offset)
    offset += GIFFileHeader.size()
    if file_header.tag not in (b'GIF87a', b'GIF89a'):
        raise ValueError("Not a valid GIF")
    color_table_count = 1<<((file_header.flags&0x07)+1)
    color_table_sorted = bool(file_header.flags&0x08)
    color_depth = (file_header.flags&0x70)+1
    has_color_table = bool(file_header.flags&0x80)
    if has_color_table:
        color_table_type = GIFColorTable(color_table_count)
        global_color_table = color_table_type.read(view, offset=offset)
        offset += color_table_type.size()
    else:
        global_color_table = None

    def iter_data_subblocks():
        nonlocal offset, view
        while True:
            b = view[offset]
            offset += 1
            if b==0: break
            block = view[offset:offset+b]
            offset += b
            yield from block

    def decompress_lzw(data, symbol_bit_width, decompressed_size):
        # This is mediocre lzw decompression code that is pretty slow and quite
        # wasteful on memory usage. But it's only going to be used on a handful
        # of under-100kb gifs at a time. It's good enough.
        table = []
        code_size = symbol_bit_width+1
        clr_code = (1<<symbol_bit_width)
        end_code = ((1<<symbol_bit_width)|1)
        prev_code = -1
        def clear_table():
            nonlocal table, code_size, prev_code
            table = [bytes((i,)) for i in range(1<<symbol_bit_width)]
            table.append(b'clr_code')
            table.append(b'end_code')
            code_size = symbol_bit_width+1
            prev_code = -1
        decompressed = bytearray(int(decompressed_size))
        write_cursor = 0
        def write_bytes(b):
            nonlocal write_cursor
            start = write_cursor
            end = start+len(b)
            decompressed[ start:end ] = b
            write_cursor = end
        try:
            data = iter(data)
            input_byte = next(data)
            input_bit = 0
            while True:
                # Read one code
                code = 0
                code_bit = 0
                while True:
                    code_bits_left = code_size-code_bit
                    if code_bits_left==0:
                        break
                    input_bits_left = 8-input_bit
                    if input_bits_left==0:
                        input_byte = next(data)
                        input_bit = 0
                        input_bits_left = 8-input_bit
                    bits = min(input_bits_left, code_bits_left)
                    mask = (1<<bits)-1
                    code |= (((input_byte>>input_bit)&mask)<<code_bit)
                    code_bit += bits
                    input_bit += bits
                # Process the code
                if code==clr_code:
                    clear_table()
                elif code==end_code:
                    # Done
                    break
                elif code<len(table):
                    s = table[code]
                    write_bytes(s)
                    if prev_code!=-1:
                        prev_s = table[prev_code]
                        table.append(prev_s+s[:1])
                    prev_code = code
                elif code==len(table):
                    prev_s = table[prev_code]
                    s = prev_s+prev_s[:1]
                    write_bytes(s)
                    table.append(s)
                    prev_code = code
                else:
                    raise ValueError(f"Error in GIF LZW stream")
                # Increase the code size when the table is full.
                if len(table)==(1<<code_size):
                    if code_size<12:
                        code_size += 1
        except StopIteration:
            raise ValueError(f"Error in GIF")
        return decompressed

    while True:
        b = view[offset]
        offset += 1
        if b==GIF_IMAGE_BLOCK:
            image_header = GIFImageHeader.read(view, offset=offset)
            if (image_header.x!=0
            or image_header.y!=0
            or image_header.width!=file_header.width
            or image_header.height!=file_header.height):
                raise ValueError(f"Nonzero image position in GIF is not supported.")
            offset += image_header.size()
            color_table_count = 1<<((image_header.flags&0x07)+1)
            color_table_sorted = bool(image_header.flags&0x20)
            is_interlaced = bool(image_header.flags&0x40)
            has_color_table = bool(image_header.flags&0x80)
            if has_color_table:
                color_table_type = GIFColorTable(color_table_count)
                color_table = color_table_type.read(view, offset=offset)
                offset += color_table_type.size()
            else:
                color_table = global_color_table
            if color_table is None:
                raise ValueError(f"Missing color table in GIF at 0x{offset:08x}")
            # Read the image data
            symbol_bit_width = view[offset]
            offset += 1
            subblocks = iter_data_subblocks()
            width = image_header.width
            height = image_header.height
            image_size = height*width
            image_data = decompress_lzw(subblocks, symbol_bit_width, image_size)
            # Construct a blender image: flat array, y-up, of rgba floats:
            pixels = numpy.frombuffer(image_data, dtype=numpy.uint8)
            pixels.shape = (height, width)
            pixels = pixels[::-1]
            pixels = color_table[pixels]
            pixels.shape = (-1,)
            name = os.path.basename(filename)
            image = bpy.data.images.new(name, width, height, alpha=True)
            image.pixels = pixels
            # We don't care about animated gifs; just return the first image.
            return image
        elif b==GIF_EXTENSION_BLOCK:
            # We don't care about extension blocks, so skip them.
            extension_type = view[offset]
            offset += 1
            subblocks = bytearray(iter_data_subblocks())
        elif b==GIF_EOF_BLOCK:
            # End of file.
            break
        else:
            raise ValueError(f"Error in GIF at 0x{offset:08x}")
    # If we didn't find and return an image already, something went wrong.
    raise ValueError(f"No image found in GIF")

#---------------------------------------------------------------------------#
# PCX

class PCXHeader(Struct):
    tag: uint8
    version: uint8
    compression: uint8
    bpp: uint8
    x_min: uint16
    y_min: uint16
    x_max: uint16
    y_max: uint16
    x_dpi: uint16
    y_dpi: uint16
    palette: Array(uint8, 48)
    reserved0: uint8
    planes: uint8
    bytes_per_row: uint16
    palette_mode: uint16
    x_dpi_src: uint16
    y_dpi_src: uint16
    reserved1: Array(uint8, 54)

def load_pcx(filename):
    with open(filename, "rb") as f:
        data = f.read()
    view = memoryview(data)
    offset = 0
    header = PCXHeader.read(view, offset=offset)
    if header.tag != 0x0A:
        raise ValueError("Not a valid PCX")
    if header.version not in (0, 2, 3, 4, 5):
        raise ValueError(f"Version {header.version} PCX is not supported")
    if header.compression != 1:
        raise ValueError(f"Compression mode {header.compression} is not supported")
    # I would have implemented support for other bit depths or planes, but it
    # turns out the only Thief .pcxes that use them are some newsky textures.
    if header.bpp != 8:
        raise ValueError(f"{header.bpp}bpp is not supported")
    if header.planes != 1:
        raise ValueError(f"{header.planes} planes is not supported")
    offset += PCXHeader.size()
    # Read the pixel data
    width = header.x_max-header.x_min+1
    height = header.y_max-header.y_min+1
    scanline = bytearray(header.bytes_per_row)
    image_data = bytearray()
    for y in range(height):
        x = 0
        while x<header.bytes_per_row:
            b = view[offset]
            offset += 1
            if (b&0xC0)==0xC0:
                run_length = b&0x3f
                if run_length==0:
                    raise ValueError(f"Error in PCX at 0x{offset-1:08x}")
                b = view[offset]
                offset += 1
                for i in range(run_length):
                    scanline[x] = b
                    x += 1
            else:
                scanline[x] = b
                x += 1
        if x>header.bytes_per_row:
            raise ValueError(f"Error in PCX at 0x{offset:08x}")
        image_data.extend(scanline)
    # Check for an rgb palette
    rgb_palette_size = 3*256
    if offset<len(view) and (offset+rgb_palette_size)<len(view):
        b = view[offset]
        offset += 1
        if b==0x0C:
            rgb_palette = numpy.frombuffer(view, dtype=numpy.uint8,
                count=rgb_palette_size, offset=offset)
            offset += rgb_palette_size
    else:
        raise ValueError(f"PCX without an RGB palette is not supported")

    # Expand the rgb palette to rgba floats:
    float_palette = numpy.array(rgb_palette, dtype=float)/255.0
    float_palette.shape = (-1, 3)
    float_palette = numpy.insert(float_palette, 3, 1.0, axis=1)

    # Construct a blender image: flat array, y-up, of rgba floats:
    pixels = numpy.frombuffer(image_data, dtype=numpy.uint8)
    pixels.shape = (height, width)
    pixels = pixels[::-1]
    pixels = float_palette[pixels] # This still feels like magic!
    pixels.shape = (-1,)
    name = os.path.basename(filename)
    image = bpy.data.images.new(name, width, height, alpha=True)
    image.pixels = pixels
    return image

#---------------------------------------------------------------------------#
# Operators

class TTDebugImportGIFOperator(Operator):
    bl_idname = "object.tt_debug_import_gif"
    bl_label = "Import GIF"
    bl_options = {'REGISTER', 'UNDO'}

    filename : StringProperty()

    def execute(self, context):
        if context.mode != "OBJECT":
            self.report({'WARNING'}, f"{self.bl_label}: must be in Object mode.")
            return {'CANCELLED'}

        load_gif(self.filename)
        return {'FINISHED'}

class TTDebugImportPCXOperator(Operator):
    bl_idname = "object.tt_debug_import_pcx"
    bl_label = "Import PCX"
    bl_options = {'REGISTER', 'UNDO'}

    filename : StringProperty()

    def execute(self, context):
        if context.mode != "OBJECT":
            self.report({'WARNING'}, f"{self.bl_label}: must be in Object mode.")
            return {'CANCELLED'}

        load_pcx(self.filename)
        return {'FINISHED'}
