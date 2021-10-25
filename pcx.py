import bpy
import os
import struct
import sys

from bpy.props import StringProperty
from bpy.types import Operator
from .binstruct import *

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
    scanlines = []
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
            raise ValueError(f"Error in PCX at 0x{offset-1:08x}")
        scanlines.append(bytes(scanline))
    # Check for an rgb palette
    rgb_palette_size = 3*256
    rgb_palette = bytearray(rgb_palette_size)
    if offset<len(view) and (offset+rgb_palette_size)<len(view):
        b = view[offset]
        offset += 1
        if b==0x0C:
            rgb_palette[:] = view[offset:offset+rgb_palette_size]
            offset += rgb_palette_size
    else:
        raise ValueError(f"PCX without an RGB palette is not supported")

    # Construct a blender image (which has to be bottom-up)
    float_palette = [(b/255.0) for b in rgb_palette]
    pixels = []
    for row in reversed(scanlines):
        for x in range(width):
            i = 3*row[x]
            pixels.extend(float_palette[i:i+3])
            pixels.append(1.0)
    name = os.path.basename(filename)
    image = bpy.data.images.new(name, width, height, alpha=True)
    image.pixels = pixels
    return image

#---------------------------------------------------------------------------#
# Operators

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
