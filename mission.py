import bpy
import bmesh
import base64
import math
import mathutils
import random
import struct
import zlib

from array import array
from dataclasses import dataclass
from itertools import islice
from bpy.props import IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from mathutils import Vector
from mathutils.bvhtree import BVHTree

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

        do_mission_file(self.filename)

        #context.view_layer.objects.active = o
        #o.select_set(True)

        return {'FINISHED'}

def do_mission_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        do_mission(data)

def do_mission(data):
    assert(type(data) is bytes)
    #unpack_from(FMT_TAGFILE_HEADER
    t = TagFileHeader()

def structclass(cls):
    print(">>  >>  >>  >>  >>  >>  >>  >>  >>  ")
    parts = []
    for name, fieldtype in cls.__annotations__.items():
        try:
            fmt = fieldtype._struct_format
        except AttributeError:
            raise TypeError(f"Field '{name}' is not structclass-compatible ({fieldtype})")
        else:
            parts.append(fmt)
        print(f"{name}: {fieldtype!r}")
    fmt = ''.join(parts)
    size = struct.calcsize(fmt)
    cls._struct_format = fmt
    cls._struct_size = size
    print(f"Format: {fmt} ({size} bytes)")
    print("--  --  --  --  --  --  --  --  --  ")
    return dataclass(cls)

class int16(int):
    _struct_format = "h"

class int32(int):
    _struct_format = "i"

class uint16(int):
    _struct_format = "H"

class uint32(int):
    _struct_format = "I"

@structclass
class TagVersion:
    major : uint32
    minor : uint32

@structclass
class TagFileHeader:
    table_offset : uint32
    version : TagVersion
    pad : ??? how do we do fixed-size arrays?
    deadbeef : ???
