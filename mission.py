import bpy
import math
import mathutils
import os
import sys

from bpy.props import IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from collections import OrderedDict
from mathutils import Vector
from typing import Mapping
from .binstruct import *
from .lgtypes import *

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

        do_mission(self.filename)

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
    size: uint32

class LGDBChunkHeader(Struct):
    name: ByteString(12)
    version: LGDBVersion
    pad: uint32

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
        if (header.version.major, header.version.minor) not in ((0, 1), ):
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
        return self.view[entry.offset:entry.offset+entry.size]

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

def do_mission(filename):
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
