bl_info = {
    'name': 'Thieftools',
    'author': 'vfig',
    'version': (0, 0, 1),
    'blender': (2, 90, 0),
    'category': '(Development)',
    'description': '(in development)'
}

if "bpy" in locals():
    import importlib as imp
    imp.reload(mission)
    print("thieftools: reloaded.");
else:
    from . import mission
    print("thieftools: loaded.");


import bpy
from bpy.props import CollectionProperty
from bpy.types import Panel


#---------------------------------------------------------------------------#
# Panels

class TOOLS_PT_thieftools_debug(Panel):
    bl_label = "Thief"
    bl_idname = "TOOLS_PT_thieftools_debug"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mission"
    bl_context = "objectmode"

    @classmethod
    def poll(self, context):
        # This panel should always be available.
        return True

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        box = row.box()
        box.label(text="Foo")
        row = box.row(align=True)
        op = row.operator("object.tt_debug_import_mission", text="Import (debug)")
        op.filename = "e:\\dev\\thief\\blender\\data\\miss1.mis"


#---------------------------------------------------------------------------#
# Register and unregister

def register():
    bpy.utils.register_class(mission.TTDebugImportMissionOperator)
    bpy.utils.register_class(TOOLS_PT_thieftools_debug)
    print("thieftools: registered.");


def unregister():
    bpy.utils.unregister_class(TOOLS_PT_thieftools_debug)
    bpy.utils.unregister_class(mission.DebugImportMissionOperator)
    print("thieftools: unregistered.");
