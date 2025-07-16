bl_info = {
    'name': 'Thieftools',
    'author': 'vfig',
    'version': (0, 0, 3),
    'blender': (2, 90, 0),
    'category': '(Development)',
    'description': '(in development)'
}

if "bpy" in locals():
    import importlib as imp
    imp.reload(brushes)
    imp.reload(mesh)
    imp.reload(lgtypes)
    imp.reload(binstruct)
    imp.reload(images)
    imp.reload(mission)
    imp.reload(prefs)
    print("thieftools: reloaded.");
else:
    from . import brushes
    from . import mesh
    from . import lgtypes
    from . import binstruct
    from . import images
    from . import mission
    from . import prefs
    print("thieftools: loaded.");


import bpy
from bpy.props import CollectionProperty, PointerProperty
from bpy.types import Panel


#---------------------------------------------------------------------------#
# Panels

class TOOLS_PT_thieftools_debug(Panel):
    bl_label = "Thief"
    bl_idname = "TOOLS_PT_thieftools_debug"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Thief"
    bl_context = "objectmode"

    @classmethod
    def poll(self, context):
        # This panel should always be available.
        return True

    def draw(self, context):
        layout = self.layout

        row = layout.row(align=True)
        box = row.box()
        box.label(text="MISSION")
        row = box.row(align=True)
        op = row.operator(mission.TTImportMISOperator.bl_idname, text="Import (debug)")
        op.dump = True
        op = box.operator(mission.TT_bake_together.bl_idname)
        op = box.operator(mission.TT_remove_lightmaps.bl_idname)

        row = layout.row(align=True)
        box = row.box()
        box.label(text="BRUSHES")
        op = box.operator("object.tt_brushes_to_booleans")
        op.filename = "e:\\dev\\thief\\dump\\brushes_skull.json"
        op = box.operator("object.tt_delete_all_brushes")

        row = layout.row(align=True)
        box = row.box()
        box.label(text="MESH")
        op = layout.operator("object.tt_debug_import_mesh")
        op.filename = r"e:/dev/thief/blender/bincal/t2/pirate.bin"
        op = layout.operator("object.tt_debug_export_mesh")
        op.filename = "e:/dev/thief/blender/thieftools/test_data/face.bin"

        row = layout.row(align=True)
        box = row.box()
        box.label(text="Images")
        op = layout.operator("object.tt_debug_import_gif")
        op.filename = "e:/dev/thief/blender/thieftools/test_data/mecserv.gif"
        op = layout.operator("object.tt_debug_import_pcx")
        op.filename = "e:/dev/thief/blender/thieftools/test_data/walfresf.pcx"


def thief_test_menu_func(self, context):
    layout = self.layout
    layout.separator()
    #layout.operator("object.tt_add_armature", icon='OUTLINER_OB_ARMATURE')
    layout.operator_menu_enum("object.tt_add_armature", "skeleton_type")
    #self.layout.menu("OBJECT_MT_effector_submenu", text="Effector")

#---------------------------------------------------------------------------#
# Register and unregister

CLASSES = [
    brushes.TTDebugBrushesToBooleansOperator,
    brushes.TTDebugDeleteAllBrushesOperator,
    mission.TTMissionSettings,
    mission.TOOLS_PT_thieftools_mission,
    mission.TTImportMISOperator,
    mission.MIS_PT_import_options,
    mission.MIS_PT_import_debug,
    mission.TT_bake_together,
    mission.TT_remove_lightmaps,
    mesh.TTDebugImportMeshOperator,
    mesh.TTDebugExportMeshOperator,
    TOOLS_PT_thieftools_debug,
    mesh.TTAddArmatureOperator,
    images.TTDebugImportGIFOperator,
    images.TTDebugImportPCXOperator,
    prefs.TTAddonPreferences,
    ]

def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(mission.menu_func_import)
    bpy.types.VIEW3D_MT_armature_add.append(thief_test_menu_func)
    bpy.types.Object.tt_mission = PointerProperty(type=mission.TTMissionSettings)
    print("thieftools: registered.");


def unregister():
    del bpy.types.Object.tt_mission
    bpy.types.VIEW3D_MT_armature_add.remove(thief_test_menu_func)
    bpy.types.TOPBAR_MT_file_import.append(mission.menu_func_import)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    print("thieftools: unregistered.");
