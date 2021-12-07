import bpy
from bpy.types import Operator, AddonPreferences
from bpy.props import StringProperty, IntProperty, BoolProperty

class TTAddonPreferences(AddonPreferences):
    bl_idname = __package__

    # I don't know how to do an array of directory paths, so just have a
    # bunch of individual properties. I hope five is enough for now.
    resource_path_0: StringProperty(name="Resource Path 0", subtype='DIR_PATH')
    resource_path_1: StringProperty(name="Resource Path 1", subtype='DIR_PATH')
    resource_path_2: StringProperty(name="Resource Path 2", subtype='DIR_PATH')
    resource_path_3: StringProperty(name="Resource Path 3", subtype='DIR_PATH')
    resource_path_4: StringProperty(name="Resource Path 4", subtype='DIR_PATH')
    # The last file we imported (so that we can open to the same directory)
    last_filepath: StringProperty(name="Last file", subtype='FILE_PATH')

    def draw(self, context):
        layout = self.layout
        layout.label(text="Resource search paths:")
        layout.prop(self, "resource_path_0")
        layout.prop(self, "resource_path_1")
        layout.prop(self, "resource_path_2")
        layout.prop(self, "resource_path_3")
        layout.prop(self, "resource_path_4")

    def texture_paths(self):
        paths = [
            self.resource_path_0,
            self.resource_path_1,
            self.resource_path_2,
            self.resource_path_3,
            self.resource_path_4,
            ]
        return [p for p in paths if p]

def show_preferences():
    bpy.ops.preferences.addon_show(module=__package__)

def get_preferences(context):
    preferences = context.preferences
    return preferences.addons[__package__].preferences
