import bpy
import bmesh, bmesh.ops
import mathutils
import json

from bpy.props import StringProperty
from bpy.types import Operator
from mathutils import Vector

CUBE_VERTICES = [
    Vector((-1,1,-1)),
    Vector((-1,1,1)),
    Vector((-1,-1,1)),
    Vector((-1,-1,-1)),
    Vector((1,1,-1)),
    Vector((1,1,1)),
    Vector((1,-1,1)),
    Vector((1,-1,-1)),
    ]

CUBE_EDGES = [
    (0,1), (1,2), (2,3), (3,0),
    (0,4), (4,7), (7,3), (2,6),
    (6,5), (5,1), (4,5), (6,7),
    ]

CUBE_FACES = [
    (0,1,2,3),
    (3,2,6,7),
    (7,6,5,4),
    (4,5,1,0),
    (1,5,6,2),
    (4,0,3,7),
    ]

def make_cube(mesh, param):
    mesh.from_pydata(CUBE_VERTICES, CUBE_EDGES, CUBE_FACES)
    mesh.validate(verbose=True)

WEDGE_VERTICES = [
    Vector((1,-1,1)),
    Vector((1,1,-1)),
    Vector((1,-1,-1)),
    Vector((-1,-1,1)),
    Vector((-1,1,-1)),
    Vector((-1,-1,-1)),
    ]

WEDGE_EDGES = [
    (0,1), (1,2), (2,0),
    (3,4), (4,5), (5,3),
    (0,3), (2,5), (1,4),
    ]

WEDGE_FACES = [
    (0,3,4,1),
    (2,1,4,5),
    (0,2,5,3),
    (0,1,2),
    (3,5,4),
    ]

def make_wedge(mesh, param):
    mesh.from_pydata(WEDGE_VERTICES, WEDGE_EDGES, WEDGE_FACES)
    mesh.validate(verbose=True)

DODECAHEDRON_VERTICES = [
    Vector((0.5773502692,0.1875924741,-0.7946544723)),
    Vector((0.0,0.6070619982,-0.7946544723)),
    Vector((-0.5773502692,0.1875924741,-0.7946544723)),
    Vector((-0.3568220898,-0.4911234732,-0.7946544723)),
    Vector((0.3568220898,-0.4911234732,-0.7946544723)),
    Vector((0.9341723590,0.3035309991,-0.1875924741)),
    Vector((0.0,0.9822469464,-0.1875924741)),
    Vector((-0.9341723590,0.3035309991,-0.1875924741)),
    Vector((-0.5773502692,-0.7946544723,-0.1875924741)),
    Vector((0.5773502692,-0.7946544723,-0.1875924741)),
    Vector((0.5773502692,-0.1875924741,0.7946544723)),
    Vector((0.0,-0.6070619982,0.7946544723)),
    Vector((-0.5773502692,-0.1875924741,0.7946544723)),
    Vector((-0.3568220898,0.4911234732,0.7946544723)),
    Vector((0.3568220898,0.4911234732,0.7946544723)),
    Vector((0.9341723590,-0.3035309991,0.1875924741)),
    Vector((0.0,-0.9822469464,0.1875924741)),
    Vector((-0.9341723590,-0.3035309991,0.1875924741)),
    Vector((-0.5773502692,0.7946544723,0.1875924741)),
    Vector((0.5773502692,0.7946544723,0.1875924741)),
    ]

DODECAHEDRON_EDGES = [
    (0,1), (1,2), (2,3), (3,4), (4,0),
    (4,9), (9,15), (15,5), (5,0), (5,19),
    (19,6), (6,1), (6,18), (18,7), (7,2),
    (7,17), (17,8), (8,3), (8,16), (16,9),
    (10,11), (11,12), (12,13), (13,14), (14,10),
    (14,19), (10,15), (11,16), (17,12), (13,18),
    ]

DODECAHEDRON_FACES = [
    (0,1,2,3,4),
    (0,4,9,15,5),
    (1,0,5,19,6),
    (2,1,6,18,7),
    (3,2,7,17,8),
    (4,3,8,16,9),
    (10,11,12,13,14),
    (10,14,19,5,15),
    (11,10,15,9,16),
    (12,11,16,8,17),
    (13,12,17,7,18),
    (14,13,18,6,19),
    ]

def make_dodecahedron(mesh, param):
    mesh.from_pydata(DODECAHEDRON_VERTICES, DODECAHEDRON_EDGES, DODECAHEDRON_FACES)
    mesh.validate(verbose=True)

def ngon_vertices(n, z, face_align):
    from math import pi, sin, cos
    face_mod = 1.0 if face_align else 0.0
    scale_f = 1.0
    pts = []
    for i in range(n):
        ang = 2*pi*(i*2.0+face_mod)/(n*2.0);
        if face_align and (i==0):
            scale_f = 1.0/cos(ang);
        pts.append(Vector((
            -sin(ang)*scale_f,
            cos(ang)*scale_f,
            z)))
    return pts

def ngon_edges(n, base_idx):
    edges = []
    for i in range(n-1):
        edges.append((base_idx+i,base_idx+i+1))
    edges.append((base_idx+n-1,base_idx))
    return edges

def make_ngon_cylinder(n, face_align):
    vertices = []
    vertices.extend(ngon_vertices(n, -1, face_align))
    vertices.extend(ngon_vertices(n, 1, face_align))
    edges = []
    edges.extend(ngon_edges(n,0))
    edges.extend(ngon_edges(n,n))
    for i in range(n):
        edges.append((i,n+i))
    faces = []
    for i in range(n):
        faces.append((i, n+i, n+((i+1)%n), (i+1)%n))
    faces.append(tuple([n] + [2*n-i for i in range(1,n)]))
    faces.append(tuple(i for i in range(n)))
    return (vertices, edges, faces)

def make_vertex_cylinder(mesh, param):
    vertices, edges, faces = make_ngon_cylinder(3+param, False)
    mesh.from_pydata(vertices, edges, faces)
    mesh.validate(verbose=True)

def make_edge_cylinder(mesh, param):
    vertices, edges, faces = make_ngon_cylinder(3+param, True)
    mesh.from_pydata(vertices, edges, faces)
    mesh.validate(verbose=True)

def make_ngon_pyramid(n, corner_pyramid, face_align):
    vertices = []
    vertices.extend(ngon_vertices(n, -1, face_align))
    if corner_pyramid:
        vertices.append(Vector((vertices[0].x,vertices[0].y,1.0)))
    else:
        vertices.append(Vector((0,0,1.0)))
    edges = []
    edges.extend(ngon_edges(n,0))
    for i in range(n):
        edges.append((i,n))
    faces = []
    for i in range(n):
        faces.append((i,n,(i+1)%n))
    faces.append(tuple(i for i in range(n)))
    return (vertices, edges, faces)

def make_vertex_pyramid(mesh, param):
    vertices, edges, faces = make_ngon_pyramid(3+param, False, False)
    mesh.from_pydata(vertices, edges, faces)
    mesh.validate(verbose=True)

def make_edge_pyramid(mesh, param):
    vertices, edges, faces = make_ngon_pyramid(3+param, False, True)
    mesh.from_pydata(vertices, edges, faces)
    mesh.validate(verbose=True)

def make_vertex_corner_pyramid(mesh, param):
    vertices, edges, faces = make_ngon_pyramid(3+param, True, False)
    mesh.from_pydata(vertices, edges, faces)
    mesh.validate(verbose=True)

def make_edge_corner_pyramid(mesh, param):
    vertices, edges, faces = make_ngon_pyramid(3+param, True, True)
    mesh.from_pydata(vertices, edges, faces)
    mesh.validate(verbose=True)

def decode_shape(raw):
    shape_category = (raw & 0xFF00) >> 8
    shape_param = (raw & 0x00FF)
    if shape_category==0:
        fn = {
            1: make_cube,
            6: make_dodecahedron,
            7: make_wedge,
            }.get(shape_param)
        param = 0
    else:
        fn = {
            2: make_vertex_cylinder,
            3: make_edge_cylinder,
            4: make_vertex_pyramid,
            5: make_edge_pyramid,
            6: make_vertex_corner_pyramid,
            7: make_edge_corner_pyramid,
            }.get(shape_category)
        param = shape_param
    return (fn, param)

BRUSH_OP_SOLID = 0
BRUSH_OP_AIR = 1
BRUSH_OP_WATER = 2
BRUSH_OP_AIR2WATER = 3
BRUSH_OP_FLOOD = BRUSH_OP_AIR2WATER
BRUSH_OP_WATER2AIR = 4
BRUSH_OP_EVAPORATE = BRUSH_OP_WATER2AIR
BRUSH_OP_SOLID2WATER = 5
BRUSH_OP_SOLID2AIR = 6
BRUSH_OP_AIR2SOLID = 7
BRUSH_OP_WATER2SOLID = 8
BRUSH_OP_BLOCKABLE = 9

BRUSH_COLOR_WATER = [0.197,0.321,1.0,1.0]
BRUSH_COLOR_AIR = [0.319,0.570,0.301,1.0]
BRUSH_COLOR_EARTH = [0.196,0.146,0.115,1.0]
BRUSH_COLOR_FIRE = [1.0,0.047,0.052,1.0]
BRUSH_COLORS = {
    BRUSH_OP_SOLID: BRUSH_COLOR_EARTH,
    BRUSH_OP_AIR: BRUSH_COLOR_AIR,
    BRUSH_OP_WATER: BRUSH_COLOR_WATER,
    BRUSH_OP_AIR2WATER: BRUSH_COLOR_WATER,
    BRUSH_OP_WATER2AIR: BRUSH_COLOR_AIR,
    BRUSH_OP_SOLID2WATER: BRUSH_COLOR_WATER,
    BRUSH_OP_SOLID2AIR: BRUSH_COLOR_AIR,
    BRUSH_OP_AIR2SOLID: BRUSH_COLOR_EARTH,
    BRUSH_OP_WATER2SOLID: BRUSH_COLOR_EARTH,
    BRUSH_OP_BLOCKABLE: BRUSH_COLOR_FIRE,
    }

def create_brush(context, brush_json):
    id = brush_json['id']
    raw_shape = brush_json['shape']
    op = brush_json['type']
    position = Vector(brush_json['position']);
    facing = Vector(brush_json['facing']);
    size = Vector(brush_json['size']);
    name = f"Brush {id} shape"
    fn, param = decode_shape(raw_shape)
    if fn:
        mesh = bpy.data.meshes.new(f"{name} mesh")
        fn(mesh, param);
    else:
        raise RuntimeError("Unknown brush type!")
    o = bpy.data.objects.new(name, mesh)
    o.location = position
    o.rotation_mode = 'XYZ'
    o.rotation_euler = facing
    o.scale = size
    o.color = BRUSH_COLORS.get(op, [1.0,1.0,1.0,1.0])
    is_air_brush = (op in (BRUSH_OP_AIR, BRUSH_OP_WATER2AIR, BRUSH_OP_SOLID2AIR))
    o.display_type = 'WIRE' if is_air_brush else 'TEXTURED'
    context.scene.collection.objects.link(o)
    return o

def create_brushes(filename):
    with open(filename) as f:
        raw_data = f.read()
        json_data = json.loads(raw_data)
    brushes = []
    for brush_json in json_data:
        create_brush(bpy.context, brush_json)

    # TODO:
    #     4. start doing boolean ops!

def delete_all_brushes():
    to_delete = [o for o in bpy.data.objects
        if o.name.startswith("Brush ")]
    for o in to_delete:
        bpy.data.objects.remove(o)

class TTDebugDeleteAllBrushesOperator(Operator):
    bl_idname = "object.tt_delete_all_brushes"
    bl_label = "Delete all brushes"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return (context.mode == "OBJECT")

    def execute(self, context):
        if context.mode != "OBJECT":
            self.report({'WARNING'}, f"{self.bl_label}: must be in Object mode.")
            return {'CANCELLED'}

        delete_all_brushes()
        return {'FINISHED'}

class TTDebugBrushesToBooleansOperator(Operator):
    bl_idname = "object.tt_brushes_to_booleans"
    bl_label = "Import brushes"
    bl_options = {'REGISTER'}

    filename : StringProperty()

    @classmethod
    def poll(cls, context):
        return (context.mode == "OBJECT")

    def execute(self, context):
        if context.mode != "OBJECT":
            self.report({'WARNING'}, f"{self.bl_label}: must be in Object mode.")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        print(f"filename: {self.filename}")
        create_brushes(self.filename)
        return {'FINISHED'}
