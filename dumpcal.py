import itertools
import math
import os, os.path
import random
import struct
import zlib

from binstruct import *
from lgtypes import *

def vec_add(a,b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def vec_mulf(f,a):
    return (a[0]*f, a[1]*f, a[2]*f)

def vec_fmt(a):
    return f"({a[0]:0.6f},{a[1]:0.6f},{a[2]:0.6f})"

def dump_cal(cal_filename):
    with open(cal_filename, 'rb') as f:
        cal_data = f.read()
    #dump_filename = os.path.splitext(cal_filename)[0]+'.dump'
    #dumpf = open(dump_filename, 'w')
    import sys
    dumpf = sys.stdout

    # Parse the .cal file
    cal_view = memoryview(cal_data)
    offset = 0
    cal_header = LGCALHeader.read(cal_view, offset=offset)
    if cal_header.version not in (1,):
        raise ValueError("Only version 1 .cal files are supported")
    offset += LGCALHeader.size()
    p_torsos = StructView(cal_view, LGCALTorso, offset=offset, count=cal_header.torsos)
    offset += p_torsos.size()
    p_limbs = StructView(cal_view, LGCALLimb, offset=offset, count=cal_header.limbs)
    offset += p_limbs.size()
    cal_footer = LGCALFooter.read(cal_view, offset=offset)

    print("CAL:", file=dumpf)
    print(f"  version: {cal_header.version}", file=dumpf)
    print(f"  torsos: {cal_header.torsos}", file=dumpf)
    print(f"  limbs: {cal_header.limbs}", file=dumpf)
    for i, torso in enumerate(p_torsos):
        print(f"torso {i}:", file=dumpf)
        print(f"  joint: {torso.joint}", file=dumpf)
        print(f"  parent: {torso.parent}", file=dumpf)
        print(f"  fixed_points: {torso.fixed_points}", file=dumpf)
        print(f"  joint_id:", file=dumpf)
        k = torso.fixed_points
        for joint_id in torso.joint_id[:k]:
            print(f"    {joint_id}", file=dumpf)
        print(f"  pts:", file=dumpf)
        for pt in torso.pts[:k]:
            print(f"    {pt.x}, {pt.y}, {pt.z}", file=dumpf)
    for i, limb in enumerate(p_limbs):
        print(f"limb {i}:", file=dumpf)
        print(f"  torso_id: {limb.torso_id}", file=dumpf)
        print(f"  bend: {limb.bend}", file=dumpf)
        print(f"  segments: {limb.segments}", file=dumpf)
        print(f"  joint_id:", file=dumpf)
        k = limb.segments
        for joint_id in limb.joint_id[:k+1]:
            print(f"    {joint_id}", file=dumpf)
        print(f"  seg:", file=dumpf)
        for seg in limb.seg[:k]:
            print(f"    {seg}")
        print(f"  seg_len:", file=dumpf)
        for seg_len in limb.seg_len[:k]:
            print(f"    {seg_len}", file=dumpf)
    print(f"scale: {cal_footer.scale}", file=dumpf)
    print(file=dumpf)

    #dumpf.close()

    # Find the origin point of every joint
    head_by_joint_id = {}
    tail_by_joint_id = {}
    parent_by_joint_id = {}
    is_connected_by_joint_id = {}
    is_limb_end_by_joint_id = {}
    for torso in p_torsos:
        if torso.parent == -1:
            j = torso.joint
            # Some skeletons (burrick, spider, ...) have multiple torsos defined
            # with the same joint id (presumably from before they increased the
            # torso's fixed-point maximum to 16). When we encounter those, we
            # just ignore that; the rest of the skeleton should continue to
            # import correctly.
            if j not in head_by_joint_id:
                assert j not in head_by_joint_id, f"joint {j} already in head list!"
                assert j not in tail_by_joint_id, f"joint {j} already in tail list!"
                head_by_joint_id[j] = (0,0,0)
                tail_by_joint_id[j] = (1,0,0)
                assert j not in parent_by_joint_id, f"joint {j} already in parent list!"
                parent_by_joint_id[j] = -1
        else:
            j = torso.joint
            assert j in head_by_joint_id, f"joint {j} not found in head list!"
            assert j not in tail_by_joint_id, f"joint {j} already in tail list!"
            tail_by_joint_id[j] = vec_add(head_by_joint_id[j], (1,0,0))
            assert j not in parent_by_joint_id, f"joint {j} already in parent list!"
            parent_by_joint_id[j] = p_torsos[torso.parent].joint
        is_connected_by_joint_id[j] = False
        is_limb_end_by_joint_id[j] = False
        root = head_by_joint_id[torso.joint]
        k = torso.fixed_points
        parts = zip(
            torso.joint_id[:k],
            torso.pts[:k])
        for j, pt in parts:
            assert j not in head_by_joint_id, f"joint {j} already in head list!"
            head_by_joint_id[j] = vec_add(root, pt)

    for limb in p_limbs:
        j = limb.joint_id[0]
        assert j in head_by_joint_id, f"joint {j} not found in head list!"
        head = head_by_joint_id[j]
        k = limb.segments
        parts = zip(
            limb.joint_id[:k+1],
            limb.seg[:k] + [limb.seg[k-1]], # finger etc. bones, tail gets wrist vector
            limb.seg_len[:k] + [0.25])      # finger etc. bones, get fixed length
        pj = j
        for i, (j, seg, seg_len) in enumerate(parts):
            head_by_joint_id[j] = head
            tail = vec_add(head, vec_mulf(seg_len, seg))
            tail_by_joint_id[j] = tail
            head = tail
            assert j not in parent_by_joint_id, f"joint {j} already in parent list!"
            if i==0:
                parent_by_joint_id[j] = p_torsos[limb.torso_id].joint
                is_connected_by_joint_id[j] = False
            else:
                parent_by_joint_id[j] = pj
                is_connected_by_joint_id[j] = True
            pj = j

    parent_joint_ids = set(parent_by_joint_id.values())
    for j in head_by_joint_id.keys():
        is_limb_end_by_joint_id[j] = (j not in parent_joint_ids)

    # some .cals have torso fixed points without limb definitions. (Apparition
    # 'toe' joint is like this). So add in default values for each.
    for torso in p_torsos:
        k = torso.fixed_points
        for j in torso.joint_id[:k]:
            if j not in tail_by_joint_id:
                head = head_by_joint_id[j]
                tail_by_joint_id[j] = vec_add(head, (1,0,0))
                parent_by_joint_id[j] = torso.joint
                is_connected_by_joint_id[j] = False
                is_limb_end_by_joint_id[j] = True

    joint_ids = sorted(head_by_joint_id.keys())
    assert sorted(tail_by_joint_id.keys())==joint_ids
    assert sorted(parent_by_joint_id.keys())==joint_ids
    assert sorted(is_connected_by_joint_id.keys())==joint_ids
    assert sorted(is_limb_end_by_joint_id.keys())==joint_ids

    bones_by_joint_id = {}
    for j in sorted(head_by_joint_id.keys()):
        # joint_id, name, parent, connected, head_pos, tail_pos, limb_end)
        print(
            f"({j:02d}, 'xxnamexx', "
            f"{parent_by_joint_id[j]}, "
            f"{is_connected_by_joint_id[j]}, "
            f"{vec_fmt(head_by_joint_id[j])}, "
            f"{vec_fmt(tail_by_joint_id[j])}, "
            f"{is_limb_end_by_joint_id[j]}),")

if __name__=='__main__':
    import sys
    cal_filename = sys.argv[1]
    dump_cal(cal_filename)