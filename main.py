import math
import time

import numpy as np

import taichi as ti
from taichi.lang.struct import StructType

ti.init(arch=ti.vulkan)


Vec3 = ti.types.vector(3, dtype=ti.f32)


@ti.dataclass
class Sphere:
    center: Vec3
    radius: ti.f32

    @ti.func
    def sdf(self, point: Vec3) -> ti.f32:
        return (point - self.center).norm() - self.radius



"""
def dot2(v):
    return np.dot(v, v)

def cross(u, v):
    return np.cross(u, v)

def sign(x):
    return np.sign(x)

def sdf_triangle(p, a, b, c):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = cross(ba, ac)

    cross_ba_nor = cross(ba, nor)
    cross_cb_nor = cross(cb, nor)
    cross_ac_nor = cross(ac, nor)

    sign_sum = sign(np.dot(cross_ba_nor, pa)) + sign(np.dot(cross_cb_nor, pb)) + sign(np.dot(cross_ac_nor, pc))

    if sign_sum < 2.0:
        dist1 = dot2(ba * np.clip(np.dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa)
        dist2 = dot2(cb * np.clip(np.dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)
        dist3 = dot2(ac * np.clip(np.dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc)
        return np.sqrt(min(dist1, dist2, dist3))
    else:
        return np.sqrt((np.dot(nor, pa) ** 2) / dot2(nor))
"""
@ti.dataclass
class Triangle:
    a: Vec3
    b: Vec3
    c: Vec3

    @ti.func
    def sdf(self, point: Vec3) -> ti.f32:
        ba = self.b - self.a
        pa = point - self.a
        cb = self.c - self.b
        pb = point - self.b
        ac = self.a - self.c
        pc = point - self.c
        nor = ba.cross(ac)

        cross_ba_nor = ba.cross(nor)
        cross_cb_nor = cb.cross(nor)
        cross_ac_nor = ac.cross(nor)

        sign_sum = ti.math.sign(cross_ba_nor.dot(pa)) + ti.math.sign(cross_cb_nor.dot(pb)) + ti.math.sign(cross_ac_nor.dot(pc))

        return_val = 0.0

        if sign_sum < 2.0:
            dist1 = (ba * min(max(ba.dot(pa) / ba.norm_sqr(), 0.0), 1.0) - pa).norm_sqr()
            dist2 = (cb * min(max(cb.dot(pb) / cb.norm_sqr(), 0.0), 1.0) - pb).norm_sqr()
            dist3 = (ac * min(max(ac.dot(pc) / ac.norm_sqr(), 0.0), 1.0) - pc).norm_sqr()
            return_val = ti.sqrt(min(dist1, dist2, dist3))
        else:
            return_val = ti.sqrt((nor.dot(pa) ** 2) / nor.norm_sqr())

        return return_val



spheres = Sphere.field(shape=3)
spheres[0] = Sphere(Vec3([5.0, 0.0, 0.0]), 1.0)
spheres[1] = Sphere(Vec3([0.0, 4.0, 0.0]), 1.0)
spheres[2] = Sphere(Vec3([0.0, 0.0, 3.0]), 1.0)

triangles = Triangle.field(shape=3)
triangles[0] = Triangle(Vec3([0.0, 0.0, 0.0]), Vec3([1.0, 0.0, 0.0]), Vec3([0.0, 1.0, 0.0]))
triangles[1] = Triangle(Vec3([0.0, 0.0, 0.0]), Vec3([0.0, 1.0, 0.0]), Vec3([0.0, 0.0, 1.0]))
triangles[2] = Triangle(Vec3([0.0, 0.0, 0.0]), Vec3([1.0, 0.0, 0.0]), Vec3([0.0, 0.0, 1.0]))


point = Vec3([5.0, 0.0, 0.0])

results = ti.field(dtype=ti.f32, shape=(triangles.shape[0]))
out = ti.field(ti.f32, ())



@ti.kernel
def my_kernel():
    for i in range(triangles.shape[0]):
        results[i] = triangles[i].sdf(point)



def main():
    my_kernel()
    print(results)


if __name__ == "__main__":
    main()
