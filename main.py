import math
import time

import numpy as np

import taichi as ti
from taichi.lang.struct import StructType

ti.init(arch=ti.vulkan)

Vec3 = ti.types.vector(3, dtype=ti.f32)



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
