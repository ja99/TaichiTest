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


spheres = Sphere.field(shape=3)
spheres[0] = Sphere(Vec3([5.0, 0.0, 0.0]), 1.0)
spheres[1] = Sphere(Vec3([0.0, 5.0, 0.0]), 1.0)
spheres[2] = Sphere(Vec3([0.0, 0.0, 5.0]), 1.0)


point = Vec3([0.0, 0.0, 0.0])

results = ti.field(dtype=ti.f32, shape=(spheres.shape[0]))
out = ti.field(ti.f32, 1)



@ti.kernel
def my_kernel():
    for i in range(spheres.shape[0]):
        results[i] = spheres[i].sdf(point)



def main():
    my_kernel()
    print(results)


if __name__ == "__main__":
    main()
