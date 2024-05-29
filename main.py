import math
import time

import numpy as np

import taichi as ti

ti.init(arch=ti.vulkan)

spheres_np = np.array(
    [
        [2.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 7.0, 0.5]
    ],
    dtype=np.float32,
)  # n x (x, y, z, radius)

spheres = ti.field(dtype=ti.f32, shape=spheres_np.shape)  # n x (x, y, z, radius)
spheres.from_numpy(spheres_np)

point = ti.Vector([0.0, 0.0, 0.0])

out = ti.field(dtype=ti.f32, shape=(spheres.shape[0]))


@ti.func
def sdf_circle(point_pos, center_pos, radius: float):
    return (point_pos - center_pos).norm() - radius


@ti.kernel
def my_kernel():
    for i in range(spheres.shape[0]):
        out[i] = sdf_circle(point, ti.Vector([spheres[i, 0], spheres[i, 1], spheres[i, 2]]), spheres[i, 3])


def main():
    my_kernel()
    print(out.to_numpy())


if __name__ == "__main__":
    main()
