import taichi as ti

ti.init(arch=ti.vulkan)
ti.init(default_ip=ti.i32)  # Sets the default integer type to ti.i32
ti.init(default_fp=ti.f32)  # Sets the default floating-point type to ti.f32

Vec3 = ti.types.vector(3, dtype=ti.f32)
INF = ti.field(dtype=ti.f32, shape=())
INF[None] = 1e10
EPS = ti.field(dtype=ti.f32, shape=())
EPS[None] = 1e-4
MAX_STEPS = ti.field(dtype=ti.i32, shape=())
MAX_STEPS[None] = 100


@ti.dataclass
class Pair:
    a: Vec3
    b: Vec3


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

        sign_sum = ti.math.sign(cross_ba_nor.dot(pa)) + ti.math.sign(cross_cb_nor.dot(pb)) + ti.math.sign(
            cross_ac_nor.dot(pc))

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

pairs = Pair.field(shape=3)
pairs[0] = Pair(Vec3([0.0, 0.0, 10.0]), Vec3([1.0, 0.0, 10.0]))
pairs[1] = Pair(Vec3([0.0, 0.0, 0.0]), Vec3([0.0, 1.0, 0.0]))
pairs[2] = Pair(Vec3([0.0, 0.0, 0.0]), Vec3([0.0, 0.0, 1.0]))

results = ti.field(dtype=ti.u1, shape=(pairs.shape[0]))


@ti.kernel
def ray_march():
    for pair in range(pairs.shape[0]):
        start_pos = pairs[pair].a
        dir = (pairs[pair].b - pairs[pair].a).norm()
        pos = start_pos
        hit = False

        for step in range(MAX_STEPS[None]):
            dist = INF[None]
            for j in range(triangles.shape[0]):
                dist = min(dist, triangles[j].sdf(pos))
            pos += dir * dist
            if dist < EPS[None]:
                hit = True
                break

        results[pair] = hit


def main():
    ray_march()
    print(results)


if __name__ == "__main__":
    main()
