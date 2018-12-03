from film import Film
from tileSampler import TileSampler
from raytracer import RayTracer
from camera import Camera


class Scene:

    def __init__(self):
        self.film = Film()
        self.sampler = TileSampler()
        self.ray_tracer = RayTracer()
        self.camera = Camera()

    def render(self):
        width = self.film.image.shape[1]
        height = self.film.image.shape[0]
        for sample in self.sampler.get_sample():
            ray = self.camera.generate_ray(sample, width, height)
            color = self.ray_tracer.trace(ray, 0)
            self.film.commit(sample, color)

            if int(sample[1]) % 50 == 0:
                print('%s: (%d, %d)' % (self.film.filename, sample[0], sample[1]))

        self.film.write_image()
