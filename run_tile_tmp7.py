from multiprocessing import Pool
import copy

from parse import Parser
from tileSampler import TileSampler


if __name__ == '__main__':
    out_dir = 'out/scene7_4/'
    testfile = r'out/scene7.test'

    origin = 96

    tile_width = 224 // 8
    tile_height = 6
    row = 1
    col = 8

    left_start = origin + 0 * tile_width
    # top_start = origin + 6 * tile_height
    top_start = origin + 70 + 4

    scenes = []
    tasks = []

    for left in range(left_start, left_start + col * tile_width, tile_width):
        for top in range(top_start, top_start + row * tile_height, tile_height):
            parser = Parser()
            s = parser.parse(testfile)
            scenes.append(s)

            scenes[-1].sampler = TileSampler(left, top, tile_width, tile_height)
            name = out_dir + '%d.%d.png' % (left, top)
            scenes[-1].film = copy.deepcopy(scenes[-1].film)
            scenes[-1].film.filename = name
            tasks.append(scenes[-1].render)

    p = Pool(len(tasks))
    for s in scenes:
        print(s.film.filename)
        print('%d, %d' % (s.sampler.left, s.sampler.top))

    for task in tasks:
        p.apply_async(task)

    p.close()
    p.join()
