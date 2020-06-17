from abc import ABC, abstractmethod
from PIL import Image, ImageDraw
import math
import numpy as np
from numpy import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from src.kp.KandinskyTruth import KandinskyTruthInterfce
from src.kp.KandinskyUniverse import KandinskyUniverse
import pandas as pd


class KandinskyShape(ABC):
    def __init__(self, shape, color, x, y, size):
        self.shape = shape
        self.color = color
        self.x = x
        self.y = y
        self.size = size

    @abstractmethod
    def draw(self, d: ImageDraw.ImageDraw, w=1):
        pass

    @abstractmethod
    def get_coordinates(self, w=1):
        pass

    def get_bbox(self, w=1):
        coords = self.get_coordinates(w)
        coords = np.array(coords)
        min_coord = np.min(coords, axis=0)
        max_coord = np.max(coords, axis=0)
        return tuple(min_coord), tuple(max_coord)

    def is_within_canvas(self):
        # compute bounding box
        bbox = self.get_bbox()
        # check if its edges are outside the frame
        # minimum vertex
        for v in bbox[0]:
            if v < 0:
                return False
        # maximum vertex
        for v in bbox[1]:
            if v > 1:
                return False
        # if within return true
        return True

    def __str__(self):
        return self.color + " " + self.shape + " (" + \
               str(self.size) + "," + str(self.x) + "," + str(self.y) + ")"


class Square(KandinskyShape):
    def __init__(self, color, x, y, size):
        super(Square, self).__init__('square', color, x, y, size)

    def draw(self, d: ImageDraw.ImageDraw, w=1):
        bbox = self.get_coordinates(w)
        d.rectangle(bbox, fill=self.color)

    def get_coordinates(self, w=1):
        s = self.size * w
        s *= 0.6
        cx = self.x * w
        cy = self.y * w
        x0 = cx - s / 2
        y0 = cy - s / 2
        x1 = cx + s / 2
        y1 = cy + s / 2
        return (x0, y0), (x1, y1)


class Circle(KandinskyShape):
    def __init__(self, color, x, y, size):
        super(Circle, self).__init__('circle', color, x, y, size)

    def draw(self, d: ImageDraw.ImageDraw, w=1):
        bbox = self.get_bbox(w)
        d.ellipse(bbox, fill=self.color)

    def get_coordinates(self, w=1):
        s = self.size * w
        cx = self.x * w
        cy = self.y * w
        # correct the size to  the same area as an square
        s = 0.6 * math.sqrt(4 * s * s / math.pi)
        return (cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)


class Triangle(KandinskyShape):
    def __init__(self, color, x, y, size):
        self.r = math.radians(30)
        super(Triangle, self).__init__('triangle', color, x, y, size)

    def draw(self, d: ImageDraw.ImageDraw, w=1):
        coords = self.get_coordinates(w)
        d.polygon(coords, fill=self.color)

    def get_coordinates(self, w=1):
        s = self.size * w
        cx = self.x * w
        cy = self.y * w
        s = 0.6 * math.sqrt(4 * s * s / math.sqrt(3))
        s = math.sqrt(3) * s / 3
        dx = s * math.cos(self.r)
        dy = s * math.sin(self.r)
        return (cx, cy - s), (cx + dx, cy + dy), (cx - dx, cy + dy)


class KandinskyFigure:
    def __init__(self, shapes: list = None):
        if shapes is None:
            self.shapes = []
        else:
            for x in shapes:
                assert isinstance(x, KandinskyShape), "All elements in shapes should be subclass of KandinskyShape."
            self.shapes = shapes

    def add_shape(self, shape: KandinskyShape):
        assert isinstance(shape, KandinskyShape), "All elements in shapes should be subclass of KandinskyShape."
        self.shapes.append(shape)

    def overlap(self, test_width=1024):
        overlap_fill = 10
        image = Image.new("L", (test_width, test_width), 0)
        sumarray = np.array(image)
        w = test_width
        for s in self.shapes:
            # save color
            tmp_color = s.color
            s.color = overlap_fill
            # create new image
            image = Image.new("L", (test_width, test_width), 0)
            # draw shape on image
            d = ImageDraw.Draw(image)
            s.draw(d, w)
            sumarray = sumarray + np.array(image)
            # return color to original state
            s.color = tmp_color

        # sumimage = Image.fromarray(sumarray)
        # return sumimage.getextrema()[1] > 10
        return np.max(sumarray) > overlap_fill

    def as_image(self, width=600, subsampling=4):
        assert self.shapes is not None, "No shapes in the figure."
        image = Image.new("RGBA", (subsampling * width, subsampling * width), (150, 150, 150, 255))
        d = ImageDraw.Draw(image)
        w = subsampling * width
        for s in self.shapes:
            s.draw(d, w)
        if subsampling > 1:
            image.thumbnail((width, width), Image.ANTIALIAS)
        return image


class MyUniverse(KandinskyUniverse):
    """
    Universe with 3 main shapes and many colors.
    """
    def __init__(self):
        super(MyUniverse, self).__init__(
            kandinsky_shapes=[Square, Circle],
            kandinsky_colors=[
                'red', 'blue', 'yellow', 'orchid', 'coral', 'aqua', 'gold', 'pink',
                'green', 'brown', 'tomato', 'orange', 'lime', 'cyan', 'crimson',
                'LightGreen', 'LightBlue', 'Teal', 'Indigo', 'Khaki'
            ]
        )


class MyTruth(KandinskyTruthInterfce):
    def __init__(self):
        self.true_kf_collection = []
        self.false_kf_collection = []
        self.min_size = 0.1
        self.max_size = 1
        super(MyTruth, self).__init__(MyUniverse(), min=1, max=2)

    def true_kf(self, n=1):
        kf_list = []
        for i in range(n):
            new_kf = self._generate_one_true()
            assert isinstance(new_kf, KandinskyFigure)
            kf_list.append(new_kf)
        self.true_kf_collection = kf_list
        return kf_list

    def _generate_one_true(self):
        kf = KandinskyFigure()
        good = False
        while not good:
            shape = np.random.choice(self.u.kandinsky_shapes)
            color = np.random.choice(self.u.kandinsky_colors)
            size = self.min_size + (self.max_size - self.min_size) * random.random()
            x = size / 2 + random.random() * (1 - size)
            y = size / 2 + random.random() * (1 - size)
            new_ks = shape(color, x, y, size)
            # double check & allow auto-completion
            assert isinstance(new_ks, KandinskyShape)
            good = new_ks.is_within_canvas()
            if good:
                kf.add_shape(new_ks)
        return kf


def generate_and_save(basedir, generator: KandinskyTruthInterfce, n=50,  width=200, subsampling=4):
    # make directories
    true_dir = os.path.join(basedir, 'true')
    false_dir = os.path.join(basedir, 'false')
    os.makedirs(true_dir, exist_ok=True)
    os.makedirs(false_dir, exist_ok=True)
    # true ----
    kf_list = generator.true_kf(n)
    for i, kf in tqdm(enumerate(kf_list)):
        assert isinstance(kf, KandinskyFigure)
        im = kf.as_image(width=width, subsampling=subsampling)
        im.save(true_dir+'/{:06d}.png'.format(i))


if __name__ == '__main__':
    sq = Square('orchid', 0.1, 0.5, 1)
    sq2 = Square('coral', 0.7, 0.3, 0.1)
    ci = Circle('aqua', 0.9, 0.5, 0.5)
    tri = Triangle('gold', 0.3, 0.9, 0.15)
    shapes = [sq, ci, sq2, tri]
    f = KandinskyFigure([sq, ci, sq2, tri])
    print('overlap', f.overlap())
    for s in shapes:
        print('{} within = {}'.format(s.shape, s.is_within_canvas()))
    im = f.as_image()
    plt.imshow(im)
    plt.show()
    # universe
    gen = MyTruth()
    gen.true_kf(10)
    for f in gen.true_kf_collection:
        im = f.as_image()
        plt.imshow(im)
        plt.show()

    generate_and_save('./test/test/', gen, 1000, 600)
    # make labels
    with open('./test/test/data.csv', 'w') as f:
        f.write('idx,shape,color,x,y,size\n')
        for i, kf in enumerate(gen.true_kf_collection):
            s = kf.shapes[0]
            f.write('{},{},{},{},{},{}\n'.format(
                i, s.shape, s.color, s.x, s.y, s.size
            ))
    # check
    df = pd.read_csv('./test/test/data.csv')