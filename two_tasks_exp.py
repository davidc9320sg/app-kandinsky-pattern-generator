from src.kp.Shapes import *
from src.kp import generate_and_save

class TwoShapes_TwoColors_Universe(KandinskyUniverse):
    """
    Universe with 2 main shapes and many colors.
    """
    def __init__(self):
        super(TwoShapes_TwoColors_Universe, self).__init__(
            kandinsky_shapes=[Square, Circle],
            kandinsky_colors=[
                'red', 'blue'
            ]
        )


class TwoTask_Setting1(KandinskyTruthInterfce):
    """
    *Modes*
    - 1: all random
    - 2:
    """
    def __init__(self, mode=1):
        self.true_kf_collection = []
        self.false_kf_collection = []
        self.min_size = 0.1
        self.max_size = 1
        self.mode = mode
        super(TwoTask_Setting1, self).__init__(TwoShapes_TwoColors_Universe(), min=1, max=2)

    def true_kf(self, n=1):
        kf_list = []
        # mode 1 creates the same number of instances for each unique elment in the cross product
        # shape x color
        if self.mode == 1:
            cross_n = len(self.u.kandinsky_shapes) * len(self.u.kandinsky_colors)
            n_per_type = n // cross_n
            for shape in self.u.kandinsky_shapes:
                for color in self.u.kandinsky_colors:
                    for i in range(n_per_type):
                        new_kf = self._generate_given_shape_and_color(shape, color)
                        assert isinstance(new_kf, KandinskyFigure)
                        kf_list.append(new_kf)
            # new_kf = self._generate_one_true_random()
        elif self.mode == 2:
            for i in range(n):
                if i < n*0.5:
                    shape = self.u.kandinsky_shapes[0]
                    color = self.u.kandinsky_colors[0]
                else:
                    shape = self.u.kandinsky_shapes[1]
                    color = self.u.kandinsky_colors[1]
                new_kf = self._generate_given_shape_and_color(shape, color)
                assert isinstance(new_kf, KandinskyFigure)
                kf_list.append(new_kf)
        self.true_kf_collection = kf_list
        return kf_list

    def _generate_one_true_random(self):
        kf = KandinskyFigure()
        # determine shape
        shape = np.random.choice(self.u.kandinsky_shapes)
        # determine color
        color = np.random.choice(self.u.kandinsky_colors)
        # randomly generate position
        new_element = self._generate_position_and_check(shape, color)
        #
        kf.add_shape(new_element)
        return kf

    def _generate_given_shape_and_color(self, shape, color, timeout=1e3):
        kf = KandinskyFigure()
        ks = self._generate_position_and_check(shape, color, timeout=timeout)
        kf.add_shape(ks)
        return kf

    def _generate_position_and_check(self, shape, color, timeout=1e3):
        good = False
        t = 0.
        new_ks = None
        while not good and t <= timeout:
            size = self.min_size + (self.max_size - self.min_size) * random.random()
            x = size / 2 + random.random() * (1 - size)
            y = size / 2 + random.random() * (1 - size)
            new_ks = shape(color, x, y, size)
            # double check & allow auto-completion
            assert isinstance(new_ks, KandinskyShape)
            good = new_ks.is_within_canvas()
            t += 1
        if t > timeout:
            raise Exception('Timeout reached')
        return new_ks


if __name__ == '__main__':
    gen = TwoTask_Setting1(mode=1)
    # gen.true_kf(10)
    # for f in gen.true_kf_collection:
    #     im = f.as_image()
    #     plt.imshow(im)
    #     plt.show()

    data_root = './output/test_mode1'
    generate_and_save(data_root, gen, 3000, 300)
    # make labels
    with open(data_root + '/data.csv', 'w') as f:
        f.write('idx,shape,color,x,y,size\n')
        for i, kf in enumerate(gen.true_kf_collection):
            s = kf.shapes[0]
            f.write('{},{},{},{},{},{}\n'.format(
                i, s.shape, s.color, s.x, s.y, s.size
            ))
    # check
    df = pd.read_csv(data_root + '/data.csv')
    details = {}
    for r, row in df.iterrows():
        sh = row['shape']
        cl = row['color']
        if sh in list(details.keys()):
            if cl in list(details[sh].keys()):
                details[sh][cl] += 1
            else:
                details[sh][cl] = 1
        else:
            details[sh] = {}
            details[sh][cl] = 1
    details_df = pd.DataFrame(details)