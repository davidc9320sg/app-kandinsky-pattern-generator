print(f'Invoking __init__.py for {__name__}')
import os
from tqdm import tqdm
from .KandinskyTruth import KandinskyTruthInterfce
from .Shapes import KandinskyFigure


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