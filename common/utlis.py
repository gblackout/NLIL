import os
import shutil
from os.path import join as joinpath


class EarlyStopMonitor:

    def __init__(self, patience):
        self.patience = patience
        self.cnt = 0
        self.cur_best = float('inf')

    def update(self, loss):
        """

        :param loss:
        :return:
            return True if patience exceeded
        """
        if loss < self.cur_best:
            self.cnt = 0
            self.cur_best = loss
        else:
            self.cnt += 1

        if self.cnt >= self.patience:
            return True
        else:
            return False

    def reset(self):
        self.cnt = 0
        self.cur_best = float('inf')


def makedir(_path, remove_old=False):
    if os.path.isdir(_path):
        if not remove_old:
            raise Exception('old folder exists at %s please use remove_old flag to remove' % _path)
        shutil.rmtree(_path)

    os.mkdir(_path)


def get_output_folder(parent_dir, run_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    run_name: str
      string description for the experiment which is used as name of this sub-folder
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """

    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)

    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if (not os.path.isdir(joinpath(parent_dir, folder_name))) or (run_name not in folder_name):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = joinpath(parent_dir, run_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def rmfile(_path):
    if os.path.isfile(_path):
        os.remove(_path)
    elif os.path.isdir(_path):
        raise ValueError('remove target at %s is a dir' % _path)
    else:
        raise ValueError('remove target at %s not exists' % _path)


def iterline(fpath):
    with open(fpath) as f:

        for line in f:

            line = line.strip()
            if line == '':
                continue

            yield line


def is_empty(mat):
    return min(mat.size()) == 0


def mask_select(x, mask, dim):
    shape = [x.size(0), x.size(1)]
    shape[dim] = mask.sum()
    return x.masked_select(mask).view(tuple(shape))


def flatten(l):
    return [e for sub_l in l for e in sub_l]