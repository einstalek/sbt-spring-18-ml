# coding=utf-8

from sklearn.model_selection import cross_val_score
import numpy as np
import os
from gb_impl_arganaidi import SimpleGB
import signal
import pandas
import traceback


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self, data_path=SCRIPT_DIR + '/HR.csv'):
        df = pandas.read_csv(data_path)
        target = 'left'
        features = [c for c in df if c != target]
        self.target = np.array(df[target])
        self.target = np.array([-1 if x == 0 else 1 for x in self.target])
        self.data = np.array(df[features])
        self.application = 0

    def check(self, script_path):
        try:
            signal.signal(signal.SIGALRM, signal_handler)
            # Time limit на эту задачу 1 минута
            signal.alarm(60)
            self.application += 1
            algo = SimpleGB(
                tree_params_dict={'max_depth': 2},
                iters=100,
                tau=0.6
            ).fit(self.data, self.target)
            return np.mean(cross_val_score(algo, self.data, self.target, cv=3, scoring='accuracy'))
        except:
            traceback.print_exc()
            return None


if __name__ == '__main__':
    print(Checker().check(SCRIPT_DIR + '/gb_impl_arganaidi.py'))