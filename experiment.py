import datetime
from typing import Callable
from multiprocessing import Process
import sys
import pathlib, jsonpickle
import tensorflow as tf
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
from baselines import logger
from singleton_decorator import singleton


_CURRENT = None


@singleton
class Experiment:
    def __init__(self, name: str, log_folder: str = 'logs'):
        global _CURRENT
        self.logger_formats = ['stdout', 'log', 'csv', 'tensorboard']
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_folder = osp.join(log_folder, "%s-%s" % (timestamp, name))
        pathlib.Path(self.log_folder).mkdir(parents=True, exist_ok=True)

        self.run_json = {
            "time": timestamp,
            "name": name,
            "settings": {}
        }
        if len(sys.argv) > 0:
            self.run_json["src_files"] = {
                osp.basename(sys.argv[0]): "".join(open(sys.argv[0], "r"))
            }
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            self.run_json["git"] = {
                "head": repo.head.object.hexsha,
                "branch": str(repo.active_branch),
                "summary": repo.head.object.summary,
                "time": repo.head.object.committed_datetime.strftime("%Y-%m-%d-%H-%M-%S"),
                "author": {
                    "name": repo.head.object.author.name,
                    "email": repo.head.object.author.email
                }
            }
        except:
            print("Could not gather git repo information.", file=sys.stderr)
        self.save_json()

    def update_parameters(self, params: dict):
        self.run_json['settings'].update(params)
        self.save_json()

    def save_json(self):
        with open(osp.join(self.log_folder, 'experiment.json'), 'w') as f:
            f.write(jsonpickle.encode(self.run_json))

    def run(self,
            run_function: Callable[[int], None],
            num_cpu: int = 1):
        def run(seed: int):
            global _CURRENT
            _CURRENT = self
            logger.configure(
                osp.join(self.log_folder, "seed_%i" % seed),
                format_strs=self.logger_formats)
            config = tf.ConfigProto(allow_soft_placement=True,
                                    intra_op_parallelism_threads=num_cpu,
                                    inter_op_parallelism_threads=num_cpu,
                                    gpu_options=tf.GPUOptions(
                                        per_process_gpu_memory_fraction=1. / num_cpu,
                                        allow_growth=True))
            tf.Session(config=config).__enter__()
            return run_function(seed)

        if num_cpu <= 1:
            import random
            seed = random.randint(0, 2 ** 32 - 1)
            pathlib.Path(self.log_folder, "seed_%i" % seed).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.log_folder, "seed_%i" % seed, "videos").mkdir(parents=True, exist_ok=True)
            run(seed)
            return

        for seed in range(1, num_cpu + 1):
            pathlib.Path(self.log_folder, "seed_%i" % seed).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.log_folder, "seed_%i" % seed, "videos").mkdir(parents=True, exist_ok=True)
            process = Process(target=run,
                              name="Worker %i" % seed,
                              args=[seed])
            process.start()


def log_parameters(original_function: Callable) -> Callable:
    def new_function(*args, **kwargs):
        if _CURRENT is not None:
            _CURRENT.update_parameters({
                '%s_%s' % (original_function.__name__, k): v for k, v in kwargs.items()
            })
        return original_function(*args, **kwargs)

    return new_function
