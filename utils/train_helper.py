import os
import subprocess
from log_helper import str_warning

__exec__ = subprocess.call

__dir_name__ = os.path.dirname
__cwd__ = __dir_name__(__dir_name__(__file__))

__ENV__ = os.environ
if 'PYTHONPATH' not in __ENV__:
    __ENV__['PYTHONPATH'] = ''
__ENV__['PYTHONPATH'] += rf'{__cwd__}\models\research;' \
                         rf'{__cwd__}\models\research\slim;'


async def train_async(pipeline_config_path, model_dir):
    if not pipeline_config_path or not model_dir:
        raise ValueError(
            'Variable ${pipeline_config_path} or ${model_dir} not set!')
    __call_str__ = (r'python models/research/object_detection/model_main.py '
                    f'--pipeline_config_path={pipeline_config_path} '
                    f'--model_dir={model_dir} '
                    r'--alsologtostderr')
    print(f'calling: [{__call_str__}]')
    __exec__(__call_str__, env=__ENV__)


async def train_legacy_async(pipeline_config_path,
                             train_dir,
                             worker_replicas=1,
                             num_clones=1,
                             ps_tasks=1):
    if not pipeline_config_path or not train_dir:
        raise ValueError(
            'Variable ${pipeline_config_path} or ${train_dir} not set!')
    __call_str__ = (r'python models/research/object_detection/legacy/train.py '
                    f'--pipeline_config_path={pipeline_config_path} '
                    f'--train_dir={train_dir} '
                    f'--worker_replicas={worker_replicas} '
                    f'--num_clones={num_clones} '
                    f'--ps_tasks={ps_tasks} '
                    r'--logtostderr')
    print(f'calling: [{__call_str__}]')
    __exec__(__call_str__, env=__ENV__)


async def export_inference_graph_async(pipeline_config_path,
                                       output_path,
                                       ckpt_path_prefix_only,
                                       input_type=None):
    if not pipeline_config_path or not ckpt_path_prefix_only or not output_path:
        raise ValueError('Variable ${pipeline_config_path} or ${ckpt_prefix}'
                         ' or ${output_dir} not set!')
    if os.path.realpath(pipeline_config_path) == os.path.realpath(output_path):
        print(str_warning)
        print(
            'Using same ${pipeline_config_path} and ${ckpt_prefix} may ruin existing data!'
        )
    __call_str__ = (
        r'python models/research/object_detection/export_inference_graph.py '
        f'--pipeline_config_path {pipeline_config_path} '
        f'--output_directory {output_path} '
        f'--trained_checkpoint_prefix {ckpt_path_prefix_only} ' +
        (f'--input_type {input_type} ' if input_type else ''))
    print(f'calling: [{__call_str__}]')
    __exec__(__call_str__, env=__ENV__)