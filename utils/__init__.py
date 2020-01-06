import subprocess
import os

__cwd__ = os.getcwd()

__PYTHONPATH_ENV_VAR__ = {'PYTHONPATH': rf'{__cwd__}\models\research;'
                                        rf'{__cwd__}\models\research\slim;'}


async def train_async(pipeline_config_path, model_dir):
    if not pipeline_config_path or not model_dir:
        raise ValueError('Variable ${pipeline_config_path} or ${model_dir} not set!')
    __call_str__ = (
        r'python models/research/object_detection/model_main.py '
        f'--pipeline_config_path={pipeline_config_path} '
        f'--model_dir={model_dir} '
        r'--alsologtostderr'
    )
    subprocess.check_call(__call_str__, env=__PYTHONPATH_ENV_VAR__)


async def train_legacy_async(pipeline_config_path, train_dir,
                             worker_replicas=1, num_clones=1, ps_tasks=1):
    if not pipeline_config_path or not train_dir:
        raise ValueError('Variable ${pipeline_config_path} or ${train_dir} not set!')
    __call_str__ = (
        r'python models/research/object_detection/legacy/train.py '
        f'--pipeline_config_path={pipeline_config_path} '
        f'--train_dir={train_dir} '
        f'--worker_replicas={worker_replicas} '
        f'--num_clones={num_clones} '
        f'--ps_tasks={ps_tasks} '
        r'--logtostderr'
    )
    subprocess.check_call(__call_str__, env=__PYTHONPATH_ENV_VAR__)