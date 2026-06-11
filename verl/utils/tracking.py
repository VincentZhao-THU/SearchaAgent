# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
import json
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any

from verl.utils.checkpoint_manager import resolve_checkpoint_step


class Tracking(object):
    supported_backend = ['wandb', 'mlflow', 'console']

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
            if WANDB_API_KEY:
                wandb.login(key=WANDB_API_KEY)
            trainer_config = config.get('trainer', {}) if config is not None else {}
            wandb_init_kwargs = {
                'project': project_name,
                'name': experiment_name,
                'config': config,
            }
            wandb_local_dir = None
            wandb_meta_path = None
            if config is not None:
                default_local_dir = trainer_config.get('default_local_dir', None)
                if default_local_dir:
                    default_local_dir = os.path.abspath(os.path.expanduser(default_local_dir))
                    os.makedirs(default_local_dir, exist_ok=True)
                    wandb_local_dir = os.path.join(default_local_dir, 'wandb')
                    os.makedirs(wandb_local_dir, exist_ok=True)
                    wandb_meta_path = os.path.join(default_local_dir, 'wandb_run.json')
                    wandb_init_kwargs['dir'] = wandb_local_dir

            wandb_run_id = _load_wandb_run_id(wandb_meta_path, project_name, experiment_name)
            checkpoint_step = _resolve_wandb_checkpoint_step(config)
            if wandb_run_id is not None and checkpoint_step is not None:
                wandb_init_kwargs['resume_from'] = f'{wandb_run_id}?_step={checkpoint_step}'
            elif wandb_run_id is not None and trainer_config.get('resume_mode', 'disable') == 'disable':
                # Intentionally starting a fresh run with the same experiment name.
                pass

            run = _init_wandb_run_with_fallback(
                wandb=wandb,
                wandb_init_kwargs=wandb_init_kwargs,
                wandb_run_id=wandb_run_id,
                checkpoint_step=checkpoint_step,
            )
            if wandb_meta_path is not None and run is not None and run.id is not None:
                _save_wandb_run_meta(wandb_meta_path, project_name, experiment_name, run.id)
            self.logger['wandb'] = wandb

        if 'mlflow' in default_backend:
            import mlflow
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow
        mlflow.log_metrics(metrics=data, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    return ans


def _load_wandb_run_id(meta_path: Union[str, None], project_name: str, experiment_name: str) -> Union[str, None]:
    if meta_path is None or not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if metadata.get('project_name') != project_name:
        return None
    if metadata.get('experiment_name') != experiment_name:
        return None

    run_id = metadata.get('run_id', None)
    if isinstance(run_id, str) and run_id:
        return run_id
    return None


def _save_wandb_run_meta(meta_path: str, project_name: str, experiment_name: str, run_id: str) -> None:
    metadata = {
        'project_name': project_name,
        'experiment_name': experiment_name,
        'run_id': run_id,
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)


def _resolve_wandb_checkpoint_step(config) -> Union[int, None]:
    if config is None:
        return None

    trainer_config = config.get('trainer', {})
    resume_mode = trainer_config.get('resume_mode', 'disable')
    default_local_dir = trainer_config.get('default_local_dir', None)
    resume_from_path = trainer_config.get('resume_from_path', None)
    if not default_local_dir:
        return None

    try:
        return resolve_checkpoint_step(
            resume_mode=resume_mode,
            default_local_dir=default_local_dir,
            resume_from_path=resume_from_path,
        )
    except (AssertionError, ValueError):
        return None


def _init_wandb_run_with_fallback(wandb, wandb_init_kwargs: Dict[str, Any], wandb_run_id: Union[str, None],
                                  checkpoint_step: Union[int, None]):
    try:
        return wandb.init(**wandb_init_kwargs)
    except Exception as exc:
        if not _should_fallback_from_rewind(exc, wandb_init_kwargs, wandb_run_id):
            raise

        fallback_kwargs = dict(wandb_init_kwargs)
        fallback_kwargs.pop('resume_from', None)
        fallback_kwargs['id'] = wandb_run_id
        fallback_kwargs['resume'] = 'allow'
        _reset_wandb_state_for_retry(wandb)
        print(
            f'wandb rewind is unavailable for run {wandb_run_id} at checkpoint step {checkpoint_step}; '
            'falling back to resume="allow". W&B history after the checkpoint step will be kept as-is.'
        )
        return wandb.init(**fallback_kwargs)


def _should_fallback_from_rewind(exc: Exception, wandb_init_kwargs: Dict[str, Any],
                                 wandb_run_id: Union[str, None]) -> bool:
    if 'resume_from' not in wandb_init_kwargs or wandb_run_id is None:
        return False

    error_message = str(exc)
    return 'Rewind is in private preview' in error_message


def _reset_wandb_state_for_retry(wandb) -> None:
    teardown = getattr(wandb, 'teardown', None)
    if callable(teardown):
        teardown(exit_code=1)
