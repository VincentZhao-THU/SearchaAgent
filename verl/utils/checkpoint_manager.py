import os
import re
from typing import Optional


_GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")


def extract_global_step(path: str) -> Optional[int]:
    match = _GLOBAL_STEP_PATTERN.match(os.path.basename(os.path.normpath(path)))
    if match is None:
        return None
    return int(match.group(1))


def find_latest_ckpt_path(checkpoint_root: str) -> Optional[str]:
    checkpoint_root = os.path.abspath(os.path.expanduser(checkpoint_root))
    if not os.path.isdir(checkpoint_root):
        return None

    latest_step = None
    latest_path = None
    for entry in os.listdir(checkpoint_root):
        full_path = os.path.join(checkpoint_root, entry)
        if not os.path.isdir(full_path):
            continue
        step = extract_global_step(entry)
        if step is None:
            continue
        if latest_step is None or step > latest_step:
            latest_step = step
            latest_path = full_path

    return latest_path


def resolve_checkpoint_step(
    resume_mode: str,
    default_local_dir: str,
    resume_from_path: Optional[str] = None,
) -> Optional[int]:
    resolved_path = resolve_checkpoint_path(
        resume_mode=resume_mode,
        default_local_dir=default_local_dir,
        resume_from_path=resume_from_path,
    )
    if resolved_path is None:
        return None
    return extract_global_step(resolved_path)


def resolve_checkpoint_path(
    resume_mode: str,
    default_local_dir: str,
    resume_from_path: Optional[str] = None,
) -> Optional[str]:
    if resume_mode == 'disable':
        return None

    checkpoint_root = os.path.abspath(os.path.expanduser(default_local_dir))

    if resume_mode == 'auto':
        global_step_folder = find_latest_ckpt_path(checkpoint_root)
        if global_step_folder is not None:
            return global_step_folder

        legacy_actor_root = os.path.join(checkpoint_root, 'actor')
        legacy_actor_path = find_latest_ckpt_path(legacy_actor_root)
        if legacy_actor_path is not None:
            return legacy_actor_path
        return None

    if resume_mode == 'resume_path':
        assert isinstance(resume_from_path, str) and resume_from_path, \
            'resume_from_path must be a non-empty string when resume_mode=resume_path'
        global_step_folder = os.path.abspath(os.path.expanduser(resume_from_path))
        parent_dir = os.path.basename(os.path.dirname(os.path.normpath(global_step_folder)))
        if parent_dir in {'actor', 'critic'}:
            return global_step_folder
        return global_step_folder

    raise ValueError(f'Unsupported resume_mode: {resume_mode}')
