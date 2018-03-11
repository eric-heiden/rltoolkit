import gym


def create_environment(name: str) -> gym.Env:
    ids = name.split('-')
    framework = ids[0].lower()
    env_id = '-'.join(ids[1:])
    if framework == 'dm':
        from envs.deepmind import DMSuiteEnv
        return DMSuiteEnv(env_id)
    elif framework == 'gym':
        return gym.make(env_id)
    elif framework == 'rllab':
        from envs.rllab import RllabEnv
        return RllabEnv(env_id)

    raise LookupError("Could not find environment \"%s\"." % env_id)


def load_dm_control():
    import sys, os, subprocess
    import os.path as osp

    sys.path.insert(0, osp.join(osp.dirname(__file__), 'dm_control'))
    print("Setting up MuJoCo bindings for dm_control...")
    DEFAULT_HEADERS_DIR = '~/.mujoco/mjpro150/include'

    # Relative paths to the binding generator script and the output directory.
    AUTOWRAP_PATH = 'dm_control/dm_control/autowrap/autowrap.py'
    MJBINDINGS_DIR = 'dm_control/dm_control/mujoco/wrapper/mjbindings'

    # We specify the header filenames explicitly rather than listing the contents
    # of the `HEADERS_DIR` at runtime, since it will probably contain other stuff
    # (e.g. `glfw.h`).
    HEADER_FILENAMES = [
        'mjdata.h',
        'mjmodel.h',
        'mjrender.h',
        'mjvisualize.h',
        'mjxmacro.h',
        'mujoco.h',
    ]

    # A default value must be assigned to each user option here.
    inplace = 0
    headers_dir = os.path.expanduser(DEFAULT_HEADERS_DIR)

    header_paths = []
    for filename in HEADER_FILENAMES:
        full_path = os.path.join(headers_dir, filename)
        if not os.path.exists(full_path):
            raise IOError('Header file {!r} does not exist.'.format(full_path))
        header_paths.append(full_path)
    header_paths = ' '.join(header_paths)

    cwd = os.path.realpath(os.curdir)
    output_dir = os.path.join(cwd, MJBINDINGS_DIR)
    command = [
        sys.executable or 'python',
        AUTOWRAP_PATH,
        '--header_paths={}'.format(header_paths),
        '--output_dir={}'.format(output_dir)
    ]
    old_environ = os.environ.copy()
    try:
        # Prepend the current directory to $PYTHONPATH so that internal imports
        # in `autowrap` can succeed before we've installed anything.
        new_pythonpath = [cwd]
        if 'PYTHONPATH' in old_environ:
            new_pythonpath.append(old_environ['PYTHONPATH'])
        os.environ['PYTHONPATH'] = ':'.join(new_pythonpath)
        subprocess.check_call(command)
    finally:
        os.environ = old_environ

    # XXX this hack works around bug in dm_control where it cannot find a GL environment
    try:
        from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
    except:
        pass

    try:
        from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
    except:
        pass

    try:
        from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
    except:
        pass
