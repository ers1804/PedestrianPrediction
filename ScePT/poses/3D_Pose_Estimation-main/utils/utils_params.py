from imp import load_module
import os
import datetime


def gen_run_folder(suffix, path_model_id=''):
    run_paths = dict()
    load_model = False

    if not os.path.exists(path_model_id):
        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'runs'))
        date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        run_id = 'run_' + date_creation
        if path_model_id:
            run_id += '_' + path_model_id
        if suffix:
            run_paths['path_model_id'] = os.path.join(path_model_root, run_id) + '_' + suffix
        else: 
            run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        # normally we only vis models from already trained models so search for ckpt in string
        load_model = True
        if 'ckpts' in path_model_id:
            run_paths['path_model_id'] = path_model_id.split("ckpts")[0]
        else:
            run_paths['path_model_id'] = path_model_id

    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'train.log')
    run_paths['path_logs_eval'] = os.path.join(run_paths['path_model_id'], 'logs', 'eval.log')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')
    run_paths['path_flags'] = os.path.join(run_paths['path_model_id'], 'FLAGS.txt')
    run_paths['path_animations'] = os.path.join(run_paths['path_model_id'], 'animations')
    run_paths['path_animations_2d_pngs'] = os.path.join(run_paths['path_model_id'], 'animations/2d_pngs')
    run_paths['path_animations_3d_projections_pngs'] = os.path.join(run_paths['path_model_id'], 'animations/3d_projections_pngs')
    run_paths['path_animations_rotations'] = os.path.join(run_paths['path_model_id'], 'animations/rotations')
    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ['path_model', 'path_ckpts', 'path_plot', 'path_animations', 'path_animations_2d_pngs', 'path_animations_3d_projections_pngs', 'path_animations_rotations']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient

    return run_paths, load_model


def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)


def update_flags(path, FLAGS):
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        if "model_type" in line:
            FLAGS.model_type = line.replace("--model_type=", "").replace("\n", "")
            return 0

    return -1


def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v

    return data
