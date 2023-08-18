waymo_weakly_supervised = {
    "project": "waymo_weakly_supervised",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "method": "grid",
    "parameters": {

        "load.name": {'values': ["waymo_weakly_supervised"]},

        "SelfSupervisedTrainer.epochs": {'values': [20]},
        # "SelfSupervisedTrainer.lr_G": {'values': [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]},
        # "SelfSupervisedTrainer.lr_D": {'values': [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]},
        "SelfSupervisedTrainer.direct_reprojection_weight": {'values': [0, 1, 10]},
        "SelfSupervisedTrainer.pseudo_weight": {'values': [1, 10, 20, 25]},
        "SelfSupervisedTrainer.relifting_weight": {'values': [0, 1, 10]},
        "SelfSupervisedTrainer.adv_weight": {'values': [0, 1, 10]},
        "SelfSupervisedTrainer.reprojection_weight":  {'values': [0, 1, 10]},
    }
}

waymo_alphapose_weakly_supervised = {
    "project": "waymo_weakly_supervised",
    "metric": {"name": "best_val_loss", "goal": "minimize"},
    "method": "grid",
    "parameters": {

        "load.name": {'values': ["waymo_alphapose_weakly_supervised"]},

        "SelfSupervisedTrainer.epochs": {'values': [25]},
        # "SelfSupervisedTrainer.lr_G": {'values': [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]},
        # "SelfSupervisedTrainer.lr_D": {'values': [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]},
        "SelfSupervisedTrainer.direct_reprojection_weight": {'values': [0, 1, 10]},
        "SelfSupervisedTrainer.pseudo_weight": {'values': [1, 10, 20, 25]},
        "SelfSupervisedTrainer.relifting_weight": {'values': [0, 1, 10]},
        "SelfSupervisedTrainer.adv_weight": {'values': [0, 1, 10]},
        "SelfSupervisedTrainer.reprojection_weight":  {'values': [0, 1, 10]},
    }
}

