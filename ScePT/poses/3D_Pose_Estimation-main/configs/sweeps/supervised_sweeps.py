waymo_3d_2d_projections_supervised = {
    "project": "waymo_3d_2d_projections_supervised",
    "metric": {"name": "best_val_loss", "goal": "minimize"},
    "method": "bayes",
    "parameters": {

        "load.name": {'values': ["waymo_3d_2d_projections_supervised"]},

        "Trainer.epochs": {'values': [150]},
        "Trainer.lr_decay_factor": {'values': [0.95, 0.9, 0.8, 0.99]},
        "Trainer.lr_step": {'values': [375, 500, 750, 1000]},
        "Trainer.lr": {
            'distribution': 'uniform',
            'min': 0.00005,
            'max': 0.0005
        },

        "SimpleLiftingModel.latent_dim": {'values': [2048, 1024, 512]},

        "Trainer.loss_types.l1": {'values': [0, 0.25, 0.5, 0.75, 1]},
        "Trainer.loss_types.bone_length": {'values': [0, 0.1, 0.25, 0.3, 0.5, 0.75, 1]},
        "Trainer.loss_types.feature_transform_reguliarzer": {'values': [0,  0.0001, 0.00025, 0.0005, 0.001, 0.01, 0.05, 0.1]},

    }
}

waymo_3d_projections_fusion = {
    "project": "waymo_3d_projections_fusion",
    "metric": {"name": "best_val_loss", "goal": "minimize"},
    "method": "bayes",
    "parameters": {

        "load.name": {'values': ["waymo_3d_2d_projections_supervised"]},

        "SupervisedTrainer.epochs": {'values': [250]},
        "SupervisedTrainer.lr_decay_factor": {'values': [0.95, 0.9, 0.8, 0.99]},
        "SupervisedTrainer.lr_step": {'values': [375, 500, 750, 1000]},
        "SupervisedTrainer.lr": {
            'distribution': 'uniform',
            'min': 0.00005,
            'max': 0.001
        },

        "Lidar2dKeypointFusionmodel.hidden_size": {'values': [2048, 1024, 512]},
        "Lidar2dKeypointFusionmodel.dropout": {'values': [0, 0.2, 0.35, 0.5, 0.7]},

        "SupervisedTrainer.loss_types.masked_mpjpe": {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 1
        },
        "SupervisedTrainer.loss_types.l1": {'values': [0, 0.25, 0.5, 0.75, 1]},
        "SupervisedTrainer.loss_types.bone_length": {'values': [0, 0.1, 0.25, 0.3, 0.5, 0.75, 1]},
        "SupervisedTrainer.loss_types.feature_transform_reguliarzer": {'values': [0,  0.0001, 0.00025, 0.0005, 0.001, 0.01, 0.05, 0.1]},

    }
}


waymo_3d_lidar_supervised = {
    "project": "waymo_3d_lidar_supervised",
    "metric": {"name": "best_val_loss", "goal": "minimize"},
    "method": "bayes",
    "parameters": {

        "load.name": {'values': ["waymo_2d_labels_supervised"]},

        "SupervisedTrainer.epochs": {'values': [250]},
        "SupervisedTrainer.lr_decay_factor": {'values': [0.95, 0.9, 0.8, 0.99]},
        "SupervisedTrainer.lr_step": {'values': [375, 500, 750, 1000]},
        "SupervisedTrainer.lr": {
            'distribution': 'uniform',
            'min': 0.00005,
            'max': 0.005
        },


        "PointNet.dropout": {'values': [0, 0.2, 0.35, 0.5, 0.7]},

        "SupervisedTrainer.loss_types.masked_mpjpe": {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 1
        },
        "SupervisedTrainer.loss_types.l1": {'values': [0, 0.25, 0.5, 0.75, 1]},
        "SupervisedTrainer.loss_types.bone_length": {'values': [0, 0.1, 0.25, 0.3, 0.5, 0.75, 1]},
        "SupervisedTrainer.loss_types.feature_transform_reguliarzer": {'values': [0,  0.0001, 0.00025, 0.0005, 0.001, 0.01, 0.05, 0.1]},

    }
}


waymo_2d_labels_supervised = {
    "project": "waymo_2d_labels_supervised",
    "metric": {"name": "best_val_loss", "goal": "minimize"},
    "method": "bayes",
    "parameters": {

        "load.name": {'values': ["waymo_2d_labels_supervised"]},

        "SupervisedTrainer.epochs": {'values': [150]},
        "SupervisedTrainer.lr_decay_factor": {'values': [0.95, 0.9, 0.8, 0.99]},
        "SupervisedTrainer.lr_step": {'values': [375, 500, 750, 1000]},
        "SupervisedTrainer.lr": {
            'distribution': 'uniform',
            'min': 0.00005,
            'max': 0.0005
        },

        "SimpleLiftingModel.latent_dim": {'values': [2048, 1024, 512]},

        "SupervisedTrainer.loss_types.l1": {'values': [0, 0.25, 0.5, 0.75, 1]},
        "SupervisedTrainer.loss_types.bone_length": {'values': [0, 0.1, 0.25, 0.3, 0.5, 0.75, 1]},

    }
}

waymo_2d_labels_fusion = {
    "project": "waymo_2d_labels_fusion",
    "metric": {"name": "best_val_loss", "goal": "minimize"},
    "method": "bayes",
    "parameters": {

        "load.name": {'values': ["waymo_2d_labels_supervised"]},

        "SupervisedTrainer.epochs": {'values': [250]},
        "SupervisedTrainer.lr_decay_factor": {'values': [0.95, 0.9, 0.8, 0.99]},
        "SupervisedTrainer.lr_step": {'values': [375, 500, 750, 1000]},
        "SupervisedTrainer.lr": {
            'distribution': 'uniform',
            'min': 0.00005,
            'max': 0.001
        },

        "Lidar2dKeypointFusionmodel.hidden_size": {'values': [2048, 1024, 512]},
        "Lidar2dKeypointFusionmodel.dropout": {'values': [0, 0.2, 0.35, 0.5, 0.7]},

        "SupervisedTrainer.loss_types.masked_mpjpe": {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 1
        },
        "SupervisedTrainer.loss_types.l1": {'values': [0, 0.25, 0.5, 0.75, 1]},
        "SupervisedTrainer.loss_types.bone_length": {'values': [0, 0.1, 0.25, 0.3, 0.5, 0.75, 1]},
        "SupervisedTrainer.loss_types.feature_transform_reguliarzer": {'values': [0,  0.0001, 0.00025, 0.0005, 0.001, 0.01, 0.05, 0.1]},

    }
}
