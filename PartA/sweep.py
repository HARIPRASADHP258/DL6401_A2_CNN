import wandb
from main import train
wandb.login(key="999fe4f321204bd8f10135f3e40de296c23050f9")

sweep_configuration = {
    'method': 'bayes',
    'name': 'inaturalist-sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': {
        'batch_size': {'values': [32, 64]},
        'conv_filters': {'values': [[64,128,256,512,512], [64,128,256,384,512], [64,128,256,384,384]]},
        'filter_sizes': {'values': [[3,3,3,3,3], [5,5,3,3,3], [3,3,5,5,5], [3,5,7,5,3], [7,5,3,5,7]]},
        'learning_rate': {'values': [1e-5, 5e-5, 1e-4]},
        'epochs': {'values': [10,13,15,20]},
        'activation': {'values': ['mish', 'elu', 'gelu', 'swish']},
        'dense_neurons': {'values': [32, 64, 128, 256, 512]},
        'dense_dropout': {'values': [0.2, 0.3]},
        'weight_decay': {'values': [0.001, 0.0005, 0.0001]},
        'use_batchnorm': {'values': [True]},
        'use_augmentation': {'values': [False]},
        'mean': {'value': [0.471, 0.460, 0.390]},
        'std': {'value': [0.193, 0.188, 0.184]}
    }
}

def train_sweep():
    with wandb.init() as run:
        config = dict(wandb.config)
        run.name = f"{config['batch_size']}-{config['conv_filters']}-{config['filter_sizes']}"
        model_path = train(config, run)
        run.log({"best_model_path": model_path})

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="inaturalist_classify")
    wandb.agent(sweep_id, function=train_sweep, count=1)