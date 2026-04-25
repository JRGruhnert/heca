import wandb
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig

from hoopgn.storages.storage import StorageConfig

from cli.cmd_train import TrainRunnerConfig, TrainRunner

import click


@click.command()
def sweep(cfg_path):
    click.echo(f"Not implemented yet. Would run sweep with config from {cfg_path}")


def entry_point():
    run = wandb.init()

    config = TrainRunnerConfig(
        skills=wandb.config["storage.used_skills"],
        properties=wandb.config["storage.used_states"],
        agent=HoopGNSkillConfig(
            buffer=BufferConfig(size=wandb.config["agent.batch_size"]),
            storage=StorageConfig(
                skills=wandb.config["storage.used_skills"],
                states_network=wandb.config["storage.used_states"],
                states_eval=wandb.config["storage.eval_states"],
            ),
            network=wandb.config["agent.network"],
            eval=wandb.config["agent.eval"],
            early_stop_patience=wandb.config["agent.early_stop_patience"],
            use_ema_for_early_stopping=wandb.config["agent.use_ema_for_early_stopping"],
            ema_smoothing_factor=wandb.config["agent.ema_smoothing_factor"],
            min_batches=wandb.config["agent.min_batches"],
            max_batches=wandb.config["agent.max_batches"],
            saving_freq=wandb.config["agent.saving_freq"],
            save_stats=wandb.config["agent.save_stats"],
            mini_batch_size=wandb.config["agent.mini_batch_size"],
            learning_epochs=wandb.config["agent.learning_epochs"],
            lr_annealing=wandb.config["agent.lr_annealing"],
            learning_rate=wandb.config["agent.learning_rate"],
            gamma=wandb.config["agent.gamma"],
            gae_lambda=wandb.config["agent.gae_lambda"],
            eps_clip=wandb.config["agent.eps_clip"],
            entropy_coef=wandb.config["agent.entropy_coef"],
            critic_coef=wandb.config["agent.critic_coef"],
            max_grad_norm=wandb.config["agent.max_grad_norm"],
            target_kl=wandb.config["agent.target_kl"],
            clip_value_loss=wandb.config["agent.clip_val_loss"],
        ),
        experiment=NoiseExperimentConfig(
            p_skip=wandb.config["experiment.p_empty"],
            p_rand=wandb.config["experiment.p_rand"],
            environment=CalvinEnvironmentConfig(),
            evaluator=Dense3EvaluatorConfig(
                states_network=wandb.config["storage.used_states"],
                states_eval=wandb.config["storage.eval_states"],
            ),
            min_steps=wandb.config["experiment.min_steps"],
        ),
    )

    trainer = TrainRunner(config)
    trainer.run()
