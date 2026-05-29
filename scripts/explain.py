from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode

from tapas_gmm.utils.argparse import parse_and_build_config

from scripts.train import TrainConfig
from src.agents.ppo.gnn import GNNAgent
from src.modules.buffer import Buffer
from src.modules.storage import Storage
from src.factory import (
    select_agent,
    select_environment,
    select_experiment,
    select_evaluator,
)


@dataclass
class ExplainConfig:
    source: TrainConfig
    target: TrainConfig


def run_explain(config: ExplainConfig):
    s_storage = Storage(config.source.storage)
    s_buffer = Buffer(config.source.buffer)

    s_evaluator = select_evaluator(config.source.evaluator, s_storage)
    s_env = select_environment(config.source.environment, s_evaluator, s_storage)
    s_experiment = select_experiment(config.source.experiment, s_env, s_storage)
    s_agent = select_agent(config.source.agent, s_storage, s_buffer)

    t_storage = Storage(config.target.storage)
    t_buffer = Buffer(config.target.buffer)
    t_evaluator = select_evaluator(config.target.evaluator, t_storage)
    t_env = select_environment(config.target.environment, t_evaluator, t_storage)
    t_experiment = select_experiment(config.target.experiment, t_env, t_storage)
    t_agent = select_agent(config.target.agent, t_storage, t_buffer)
    assert isinstance(s_agent, GNNAgent) and isinstance(
        t_agent, GNNAgent
    ), "Agents must be GNNAgent for explanation"

    s_obs, s_goal = s_experiment.sample_task()
    t_obs, t_goal = t_experiment.sample_task()
    while not s_evaluator.is_equal(s_obs, s_goal) or not t_evaluator.is_equal(
        t_obs, t_goal
    ):
        s_obs, s_goal = s_experiment.sample_task()
        t_obs, t_goal = t_experiment.sample_task()

    s_step = 0
    s_skills_list = []
    s_done = False
    s_episode_ended = False
    while not s_episode_ended:
        state_diff = s_evaluator.is_equal_dict(s_obs, s_goal)
        skill = s_agent.explain(s_obs, s_goal, step=s_step, state_diff=state_diff)
        # print(f"[step {s_step}] skill = {skill.name}")
        s_skills_list.append(skill.name)
        s_obs, reward, s_done, s_episode_ended = s_experiment.step(skill)
        s_agent.feedback(reward, s_done, s_episode_ended)
        s_step += 1
    s_experiment.close()

    t_step = 0
    t_skills_list = []
    t_done = False
    t_episode_ended = False
    while not t_episode_ended:
        state_diff = t_evaluator.is_equal_dict(t_obs, t_goal)
        skill = t_agent.explain(t_obs, t_goal, step=t_step, state_diff=state_diff)
        # print(f"[step {t_step}] skill = {skill.name}")
        t_skills_list.append(skill.name)
        t_obs, reward, t_done, t_episode_ended = t_experiment.step(skill)
        t_agent.feedback(reward, t_done, t_episode_ended)
        t_step += 1

    t_experiment.close()
    print(f"Executed skills: {s_skills_list}")
    print(f"Done: {s_done}, episode_ended: {s_episode_ended}")


def entry_point():
    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    dict_config["storage"]["tag"] = (
        dict_config["storage"]["tag"]
        + f"_pe{dict_config['experiment']['p_empty']}_pr{dict_config['experiment']['p_rand']}"
    )

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    run_explain(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
