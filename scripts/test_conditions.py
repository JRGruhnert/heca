from heca.agents.agent import Agent
from heca.agents.experts.tapas import TapasAgent

# from heca.agents.heca import Heca
from heca.agents.heca import Heca
from heca.conditions.pair import ConPair
from heca.learning.buffers.ppo_buffer import PPOBuffer
from heca.learning.ppo import PPO
from heca.scenes.ogbench.scene import OGBenchScene

agents = [
    # TapasAgent.Config(
    #     tag="open_drawer",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     tag="close_drawer",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    TapasAgent.Config(
        tag="open_window",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="close_window",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    # TapasAgent.Config(
    #     tag="lock_left_button",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    TapasAgent.Config(
        tag="lock_right_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    # TapasAgent.Config(
    #     tag="unlock_left_button",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    TapasAgent.Config(
        tag="unlock_right_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    # TapasAgent.Config(
    #     tag="move_block",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #    tag="move_ee",
    #    scene=OGBenchScene.Config(),
    #    use_gt=True,
    # ),
]


heca_cfg = Heca.Config(
    tag="test",
    agents=agents,
    ppo=PPO.Config(
        tag="test",
        # buffer=APPOBuffer.Config(),
        buffer=PPOBuffer.Config(),
    ),
)
path = Agent.load_dir(heca_cfg)
cons: list[ConPair] = []
for cfg in heca_cfg.agents:
    cons.extend(Agent.get(cfg).conditions)

sets = [{i} for i in range(len(cons))]
while True:
    merged = False
    for i in range(len(cons)):
        for j in range(i + 1, len(cons)):
            a = cons[i]
            b = cons[j]
            if a.can_merge(b, path):
                print(f"{a.label} and {b.label} merge")
                a_set = sets[i]
                b_set = sets[j]
                new_set = a_set | b_set
                ids = map(str, sorted(new_set))
                label = f"{heca_cfg.tag}_{''.join(ids)}"
                new_pair = ConPair.merge(
                    label=label,
                    a=a,
                    b=b,
                    n_samples=heca_cfg.n_samples,
                    threshold=cfg.threshold,
                )
                new_pair.plot(path)
                cons.pop(j)
                cons.pop(i)
                sets.pop(j)
                sets.pop(i)
                sets.append(new_set)
                cons.append(new_pair)

                merged = True
                break
        if merged:
            break
    if not merged:
        break

print(len(sets))
