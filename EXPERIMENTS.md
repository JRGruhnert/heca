# Experiments

## PPO: Network/Agent 1 to 1

- Normal Buffer
- Every Network trained independent

## PPO: Network/Agent 1 to N

- Fair Buffer

## APPO: Network/Agent 1 to N

- Stream Buffer

## Reliability

### Subgoal unchanged

- Instead of trying to solve the Task the Agent skips and returns the subgoal as it is.
- Compare it with normal mode.
- How much better does the network learn if data is not noisy?
- With and without uncertainty (noise)

### Tapas unchanged

- Same as in "Subgoal unchanged" but only for the expert models that directly interact with the environment

## Vision Encoder

- Repeat Tests with vision encoder
- I can incorperate the uncertainty of the vision encoder into the gnn
