# Experiments

## One Run for each:

| Run | SMode | NMode   | Virtuell? |
| --- | ----- | ------- | --------- |
| 01  | None  | Default | True      |
| 02  | None  | Memory  | True      |
| 03  | None  | Compare | True      |
| 04  | None  | Both    | True      |
| 05  | Post  | Default | True      |
| 06  | Post  | Memory  | True      |
| 07  | Post  | Compare | True      |
| 08  | Post  | Both    | True      |
| 09  | Chain | Default | True      |
| 10  | Chain | Memory  | True      |
| 11  | Chain | Compare | True      |
| 12  | Chain | Both    | True      |
| 13  | None  | Default | False     |
| 14  | None  | Memory  | False     |
| 15  | None  | Compare | False     |
| 16  | None  | Both    | False     |
| 17  | Post  | Default | False     |
| 18  | Post  | Memory  | False     |
| 19  | Post  | Compare | False     |
| 20  | Post  | Both    | False     |
| 21  | Chain | Default | False     |
| 22  | Chain | Memory  | False     |
| 23  | Chain | Compare | False     |
| 24  | Chain | Both    | False     |

- Normal mode: 24x6x3 (Runs x n_scene x n_repeats)
- Federated mode: 24x3 x 5 (Runs x n_scene x n_repeats x fed_prox_hyperparameter settings)
- Visual mode: 1 Run with best Mode from previous runs
