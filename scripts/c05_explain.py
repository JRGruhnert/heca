import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from heca.agents.heca import Heca
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.graph import Graph, SubgoalMode
from heca.graphs.roles import ENMode, ENRole
from heca.learning.ppo import PPO
from heca.misc import hardware, logger
from scripts.common.helper import find_checkpoint, fmt_duration
from scripts.common.scenes import find_scene_models

import conf.networks

from heca.misc.paths import data_root

RUN_ROOT = data_root() / "network/standard"


def capture_build(heca: Heca) -> dict:
    holder: dict = {}
    build = heca.graph.build

    def capture(x, y, budget):
        holder["data"] = build(x, y, budget)
        return holder["data"]

    heca.graph.build = capture
    return holder


def logits_of(net, data) -> torch.Tensor:
    """Greedy logits for a built graph, without touching the autograd graph."""
    if getattr(data, "memory", None) is None:
        data.memory = {}
    with torch.no_grad():
        return net(data).logits.detach()


def row_index(data, graph: Graph) -> dict[str, list[int]]:
    """Entity name -> its row indices, and comp key -> its comp row index."""
    entity_rows: dict[str, list[int]] = {}
    for i, node in enumerate(graph.ns_entity.items):
        entity_rows.setdefault(node.entity, []).append(i)
    comp_rows: dict[str, int] = {key: i for i, key in enumerate(graph.ns_comp.keys)}
    return entity_rows, comp_rows


@dataclass
class StepExplanation:
    step: int
    budget: float
    chosen: int
    chosen_label: str
    logits: list[float]
    labels: list[str]
    gated: list[bool]
    value: float
    row_saliency: dict[str, float]
    entity_saliency: dict[str, float]
    edges: list[tuple[str, str, float]]
    option_saliency: list[float]
    budget_saliency: float
    gate_saliency: float
    occluded: dict[str, dict[str, float]]


def gradient_saliency(
    net, graph: Graph, data, chosen: int, entity_rows: dict[str, list[int]]
) -> dict:
    """|d logit(chosen) / d input| for every input block the model reads."""
    entity = data["entity"].x
    comp = data["comp"].x
    option = data["option"].x
    state = data["state"].x
    edge_attr = data[ConditionEdges.type].edge_attr
    blocks = (entity, comp, option, state, edge_attr)
    for block in blocks:
        block.requires_grad_(True)

    net.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = net(data)
        out.logits[0, chosen].backward()

    with torch.no_grad():
        per_row = (
            entity.grad.sum(dim=1)
            if entity.grad is not None
            else torch.zeros(entity.shape[0])
        )
        per_edge = (
            edge_attr.grad.abs().sum(dim=1)
            if edge_attr.grad is not None
            else torch.zeros(edge_attr.shape[0])
        )
        result = {
            "row_saliency": {
                graph.ns_entity.keys[i]: float(per_row[i])
                for i in range(len(graph.ns_entity.keys))
            },
            "entity_saliency": {
                name: float(per_row[rows].sum()) for name, rows in entity_rows.items()
            },
            "edges": sorted(
                [
                    (
                        graph.ns_comp.keys[src],
                        graph.ns_entity.keys[dst],
                        float(per_edge[k]),
                    )
                    for k, (src, dst) in enumerate(graph.es_condition.edges)
                ],
                key=lambda edge: -edge[2],
            ),
            "option_saliency": (
                option.grad.abs().sum(dim=1).tolist()
                if option.grad is not None
                else [0.0] * option.shape[0]
            ),
            "budget_saliency": (
                float(state.grad.abs().sum()) if state.grad is not None else 0.0
            ),
            "gate_saliency": (
                float(option.grad.abs()[:, 0].sum()) if option.grad is not None else 0.0
            ),
        }

    for block in blocks:
        block.requires_grad_(False)
        block.grad = None
    net.zero_grad(set_to_none=True)
    return result


def occlusion_delta(
    net, data, chosen: int, base: float, rows: list[int] | None, comps: list[int] | None
) -> float:
    """Change in logit(chosen) when the given rows are zeroed, then restored."""
    entity = data["entity"].x
    comp = data["comp"].x
    saved_entity = entity.clone()
    saved_comp = comp.clone() if comps else None
    if rows:
        entity[rows] = 0.0
    if comps:
        comp[comps] = 0.0
    delta = float(logits_of(net, data)[0, chosen]) - base
    entity.copy_(saved_entity)
    if saved_comp is not None:
        comp.copy_(saved_comp)
    return delta


def occlusion(
    net,
    graph: Graph,
    data,
    chosen: int,
    base: float,
    entity_rows: dict[str, list[int]],
    comp_rows: dict[str, int],
) -> dict[str, dict[str, float]]:
    roles = data["entity"].role_ids
    modes = [node.mode for node in graph.ns_entity.items]
    comps_of: dict[str, list[int]] = {}
    for key, i in comp_rows.items():
        comps_of.setdefault(graph.ns_comp.items[i].entity, []).append(i)

    result: dict[str, dict[str, float]] = {}
    for name, rows in entity_rows.items():
        current = [
            i for i in rows if roles[i] in (ENRole.START.value, ENRole.PRE.value)
        ]
        goal = [
            i for i in rows if roles[i] == ENRole.GOAL.value or modes[i] == ENMode.GOAL
        ]
        result[name] = {
            "current": occlusion_delta(net, data, chosen, base, current, None),
            "goal": occlusion_delta(net, data, chosen, base, goal, None),
            "comps": occlusion_delta(net, data, chosen, base, None, comps_of.get(name)),
        }
    return result


def plot_step(path: Path, explanation: StepExplanation) -> None:
    """Option logits (z-scored) and signed per-entity saliency, one figure.

    Minimal on purpose: no titles or axis labels, the legend lives in LaTeX.
    Panel 1: per-step z-scored logits of every option in its fixed graph index
    order, as a thin-bar distribution centred on 0 (gold = chosen, grey x =
    gated off). Panel 2: signed d logit(chosen)/d entity rows, centred on 0.
    """
    logits = np.asarray(explanation.logits, dtype=float)
    finite = np.isfinite(logits)
    centre = logits[finite].mean() if finite.any() else 0.0
    scale = logits[finite].std() if finite.sum() > 1 else 1.0
    z = np.where(finite, (logits - centre) / max(scale, 1e-9), 0.0)
    order = np.arange(len(logits))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), squeeze=False)
    ax = axes[0, 0]

    colours = [
        (
            "gold"
            if i == explanation.chosen
            else ("tab:blue" if finite[i] else "lightgrey")
        )
        for i in order
    ]
    ax.bar(order, z, width=0.6, color=colours)
    for idx in order[~finite]:  # gated off: mark the zero-height bar visibly
        ax.plot(idx, 0.0, marker="x", color="grey", markersize=3.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    span = float(np.abs(z[finite]).max()) if finite.any() else 1.0
    ax.set_xlim(-1.0, len(z))
    ax.set_ylim(-1.15 * max(span, 1e-6), 1.15 * max(span, 1e-6))
    ax.grid(True, alpha=0.3, axis="y")

    ax2 = axes[0, 1]
    names = sorted(explanation.entity_saliency)
    values = [explanation.entity_saliency[n] for n in names]
    ax2.bar(
        names,
        values,
        color=["tab:red" if v >= 0.0 else "tab:blue" for v in values],
    )
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.tick_params(axis="x", rotation=45, labelsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def logit_grid(steps: list[dict], n_steps: int = 16) -> np.ndarray:
    """(n_options, n_steps) z-scored logits, each column scaled to its own [0, 1].

    Gated options sit at -0.05 (below the viridis scale, i.e. darkest) and steps
    after the end of the episode are NaN, drawn as white.
    """
    n_options = len(steps[0]["labels"])
    zs = []
    for s in steps:
        logits = np.asarray(s["logits"], dtype=float)
        finite = np.isfinite(logits)
        centre = logits[finite].mean() if finite.any() else 0.0
        scale = logits[finite].std() if finite.sum() > 1 else 1.0
        zs.append(np.where(finite, (logits - centre) / max(scale, 1e-9), np.nan))

    taken = min(len(zs), n_steps)
    display = np.full((n_options, n_steps), np.nan)
    for t in range(taken):
        col = zs[t].copy()
        finite = np.isfinite(col)
        n_finite = int(finite.sum())
        if n_finite > 1:  # each column scaled to its own min/max
            lo, hi = float(col[finite].min()), float(col[finite].max())
            rng = (hi - lo) or 1.0
            col[finite] = (col[finite] - lo) / rng
        elif n_finite == 1:
            col[finite] = 0.5
        col[~finite] = -0.05  # gated -> below vmin -> darkest viridis
        display[:, t] = col
    return display


def saliency_grid(
    steps: list[dict], n_steps: int = 16
) -> tuple[np.ndarray, list[str]]:
    """(n_entities, n_steps) signed entity saliency and the entity names."""
    entities = sorted(steps[0]["entity_saliency"])
    taken = min(len(steps), n_steps)
    grid = np.full((len(entities), n_steps), np.nan)
    for t in range(taken):
        for row, name in enumerate(entities):
            grid[row, t] = float(steps[t]["entity_saliency"].get(name, 0.0))
    return grid, entities


def saliency_span(grid: np.ndarray) -> float:
    """Symmetric colormap limit for a saliency grid (1.0 when it is all zero)."""
    return float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0


def plot_episode_heatmaps(path: Path, steps: list[dict], n_steps: int = 16) -> None:
    """Two heatmaps over the episode: option logits and entity saliency.

    Top: steps x options (fixed order), cells = z-scored logits; every column
    is scaled to its own min/max, so each step's strongest option is yellow
    (viridis), gated options sit below the scale (darkest), white = after the
    episode ended. Bottom: steps x entities, cells = signed saliency (RdYlGn:
    red = pushes the choice down, green = up).
    """
    if not steps:
        return
    n_options = len(steps[0]["labels"])
    display = logit_grid(steps, n_steps)
    es_grid, entities = saliency_grid(steps, n_steps)
    es_span = saliency_span(es_grid)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.8, 6.75),
        gridspec_kw={"height_ratios": (n_options, len(entities))},
        squeeze=False,
    )

    ax = axes[0, 0]
    logit_cmap = plt.get_cmap("viridis").copy()
    logit_cmap.set_bad("white")
    ax.imshow(
        np.ma.masked_invalid(display),
        cmap=logit_cmap,
        vmin=0.0,
        vmax=1.0,
        aspect=0.5,
        interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([])

    ax2 = axes[1, 0]
    sal_cmap = plt.get_cmap("RdYlGn").copy()
    sal_cmap.set_bad("white")
    ax2.imshow(
        np.ma.masked_invalid(es_grid),
        cmap=sal_cmap,
        vmin=-max(es_span, 1e-6),
        vmax=max(es_span, 1e-6),
        aspect=0.5,
        interpolation="nearest",
    )
    ax2.set_yticks(range(len(entities)), entities, fontsize=10)
    ax2.set_xticks(range(n_steps))
    ax2.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_compare(
    path: Path,
    columns: list[tuple[str, dict]],
    rows: str = "both",
    ylabels: str = "left",
    n_steps: int = 16,
    shared_scale: bool = False,
    y_options: bool = False,
    dpi: int = 200,
) -> None:
    """Several explain runs side by side, each keeping the stacked layout.

    ``columns`` is a list of ``(title, payload)`` with payload a ``ckp_*_ep*.json``
    document from this script.  Every column draws the same panels as
    :func:`plot_episode_heatmaps` -- option logits above entity saliency -- so a
    column is exactly the figure that was drawn before, and the runs are
    compared by putting their columns next to each other.  The panels are
    height-ratioed by their row counts, which is what keeps the 40-option
    logits heatmap and the 5-entity saliency heatmap readable in one figure.

    ``rows``: ``"both"`` (default), ``"logits"`` or ``"saliency"`` to draw only
    one panel per column.  ``ylabels``: ``"left"`` (default) puts the entity
    names on the leftmost column only -- printing them under every column is
    what forces the large horizontal gap -- ``"all"`` or ``"none"``.  With
    ``y_options`` the leftmost column also labels its option rows.
    ``shared_scale`` uses one saliency colour scale for every column so the
    colours are comparable across runs.
    """
    if not columns:
        return
    if rows not in ("both", "logits", "saliency"):
        raise ValueError(f"rows must be both|logits|saliency, got {rows!r}")

    grids = []
    for title, payload in columns:
        steps = payload["explanations"]
        logit = logit_grid(steps, n_steps) if rows in ("both", "logits") else None
        sal, entities = (
            saliency_grid(steps, n_steps) if rows in ("both", "saliency") else (None, None)
        )
        grids.append((title, logit, sal, entities))

    n = len(grids)
    n_options = len(columns[0][1]["explanations"][0]["labels"])
    n_entities = len(grids[0][3] or [])
    span = (
        max(saliency_span(g[2]) for g in grids if g[2] is not None)
        if shared_scale
        else None
    )

    ratios = {
        "both": (n_options, n_entities),
        "logits": (n_options,),
        "saliency": (n_entities,),
    }[rows]
    # size the canvas so the cells keep aspect=0.5 exactly (0.3 in per step,
    # as in plot_episode_heatmaps); otherwise the axes are letterboxed inside
    # their boxes and the panels drift apart vertically
    cell = 0.3
    fig, axes = plt.subplots(
        len(ratios),
        n,
        figsize=(n_steps * cell * n, 0.5 * cell * sum(ratios)),
        gridspec_kw={
            "height_ratios": ratios,
            "wspace": 0.03,
            "hspace": 0.03,
        },
        squeeze=False,
    )
    logit_cmap = plt.get_cmap("viridis").copy()
    logit_cmap.set_bad("white")
    sal_cmap = plt.get_cmap("RdYlGn").copy()
    sal_cmap.set_bad("white")

    for col, (_, logit, sal, entities) in enumerate(grids):
        row = 0
        if logit is not None:
            ax = axes[row, col]
            ax.imshow(
                np.ma.masked_invalid(logit),
                cmap=logit_cmap,
                vmin=0.0,
                vmax=1.0,
                aspect=0.5,
                interpolation="nearest",
            )
            if col == 0 and y_options:
                labels = columns[col][1]["explanations"][0]["labels"]
                ax.set_yticks(range(len(labels)), labels, fontsize=5)
            else:
                ax.set_yticks([])
            ax.set_xticks(range(n_steps))
            ax.set_xticklabels([])
            row += 1
        if sal is not None:
            ax = axes[row, col]
            limit = max(span if span is not None else saliency_span(sal), 1e-6)
            ax.imshow(
                np.ma.masked_invalid(sal),
                cmap=sal_cmap,
                vmin=-limit,
                vmax=limit,
                aspect=0.5,
                interpolation="nearest",
            )
            if ylabels == "none" or (ylabels == "left" and col > 0):
                ax.set_yticks([])
            else:
                ax.set_yticks(range(len(entities)), entities, fontsize=9)
            ax.set_xticks(range(n_steps))
            ax.set_xticklabels([])

    # step numbers on the bottom panel of every column, exactly as in
    # plot_episode_heatmaps (the logits heatmap above it has none)
    for col in range(n):
        bottom = axes[-1, col]
        bottom.set_xticks(range(n_steps))
        bottom.set_xticklabels([str(i) for i in range(n_steps)], fontsize=9)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_graph(path: Path, graph: Graph, explanation: StepExplanation) -> None:
    """The graph itself, with the saliency drawn onto nodes and condition edges."""
    G = nx.MultiDiGraph()
    entity_keys, comp_keys, option_keys = (
        graph.ns_entity.keys,
        graph.ns_comp.keys,
        graph.ns_option.keys,
    )
    for key in entity_keys:
        G.add_node(key, kind="entity")
    for key in comp_keys:
        G.add_node(key, kind="comp")
    for key in option_keys:
        G.add_node(key, kind="option")

    importance = {key: 0.0 for key in G.nodes}
    for key, value in explanation.row_saliency.items():
        importance[key] = abs(value)
    top = max(importance.values()) or 1.0
    for key in G.nodes:
        importance[key] /= top

    edge_importance: dict[tuple[str, str], float] = {}
    for comp, entity, value in explanation.edges:
        edge_importance[(comp, entity)] = max(
            edge_importance.get((comp, entity), 0.0), value
        )
    top_edge = max(edge_importance.values(), default=0.0) or 1.0

    for src, dst in graph.es_condition.edges:
        G.add_edge(comp_keys[src], entity_keys[dst], kind="condition")
    for src, dst in graph.es_summary.edges:
        G.add_edge(entity_keys[src], option_keys[dst], kind="summary")
    for src, dst in graph.es_translation.edges:
        G.add_edge(entity_keys[src], entity_keys[dst], kind="translation")

    shells = [option_keys, comp_keys, entity_keys]
    pos = nx.shell_layout(G, nlist=shells, scale=3.0)

    entity_nodes = [k for k in entity_keys if k in G]
    comp_nodes = [k for k in comp_keys if k in G]
    option_nodes = [k for k in option_keys if k in G]

    fig, ax = plt.subplots(figsize=(15, 12))
    nx.draw_networkx_nodes(
        G, pos, nodelist=comp_nodes, node_color="lightgrey", node_size=120, ax=ax
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=entity_nodes,
        node_color=[importance.get(k, 0.0) for k in entity_nodes],
        cmap="Oranges",
        node_size=[300 + 2500 * importance.get(k, 0.0) for k in entity_nodes],
        edgecolors="black",
        linewidths=0.4,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=option_nodes,
        node_color=[
            (
                "gold"
                if k == explanation.chosen_label
                else ("lightgrey" if explanation.gated[i] else "tab:green")
            )
            for i, k in enumerate(option_keys)
        ],
        node_size=500,
        ax=ax,
    )
    for kind, colour in (
        ("condition", "dimgrey"),
        ("summary", "orange"),
        ("translation", "red"),
    ):
        for u, v, key in G.edges(keys=True):
            if G.edges[u, v, key]["kind"] != kind:
                continue
            width = (
                0.4 + 4.0 * edge_importance.get((u, v), 0.0) / top_edge
                if kind == "condition"
                else 0.6
            )
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                edge_color=colour,
                width=width,
                alpha=0.75 if kind == "condition" else 0.35,
                arrows=True,
                arrowsize=8,
                node_size=300,
                ax=ax,
            )
    nx.draw_networkx_labels(G, pos, {k: k[:20] for k in G.nodes}, font_size=6, ax=ax)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def explain_step(
    heca: Heca,
    graph: Graph,
    data,
    step: int,
    budget: float,
    methods: str,
    top_edges: int,
) -> StepExplanation | None:
    """Attribute the greedy choice of one step; None if nothing is selectable."""
    if getattr(data, "memory", None) is None:
        data.memory = {}
    gated = [bool(g) for g in data["option"].gated.tolist()]
    if not any(gated):
        return None
    with torch.no_grad():
        out = heca.learner.network(data)
        logits, value = out.logits[0].detach(), float(out.value.item())
    chosen = int(torch.argmax(torch.where(torch.isfinite(logits), logits, -torch.inf)))

    entity_rows, comp_rows = row_index(data, graph)
    base = float(logits[chosen])
    saliency = (
        gradient_saliency(heca.learner.network, graph, data, chosen, entity_rows)
        if methods in ("saliency", "both")
        else {
            "row_saliency": {},
            "entity_saliency": {},
            "edges": [],
            "option_saliency": [0.0] * len(gated),
            "budget_saliency": 0.0,
            "gate_saliency": 0.0,
        }
    )
    occluded = (
        occlusion(
            heca.learner.network, graph, data, chosen, base, entity_rows, comp_rows
        )
        if methods in ("occlusion", "both")
        else {name: {"current": 0.0, "goal": 0.0, "comps": 0.0} for name in entity_rows}
    )

    return StepExplanation(
        step=step,
        budget=budget,
        chosen=chosen,
        chosen_label=graph.ns_option.keys[chosen],
        logits=[float(v) for v in logits],
        labels=list(graph.ns_option.keys),
        gated=gated,
        value=value,
        row_saliency=saliency["row_saliency"],
        entity_saliency=saliency["entity_saliency"],
        edges=saliency["edges"][:top_edges],
        option_saliency=list(saliency["option_saliency"]),
        budget_saliency=saliency["budget_saliency"],
        gate_saliency=saliency["gate_saliency"],
        occluded=occluded,
    )


def as_payload(explanation: StepExplanation) -> dict:
    return {
        "step": explanation.step,
        "budget": explanation.budget,
        "chosen": explanation.chosen,
        "chosen_label": explanation.chosen_label,
        "value": explanation.value,
        "logits": explanation.logits,
        "labels": explanation.labels,
        "gated": explanation.gated,
        "entity_saliency": explanation.entity_saliency,
        "row_saliency": explanation.row_saliency,
        "option_saliency": explanation.option_saliency,
        "budget_saliency": explanation.budget_saliency,
        "gate_saliency": explanation.gate_saliency,
        "occlusion": explanation.occluded,
        "top_edges": [
            {"comp": comp, "entity": entity, "saliency": value}
            for comp, entity, value in explanation.edges
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tag",
        nargs="+",
        required=True,
        help="run dir(s) under data/network/standard: one per trained seed",
    )
    ap.add_argument("--ckp", default="latest", help="file name, or 'latest'")
    ap.add_argument("--network", default="x0", help="config name from conf.networks")
    ap.add_argument("--scene", default="scene0")
    ap.add_argument(
        "--episodes", type=int, default=3, help="random episodes to explain"
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=3,
        help="explained steps per episode, counted from the start; 0 = all",
    )
    ap.add_argument("--seed", type=int, default=0, help="env RNG seed")
    ap.add_argument(
        "--mode",
        choices=("randomized", "task"),
        default="randomized",
        help="scene mode to explain: the training distribution, or one fixed "
        "task setup (repeatable across networks via --seed and --task-id)",
    )
    ap.add_argument(
        "--task-id",
        type=int,
        default=1,
        help="with --mode task: the ogbench task id (1-based, default 1)",
    )
    ap.add_argument(
        "--method",
        choices=("saliency", "occlusion", "both"),
        default="both",
        help="which attributions to compute",
    )
    ap.add_argument(
        "--top-edges", type=int, default=10, help="condition edges kept per step"
    )
    ap.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the per-step figures next to the JSON",
    )
    ap.add_argument("--gt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--virtual", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    heca = Heca.get(
        Heca.Config(
            experts=find_scene_models(args.scene),
            learner=PPO.Config(
                tag=args.tag[0],
                network=conf.networks.get(args.network),
                wandb=logger.WandBConfig(enabled=False),
                max_update=0,
                lr_annealing=False,
                normalize_rewards=False,
            ),
            inference=True,
            virtual=args.virtual,
            reload=False,
            use_gt=args.gt,
            smode=SubgoalMode.BOTH,
        )
    )
    heca.scene.set_mode(args.mode)
    graph = heca.graph
    captured = capture_build(heca)

    logger.info(
        f"{len(args.tag)} tag(s), net={args.network}, {args.episodes} episode(s), "
        f"{args.steps or 'all'} step(s) each, method={args.method}, "
        f"mode={args.mode}" + (f", task_id={args.task_id}" if args.mode == "task" else "")
    )
    if not heca.learner.network.cfg.use_statistics:
        logger.warning(
            "use_statistics=False: the option and state encoders return zeros, so "
            "the option, gate and budget saliency of every step is 0 by "
            "construction.  The entity, component and condition-edge attributions "
            "are unaffected; run a config with statistics on to explain those."
        )
    started = time.perf_counter()

    for tag in args.tag:
        run_dir = RUN_ROOT / tag
        path = find_checkpoint(run_dir, args.ckp)
        checkpoint = torch.load(path, map_location=hardware.device, weights_only=False)
        missing, unexpected = heca.learner.network.upgrade(checkpoint["network"])
        if missing or unexpected:
            logger.warning(
                f"  {tag}: checkpoint mismatch, {len(missing)} missing, "
                f"{len(unexpected)} unexpected parameter(s)"
            )
        heca.learner.eval()

        out_dir = run_dir / "explain" / f"{path.stem}_{args.network}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for episode in range(args.episodes):
            if args.mode == "task":
                heca.scene.task_id = args.task_id
            heca.scene.seed = args.seed + episode
            x, y = heca.sample()
            steps: list[dict] = []
            z, fb, finished = x, None, False
            step = 0
            while True:
                z, fb, finished = heca.step(z, y, fb.budget if fb else 0.0)
                if graph.dead_end:
                    logger.warning(f"  episode {episode}: no option applies, skipped")
                    steps = []
                    break
                data = captured["data"]
                if args.steps == 0 or step < args.steps:
                    explanation = explain_step(
                        heca,
                        graph,
                        data,
                        step,
                        fb.budget,
                        args.method,
                        args.top_edges,
                    )
                    if explanation is not None:
                        steps.append(as_payload(explanation))
                        if args.plots:
                            plot_step(
                                out_dir
                                / f"{path.stem}_ep{episode:02d}_step{step:02d}.png",
                                explanation,
                            )
                            plot_graph(
                                out_dir
                                / f"{path.stem}_ep{episode:02d}_step{step:02d}_graph.png",
                                graph,
                                explanation,
                            )
                        best = sorted(
                            explanation.occluded.items(),
                            key=lambda item: -abs(item[1]["current"]),
                        )[:3]
                        logger.info(
                            f"  ep{episode} step{step}: {explanation.chosen_label} "
                            f"(value {explanation.value:.2f}, budget {explanation.budget:.0f}), "
                            f"state-driven: "
                            + ", ".join(f"{n} {v['current']:+.3f}" for n, v in best)
                            + (
                                f", top edge {explanation.edges[0][0]}->{explanation.edges[0][1]} "
                                f"{explanation.edges[0][2]:.3g}"
                                if explanation.edges
                                else ""
                            )
                        )
                step += 1
                if fb.end or finished or step > 64:
                    break

            if args.plots and steps:
                plot_episode_heatmaps(
                    out_dir / f"{path.stem}_ep{episode:02d}_heatmap.png", steps
                )

            payload = {
                "tag": tag,
                "checkpoint": path.name,
                "scene": args.scene,
                "network": args.network,
                "mode": args.mode,
                "task_id": args.task_id if args.mode == "task" else None,
                "seed": heca.scene.seed,
                "episode": episode,
                "success": bool(fb.success) if fb else False,
                "truncated": bool(fb.truncated) if fb else True,
                "steps_taken": step,
                "explained_steps": len(steps),
                "method": args.method,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "explanations": steps,
            }
            out = out_dir / f"{path.stem}_ep{episode:02d}.json"
            out.write_text(json.dumps(payload, indent=2) + "\n")
            logger.info(
                f"  episode {episode}: {step} step(s), "
                f"success={payload['success']} truncated={payload['truncated']}"
                f" -> {out}"
            )

    logger.info(f"done in {fmt_duration(time.perf_counter() - started)}")


if __name__ == "__main__":
    main()
