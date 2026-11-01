"""Bar plot of every sampled state readout: accuracy per entity and true state.

One bar group per entity and recorded state, one bar per model, so a group reads
"how often did each model answer this entity right, on the frames whose state was
that value". Reads the states-<model>.json the samplers wrote.

    python -m scripts.e06_plot_states --scene scene0
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heca.misc import logger
from heca.scenes.scene import Scene
from scripts.common.scenes import scene_config


def load(scene_dir: Path) -> dict[str, dict]:
    """model -> {(label, state): [correct, total]} over every sample file present."""
    counts: dict[str, dict] = {}
    for path in sorted(scene_dir.glob("states-*.json")):
        payload = json.loads(path.read_text())
        model_counts = counts.setdefault(payload["model"], {})
        for record in payload["records"]:
            cell = model_counts.setdefault((record["label"], int(record["truth"])), [0, 0])
            cell[0] += int(record["predicted"] == record["truth"])
            cell[1] += 1
    return counts


def plot(counts: dict[str, dict], out: Path, scene: str) -> Path | None:
    models = list(counts)
    entities = sorted({label for model in models for label, _ in counts[model]})
    states = sorted({state for model in models for _, state in counts[model]})
    if not models or not entities:
        logger.warning(f"{scene}: no state samples to plot")
        return None

    palette = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(1.8 * len(entities) + 2.6, 4.6))
    step = 0.8 / len(states)
    bar = 0.85 * step / len(models)
    for e_index, label in enumerate(entities):
        for s_index, state in enumerate(states):
            centre = e_index + (s_index - (len(states) - 1) / 2) * step
            for m_index, model in enumerate(models):
                correct, total = counts[model].get((label, state), (0, 0))
                rate = correct / total if total else 0.0
                x = centre + (m_index - (len(models) - 1) / 2) * bar
                axis.bar(
                    x,
                    rate,
                    width=bar,
                    color=palette(m_index % 10),
                    edgecolor="black",
                    linewidth=0.4,
                    label=model if (e_index == 0 and s_index == 0) else None,
                )
                axis.text(x, rate + 0.02, f"{100 * rate:.0f}", ha="center",
                          va="bottom", fontsize=6, rotation=90)
            axis.text(centre, -0.06, f"state {state}", ha="center", va="top",
                      fontsize=8, transform=axis.get_xaxis_transform())

    axis.set_xticks(range(len(entities)), entities)
    axis.tick_params(axis="x", pad=18)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("success rate over the frames in that state")
    axis.set_title(f"{scene}: state readout vs ground truth")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False, ncols=len(models), loc="upper center",
                bbox_to_anchor=(0.5, -0.14), fontsize=8)
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)
    logger.info(f"wrote {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene0")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    scene = Scene.get(scene_config(args.scene))
    scene_dir = Path(scene.save_dir(scene.cfg))
    out = Path(args.out) if args.out else scene_dir / "state-accuracy.png"
    plot(load(scene_dir), out, args.scene)


if __name__ == "__main__":
    main()
