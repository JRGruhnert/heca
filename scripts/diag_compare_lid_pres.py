"""Compare the fitted lid pre-conditions scene2 vs scene4 (why is scene4 valid
and scene2 not), from demo data, the live sampler and the fitted StepMix models.

Usage: python scripts/diag_compare_lid_pres.py
"""

import h5py
import numpy as np

from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models

META = np.array([0.425, 0.0, 0.0])
SCALE = 10.0


def comps_world(con_or_model, label):
    """Return per-component (weight, pos_world, sigma_xyz_mm, aa) of a StepMix."""
    p = con_or_model.models[label].get_parameters()
    means = p["measurement"]["pose"]["means"]
    cov = p["measurement"]["pose"]["covariances"]
    w = p["weights"]
    out = []
    for k in range(len(w)):
        pos_w = means[k][:3] / SCALE + META
        sig_mm = np.sqrt(np.maximum(cov[k], 1e-15))[:3] / SCALE * 1000.0
        out.append((float(w[k]), pos_w, sig_mm, means[k][3:6].copy()))
    return out


def score_fail(ent, model, value):
    """Return (ok, best_k, per-dim z) replicating Entity.score_single."""
    sample = ent.model_value(value)
    p = model.get_parameters()
    import math
    from scipy.stats import chi2, norm
    pose, state = sample[:-1], int(sample[-1])
    pis = p["measurement"]["state"]["pis"]
    means = p["measurement"]["pose"]["means"]
    vars_ = p["measurement"]["pose"]["covariances"]
    w = p["weights"]
    best_k, best_post = -1, -np.inf
    for k in range(len(w)):
        var = np.maximum(vars_[k], 1e-15)
        post = np.log(w[k]) - 0.5 * np.sum(
            np.log(2 * np.pi * var) + (pose - means[k]) ** 2 / var
        )
        if post > best_post:
            best_k, best_post = k, post
    zd = np.abs(pose - means[best_k]) / np.sqrt(
        np.maximum(vars_[best_k], 1e-15)
    )
    chi_sqrt = math.sqrt(chi2.ppf(ent.cfg.z_quantile_joint, len(pose)))
    z_dim_cap = float(norm.ppf(0.5 + ent.cfg.z_quantile_dim / 2.0))
    ok = (
        float(np.sqrt((zd**2).sum())) <= chi_sqrt
        and bool(np.all(zd <= z_dim_cap))
        and state == int(np.argmax(pis[best_k]))
    )
    return ok, best_k, zd, z_dim_cap


def demo_stats(tag, skill):
    cfg = next(ac for ac in find_scene_models(tag) if ac.tag == skill)
    a = ExpertModel.get(cfg, auto_load=False)
    with h5py.File(a.load_dir(cfg) / "demos.h5", "r") as f:
        demo = np.asarray(f["demo"][:])
        segs = np.where(np.diff(demo) != 0)[0] + 1
        starts = list(np.concatenate([[0], segs]))
        ends = list(np.concatenate([segs - 1, [len(demo) - 1]]))
        pos = np.asarray(f["heca_lid0_pos"][:])
        rot = np.asarray(f["heca_lid0_rot"][:])
        sp, ep = pos[starts], pos[ends]

        def yaw(q):
            q = q / np.linalg.norm(q)
            return float(
                np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            )

        sy = np.array([yaw(rot[s]) for s in starts])

        def st(v):
            return (
                f"xy x[{v[:,0].min():.3f},{v[:,0].max():.3f}] "
                f"y[{v[:,1].min():.3f},{v[:,1].max():.3f}] | "
                f"z mean={v[:,2].mean():.4f} std={v[:,2].std():.5f} "
                f"[{v[:,2].min():.4f},{v[:,2].max():.4f}] | "
                f"yaw std={sy.std():.3f}"
            )

        return len(starts), st(sp), st(ep)


def run(tag):
    scene = Scene.get(find_scene_config(tag), auto_load=False)
    print(f"\n================ {tag} ================")

    # live sampler: raw env joint (base) z + handle site z
    env = scene.env
    z_base, z_site = [], []
    for _ in range(40):
        scene.sample_task()
        q = env._data.joint("box_lid_joint_0").qpos.copy()
        z_base.append(q[2])
        sid = env._model.site("box_lid_handle_center_0").id
        z_site.append(env._data.site_xpos[sid][2])
    print(
        f"LIVE env lid base z={np.mean(z_base):.4f}  "
        f"handle-site z={np.mean(z_site):.4f} (std {np.std(z_site):.5f})"
    )

    for skill in ("lid0_base_base", "lid0_base_box0"):
        nseg, sst, est = demo_stats(tag, skill)
        print(f"\n-- {skill} ({nseg} demo segments)")
        print(f"   demo START: {sst}")
        print(f"   demo END  : {est}")

        cfg = next(ac for ac in find_scene_models(tag) if ac.tag == skill)
        a = ExpertModel.get(cfg, auto_load=False)
        a.use_gt(True)
        a.load()
        ent = a.entities["lid0"]
        for which, con in (("PRE", a.conditions.pre), ("POST", a.conditions.post)):
            print(f"   fitted {which} lid0 components (world units):")
            for w, pw, sig, aa in comps_world(con, "lid0"):
                print(
                    f"     w={w:.3f} pos={np.round(pw,4)} "
                    f"sigma_xyz(mm)={np.round(sig,2)} aa={np.round(aa,3)}"
                )

        # acceptance over live starts + failing dims
        ok_n, fails = 0, []
        for _ in range(60):
            (x, _), _ = scene.sample_task()
            ok, k, zd, cap = score_fail(
                ent, a.conditions.pre.models["lid0"], x["lid0"].value
            )
            ok_n += int(ok)
            if not ok:
                fails.append((int(zd.argmax()), float(zd.max()), k))
        print(
            f"   PRE acceptance on live starts: {ok_n}/60  "
            f"(fail worst-dim distribution: "
            f"{ {d: fails.count([d, max(z for dd,mz,k in fails if dd==d), 0]) if False else d for d in set(f[0] for f in fails)} }"
        )
        from collections import Counter
        cnt = Counter(f[0] for f in fails)
        print(f"      failing dims: {dict(cnt)}  (0=x,1=y,2=z,3..5=rot)  "
              f"worst z examples: {sorted(f[1] for f in fails)[-3:]}")

    scene.close()


if __name__ == "__main__":
    for tag in ("scene2", "scene4"):
        run(tag)
