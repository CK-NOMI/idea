#!/usr/bin/env python3
"""Reproducible synthetic audit for team-far/R49 HVC-SD.

This is a linearized toy model, not a Transformer, not real HFL, and not a
production speculative-decoding benchmark. Requires Python >=3.10 and NumPy.
"""
import json
import platform
import sys
import numpy as np


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def eal2(theta, phi, q_maps, q_bias, p_prefix_maps, p_suffix_maps, p_bias):
    """Exact two-token sampling EAL for one context and finite vocabulary."""
    q_logits = np.einsum("rvn,n->rv", q_maps, theta) + q_bias
    p_logits = (
        np.einsum("rvn,n->rv", p_prefix_maps, theta)
        + np.einsum("rvk,k->rv", p_suffix_maps, phi)
        + p_bias
    )
    q, p = softmax(q_logits), softmax(p_logits)
    root_overlap = np.minimum(p[0], q[0])
    child_overlap = np.minimum(p[1:], q[1:]).sum(axis=1)
    eal = np.sum(root_overlap * (1.0 + child_overlap))
    return float(eal), p, q


def centered_jacobian(maps):
    """Stack centered-logit Jacobians; removes softmax's all-ones gauge."""
    vocab = maps.shape[1]
    center = np.eye(vocab) - np.ones((vocab, vocab)) / vocab
    return np.vstack([center @ block for block in maps])


def null_projector(matrix, tol=1e-9):
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    null_basis = vh[rank:].T
    return null_basis @ null_basis.T, null_basis.shape[1], singular_values


def central_difference(fun, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (fun(xp) - fun(xm)) / (2.0 * eps)
    return grad


def one_trial(seed, n=12, suffix_rank=2, vocab=2, alpha=0.1):
    rng = np.random.default_rng(seed)
    clients = 4
    regions = ((0, 1), (2, 3))
    branches = vocab + 1
    theta0 = np.zeros(n)

    client_data, js_list, jd_list, b_list, pc_list, client_dims = [], [], [], [], [], []
    for _ in range(clients):
        q_maps = rng.normal(size=(branches, vocab, n)) * 0.22
        p_prefix_maps = rng.normal(size=(branches, vocab, n)) * 0.22
        p_suffix_maps = rng.normal(size=(branches, vocab, suffix_rank)) * 0.22
        q_bias = rng.normal(size=(branches, vocab)) * 0.25
        p_bias = rng.normal(size=(branches, vocab)) * 0.25
        phi0 = rng.normal(size=suffix_rank) * 0.15

        def eal(th, ph):
            return eal2(
                th, ph, q_maps, q_bias, p_prefix_maps, p_suffix_maps, p_bias
            )[0]

        # Central-difference gradients of the exact two-token EAL.
        g_prefix = central_difference(lambda th: eal(th, phi0), theta0)
        g_suffix = central_difference(lambda ph: eal(theta0, ph), phi0)

        js = centered_jacobian(p_prefix_maps)
        jd = centered_jacobian(p_suffix_maps)
        b = (np.eye(branches * vocab) - jd @ np.linalg.pinv(jd)) @ js
        pc, dim, _ = null_projector(b)
        # Derivative along the minimum-norm suffix-compensated path:
        # u(phi) = -J_d^+ J_s v.
        g_comp = g_prefix - js.T @ np.linalg.pinv(jd).T @ g_suffix

        client_data.append({
            "q_maps": q_maps, "q_bias": q_bias,
            "p_prefix_maps": p_prefix_maps, "p_suffix_maps": p_suffix_maps,
            "p_bias": p_bias, "phi0": phi0, "g_comp": g_comp,
        })
        js_list.append(js)
        jd_list.append(jd)
        b_list.append(b)
        pc_list.append(pc)
        client_dims.append(dim)

    edge_projectors, edge_dims = [], []
    for region in regions:
        pe, dim, _ = null_projector(np.vstack([b_list[c] for c in region]))
        edge_projectors.append(pe)
        edge_dims.append(dim)
    p0, cloud_dim, _ = null_projector(np.vstack(b_list))

    gbar = np.mean([d["g_comp"] for d in client_data], axis=0)
    updates = {
        "global_unconstrained": [],
        "edge_unconstrained": [],
        "client_unconstrained": [],
        "global_safe": [],
        "edge_safe": [],
        "hierarchy_safe": [],
    }
    for c in range(clients):
        edge = 0 if c < 2 else 1
        ge = np.mean([client_data[j]["g_comp"] for j in regions[edge]], axis=0)
        v_global = p0 @ gbar
        v_edge = (edge_projectors[edge] - p0) @ ge
        v_client = (pc_list[c] - edge_projectors[edge]) @ client_data[c]["g_comp"]

        updates["global_unconstrained"].append(gbar.copy())
        updates["edge_unconstrained"].append(ge.copy())
        updates["client_unconstrained"].append(client_data[c]["g_comp"].copy())
        updates["global_safe"].append(v_global)
        updates["edge_safe"].append(v_global + v_edge)
        updates["hierarchy_safe"].append(v_global + v_edge + v_client)

    metrics = {}
    for method, raw_updates in updates.items():
        v_matrix = np.asarray(raw_updates)
        # Match aggregate early-parameter movement across the four client copies.
        v_matrix *= alpha / (np.linalg.norm(v_matrix) + 1e-12)
        delta_eal, formal_kl, residual, compensation = [], [], [], []
        for c, v in enumerate(v_matrix):
            d = client_data[c]
            js, jd = js_list[c], jd_list[c]
            u = -np.linalg.pinv(jd) @ (js @ v)
            old_eal, p_old, _ = eal2(
                theta0, d["phi0"], d["q_maps"], d["q_bias"],
                d["p_prefix_maps"], d["p_suffix_maps"], d["p_bias"]
            )
            new_eal, p_new, _ = eal2(
                theta0 + v, d["phi0"] + u, d["q_maps"], d["q_bias"],
                d["p_prefix_maps"], d["p_suffix_maps"], d["p_bias"]
            )
            delta_eal.append(new_eal - old_eal)
            formal_kl.append(float(np.sum(
                p_old * np.log((p_old + 1e-15) / (p_new + 1e-15))
            )))
            residual.append(float(np.linalg.norm(js @ v + jd @ u)))
            compensation.append(float(np.linalg.norm(u)))

        metrics[method] = {
            "mean_delta_eal": float(np.mean(delta_eal)),
            "sd_delta_eal": float(np.std(delta_eal)),
            "mean_formal_kl": float(np.mean(formal_kl)),
            "max_linearized_residual": float(np.max(residual)),
            "mean_suffix_compensation_norm": float(np.mean(compensation)),
        }

    return {
        "dimensions": {
            "client_mean": float(np.mean(client_dims)),
            "edge_mean": float(np.mean(edge_dims)),
            "cloud": int(cloud_dim),
        },
        "metrics": metrics,
    }


def rank_trial(seed, n=24, m=6, suffix_rank=3, clients=4,
               regions=((0, 1), (2, 3)), contexts=1):
    rng = np.random.default_rng(seed)
    b_list, client_dims = [], []
    for _ in range(clients):
        js = np.vstack([
            rng.normal(size=(m, n)) / np.sqrt(n) for _ in range(contexts)
        ])
        jd = np.vstack([
            rng.normal(size=(m, suffix_rank)) / np.sqrt(suffix_rank)
            for _ in range(contexts)
        ])
        b = (np.eye(contexts * m) - jd @ np.linalg.pinv(jd)) @ js
        _, dim, _ = null_projector(b)
        b_list.append(b)
        client_dims.append(dim)

    edge_dims = []
    for region in regions:
        _, dim, _ = null_projector(np.vstack([b_list[c] for c in region]))
        edge_dims.append(dim)
    _, cloud_dim, _ = null_projector(np.vstack(b_list))
    return float(np.mean(client_dims)), float(np.mean(edge_dims)), int(cloud_dim)


def main():
    seeds = 100
    toy_rows = [one_trial(seed) for seed in range(seeds)]
    methods = list(toy_rows[0]["metrics"])
    toy_summary = {}
    for method in methods:
        fields = list(toy_rows[0]["metrics"][method])
        toy_summary[method] = {}
        for field in fields:
            values = np.array([row["metrics"][method][field] for row in toy_rows])
            toy_summary[method][field + "_mean"] = float(values.mean())
            toy_summary[method][field + "_sd"] = float(values.std())
            if "residual" in field:
                toy_summary[method][field + "_max"] = float(values.max())

    rank_summary = {}
    for suffix_rank in (3, 6, 12):
        for contexts in (1, 2, 3, 4):
            rows = [
                rank_trial(seed, suffix_rank=suffix_rank, contexts=contexts)
                for seed in range(20)
            ]
            arr = np.asarray(rows)
            rank_summary[f"suffix_rank_{suffix_rank}_contexts_{contexts}"] = {
                "client_null_dimension_mean": float(arr[:, 0].mean()),
                "edge_null_dimension_mean": float(arr[:, 1].mean()),
                "cloud_null_dimension_mean": float(arr[:, 2].mean()),
                "rank_seeds": 20,
            }

    result = {
        "evidence": "synthetic linearized CPU experiment; not a Transformer, HFL deployment, GPU run, or real SD serving benchmark",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "two_token_eal_experiment": {
            "seeds": seeds,
            "clients": 4,
            "regions": 2,
            "vocabulary": 2,
            "early_parameter_dimension": 12,
            "suffix_compensator_rank": 2,
            "matched_total_early_update_norm": 0.1,
            "summary": toy_summary,
        },
        "rank_collapse_stress": {
            "seeds_per_setting": 20,
            "early_parameter_dimension": 24,
            "formal_logit_dimension_per_context": 6,
            "clients": 4,
            "regions": 2,
            "summary": rank_summary,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
