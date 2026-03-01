"""
Interactive 2D Matrix Transformation Viewer
============================================
Displays how a 2x2 matrix transforms the plane:
  - original axes and (optionally) original grid lines
  - transformed grid lines and axes
  - original and transformed standard basis vectors
  - a user-selected vector and its image under the transformation

Usage
-----
    from shared_utils.matrix_transform_viewer import visualize_transformation
    import numpy as np

    M = np.array([[2, 1],
                  [0, 1]])
    visualize_transformation(M, show_old_grid=True)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.widgets import TextBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arrow(ax, start, end, color, lw=2, zorder=5, alpha=1.0):
    """Add a FancyArrowPatch to *ax* and return it (supports .remove())."""
    patch = FancyArrowPatch(
        posA=tuple(start),
        posB=tuple(end),
        arrowstyle="->",
        color=color,
        linewidth=lw,
        mutation_scale=15,
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def _fmt(x):
    """Format a float without trailing decimal zeros (e.g. 2.0 → '2', 1.5 → '1.5')."""
    return f"{x:g}"


def _view_limits(M, grid):
    """Return (xlim, ylim) that enclose both original and transformed grid corners."""
    corners = np.array([[x, y] for x in (-grid, grid) for y in (-grid, grid)])
    transformed = corners @ M.T
    all_pts = np.vstack([corners, transformed])
    pad = 0.8
    return (
        (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad),
        (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad),
    )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def visualize_transformation(matrix, show_old_grid=True):
    """
    Visualize a 2D linear transformation interactively.

    Parameters
    ----------
    matrix : array-like, shape (2, 2)
        The 2x2 transformation matrix.
    show_old_grid : bool, optional (default=True)
        Whether to display the original (pre-transform) grid lines.
    """
    M = np.asarray(matrix, dtype=float)
    if M.shape != (2, 2):
        raise ValueError("matrix must be shape (2, 2).")

    GRID = 5  # grid lines run from -GRID to +GRID

    def T(v):
        return M @ np.asarray(v, dtype=float)

    # ------------------------------------------------------------------
    # Figure / axes layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 9))

    # Title
    fig.text(
        0.5,
        0.97,
        "2D Matrix Transformation",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    # Matrix display – aligned, no trailing decimal zeros
    a, b = _fmt(M[0, 0]), _fmt(M[0, 1])
    c, d = _fmt(M[1, 0]), _fmt(M[1, 1])
    wl = max(len(a), len(c))  # left-column width
    wr = max(len(b), len(d))  # right-column width
    row1 = f"⎡ {a:>{wl}}  {b:>{wr}} ⎤"
    row2 = f"⎣ {c:>{wl}}  {d:>{wr}} ⎦"
    fig.text(
        0.5,
        0.93,
        f"M = {row1}\n    {row2}",
        ha="center",
        va="top",
        fontsize=11,
        family="monospace",
    )

    ax = fig.add_axes([0.08, 0.26, 0.84, 0.62])

    xlim, ylim = _view_limits(M, GRID)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)

    # ------------------------------------------------------------------
    # Static elements – drawn once
    # ------------------------------------------------------------------

    # Original grid (dashed light grey)
    if show_old_grid:
        for k in range(-GRID, GRID + 1):
            ax.plot([-GRID, GRID], [k, k], color="#cccccc", lw=0.6, ls="--", zorder=1)
            ax.plot([k, k], [-GRID, GRID], color="#cccccc", lw=0.6, ls="--", zorder=1)

    # Original axes (solid grey)
    ax.plot([-GRID, GRID], [0, 0], color="#888888", lw=1.5, zorder=2)
    ax.plot([0, 0], [-GRID, GRID], color="#888888", lw=1.5, zorder=2)

    # Transformed grid lines (light blue)
    for k in range(-GRID, GRID + 1):
        p1, p2 = T([-GRID, k]), T([GRID, k])
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="#4fc3f7",
            lw=0.8,
            alpha=0.55,
            zorder=2,
        )
        p1, p2 = T([k, -GRID]), T([k, GRID])
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="#4fc3f7",
            lw=0.8,
            alpha=0.55,
            zorder=2,
        )

    # Transformed axes (blue)
    p1, p2 = T([-GRID, 0]), T([GRID, 0])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0288d1", lw=1.8, zorder=3)
    p1, p2 = T([0, -GRID]), T([0, GRID])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0288d1", lw=1.8, zorder=3)

    # Original standard basis vectors
    _arrow(ax, [0, 0], [1, 0], "#e53935", lw=2.5, zorder=6)
    _arrow(ax, [0, 0], [0, 1], "#43a047", lw=2.5, zorder=6)
    ax.text(1.08, 0.08, "î", color="#e53935", fontsize=13, fontweight="bold")
    ax.text(0.08, 1.08, "ĵ", color="#43a047", fontsize=13, fontweight="bold")

    # Transformed standard basis vectors
    e1t, e2t = T([1, 0]), T([0, 1])
    _arrow(ax, [0, 0], e1t, "#b71c1c", lw=2.5, zorder=6)
    _arrow(ax, [0, 0], e2t, "#1b5e20", lw=2.5, zorder=6)
    ax.text(
        e1t[0] + 0.12,
        e1t[1] + 0.12,
        "T(î)",
        color="#b71c1c",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        e2t[0] + 0.12,
        e2t[1] + 0.12,
        "T(ĵ)",
        color="#1b5e20",
        fontsize=12,
        fontweight="bold",
    )

    # Legend
    legend_handles = [
        Line2D([0], [0], color="#888888", lw=1.5, ls="--", label="Original axes"),
        Line2D([0], [0], color="#0288d1", lw=1.8, label="Transformed axes"),
        Line2D([0], [0], color="#e53935", lw=2.5, label="î  (original)"),
        Line2D([0], [0], color="#43a047", lw=2.5, label="ĵ  (original)"),
        Line2D([0], [0], color="#b71c1c", lw=2.5, label="T(î)"),
        Line2D([0], [0], color="#1b5e20", lw=2.5, label="T(ĵ)"),
        Line2D([0], [0], color="#ff8f00", lw=2.5, label="v  (selected)"),
        Line2D([0], [0], color="#e65100", lw=2.5, label="Mv (transformed)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.85)

    # ------------------------------------------------------------------
    # Dynamic elements – redrawn on text submission
    # ------------------------------------------------------------------
    dyn_patches: list = []
    dyn_texts: list = []
    current = {"vx": 1.0, "vy": 1.0}

    def update():
        for p in dyn_patches:
            p.remove()
        for t in dyn_texts:
            t.remove()
        dyn_patches.clear()
        dyn_texts.clear()

        vx, vy = current["vx"], current["vy"]
        v = np.array([vx, vy])
        mv = T(v)
        off = 0.15

        if np.linalg.norm(v) > 1e-9:
            dyn_patches.append(_arrow(ax, [0, 0], v, "#ff8f00", lw=3, zorder=8))
            dyn_texts.append(
                ax.text(
                    v[0] + off,
                    v[1] + off,
                    "v",
                    color="#ff8f00",
                    fontsize=11,
                    fontweight="bold",
                    zorder=9,
                )
            )

        if np.linalg.norm(mv) > 1e-9:
            dyn_patches.append(_arrow(ax, [0, 0], mv, "#e65100", lw=3, zorder=8))
            dyn_texts.append(
                ax.text(
                    mv[0] + off,
                    mv[1] + off,
                    "Mv",
                    color="#e65100",
                    fontsize=11,
                    fontweight="bold",
                    zorder=9,
                )
            )

        # Fixed info box – never overlaps
        info = f"v  = ({vx:.2f},  {vy:.2f})\n" f"Mv = ({mv[0]:.2f},  {mv[1]:.2f})"
        dyn_texts.append(
            ax.text(
                0.02,
                0.04,
                info,
                transform=ax.transAxes,
                color="#222222",
                fontsize=10,
                fontweight="bold",
                zorder=9,
                verticalalignment="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor="#cccccc",
                    alpha=0.85,
                ),
            )
        )

        fig.canvas.draw_idle()

    def on_submit_vx(text):
        try:
            current["vx"] = float(text)
            update()
        except ValueError:
            pass

    def on_submit_vy(text):
        try:
            current["vy"] = float(text)
            update()
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Text boxes
    # ------------------------------------------------------------------
    fig.text(
        0.5,
        0.18,
        "Enter vector components and press Enter:",
        ha="center",
        fontsize=10,
        color="#444444",
    )

    ax_vx = fig.add_axes([0.22, 0.10, 0.22, 0.05])
    ax_vy = fig.add_axes([0.58, 0.10, 0.22, 0.05])

    tb_vx = TextBox(
        ax_vx, "vₓ = ", initial="1.0", color="#f5f5f5", hovercolor="#e0e0e0"
    )
    tb_vy = TextBox(
        ax_vy, "vy = ", initial="1.0", color="#f5f5f5", hovercolor="#e0e0e0"
    )

    tb_vx.on_submit(on_submit_vx)
    tb_vy.on_submit(on_submit_vy)

    update()  # draw initial vectors

    plt.show()


# ---------------------------------------------------------------------------
# Example  (run this file directly to see a demo)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- shear ---
    M = np.array([[1, 1], [2, 2]])

    # --- 90-degree counter clockwise rotation ---
    # M = np.array([[0, 1], [-1, 0]])

    # --- 45-degree rotation ---
    # theta = np.pi / 4
    # M = np.array([
    #     [np.cos(theta), -np.sin(theta)],
    #     [np.sin(theta),  np.cos(theta)],
    # ])

    visualize_transformation(M, show_old_grid=True)
