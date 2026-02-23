import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

plt.style.use("./deeplearning.mplstyle")


def format_plot_interface(canvas):
    """
    Standardizes the UI for ipympl/widget plots by hiding
    web-based headers, footers, and toolbars.
    """
    canvas.toolbar_visible = False
    canvas.header_visible = False
    canvas.footer_visible = False


def plot_entropy():
    def entropy(p):
        if p == 0 or p == 1:
            return 0
        else:
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    p_array = np.linspace(0, 1, 201)
    h_array = [entropy(p) for p in p_array]

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    format_plot_interface(fig.canvas)

    ax.set_title("p vs H(p)")
    ax.set_xlabel("p")
    ax.set_ylabel("H(p)")

    h_plot = ax.plot(p_array, h_array)
    scatter = ax.scatter(0, 0, c="r", zorder=100, s=70)

    # 1. Create an empty text object at (0,0)
    # ha='center' keeps the text balanced over the dot
    text_label = ax.text(
        0.5, 0.5, "", fontsize=10, ha="center", color="red", zorder=100
    )

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])  # left, bottom, width, height
    slider = Slider(axfreq, "p", 0, 1, valinit=0, valstep=0.05)

    def update(val):
        x = val
        y = entropy(x)
        scatter.set_offsets((x, y))

        # 2. Update the text string with formatted values
        text_label.set_text(f"(p={x:.2f}, H(p)={y:.2f})")

        # 3. Update the text position to stay slightly above the red dot
        # text_label.set_position((x, y + 0.05))

        # CRITICAL: This ensures the screen actually refreshes
        fig.canvas.draw_idle()

    slider.on_changed(update)
    return slider


def compute_entropy(y):
    if len(y) == 0:
        return 0
    p = sum(y[y == 1]) / len(y)
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def split_dataset(X, node_indices, feature):
    mask = X[node_indices, feature] == 1
    left_indices = node_indices[np.flatnonzero(mask)]
    right_indices = node_indices[np.flatnonzero(~mask)]
    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):

    left_indices, right_indices = split_dataset(X, node_indices, feature)

    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    information_gain = 0

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    information_gain = node_entropy - weighted_entropy

    return information_gain


def get_best_split(X, y, node_indices):
    num_features = X.shape[1]

    best_feature = -1

    max_info_gain = 0
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature

    return best_feature


def build_tree_recursive(
    X, y, node_indices, branch_name, max_depth, current_depth, tree
):
    # --- Input Validation / Invariants ---
    # Ensure indices are a numpy array for fancy indexing
    node_indices = np.asarray(node_indices)

    # Check if we are at the very start of the recursion
    if current_depth == 0:
        X = np.asarray(X)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(
                f"X (size {len(X)}) and y (size {len(y)}) must have the same length."
            )

        if node_indices.max() >= len(X):
            raise IndexError("node_indices contains an index out of bounds for X.")

    # --- Base Case: Stopping Condition ---
    if current_depth == max_depth:
        fmt_str = " " * current_depth + "-" * current_depth
        print(f"{fmt_str} {branch_name} leaf node with indices {node_indices}")
        return

    best_feature = get_best_split(X, y, node_indices)
    fmt_str = "-" * current_depth
    print(
        f"{fmt_str} Depth {current_depth}, {branch_name}: Split on feature: {best_feature}"
    )

    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1, tree)
    build_tree_recursive(
        X, y, right_indices, "Right", max_depth, current_depth + 1, tree
    )
    return tree


def generate_node_image(node_indices):
    image_paths = ["images/%d.png" % idx for idx in node_indices]
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im = new_im.resize(
        (
            int(total_width * len(node_indices) / 10),
            int(max_height * len(node_indices) / 10),
        )
    )

    return new_im


def generate_tree_viz(root_indices, y, tree):

    G = nx.DiGraph()

    G.add_node(0, image=generate_node_image(root_indices))
    idx = 1
    root = 0

    num_images = [len(root_indices)]

    feature_name = ["Ear Shape", "Face Shape", "Whiskers"]
    y_name = ["Non Cat", "Cat"]

    decision_names = []
    leaf_names = []

    for i, level in enumerate(tree):
        indices_list = level[:2]
        for indices in indices_list:
            G.add_node(idx, image=generate_node_image(indices))
            G.add_edge(root, idx)

            # For visualization
            num_images.append(len(indices))
            idx += 1
            if i > 0:
                leaf_names.append("Leaf node: %s" % y_name[max(y[indices])])

        decision_names.append("Split on: %s" % feature_name[level[2]])
        root += 1

    node_names = decision_names + leaf_names
    pos = graphviz_layout(G, prog="dot")

    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot(111)
    ax.set_aspect("equal")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=40)

    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    for idx, n in enumerate(G):
        xx, yy = trans(pos[n])  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        piesize = num_images[idx] / 25
        p2 = piesize / 2.0
        a = plt.axes([xa - p2, ya - p2, piesize, piesize])
        a.set_aspect("equal")
        a.imshow(G.nodes[n]["image"])
        a.axis("off")
        try:
            a.set_title(node_names[idx], y=-0.8, fontsize=13, loc="left")
        except:
            pass
    ax.axis("off")
    plt.show()
