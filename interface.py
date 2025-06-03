"""Simple command line interface for visualising the custom decision tree.

The script loads the data specified in :mod:`config`, trains the
``MyDecisionTreeRegressor`` model from :mod:`decision_tree` and then exports the
resulting tree structure to a PNG image using the :mod:`graphviz` library.

Usage::

    python interface.py --output my_tree

The resulting file ``my_tree.png`` will be created in the current directory.
Different tree parameters can be tweaked via command line options.
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any

import numpy as np
import pandas as pd
from graphviz import Digraph

from config import (
    DATA_PATH,
    MAX_DEPTH,
    MAX_CATEGORIES,
    MIN_INFORMATION_GAIN,
    MIN_SAMPLES_SPLIT,
)
from decision_tree import MyDecisionTreeRegressor


def _add_nodes(graph: Digraph, tree: Any, counter: itertools.count) -> str:
    """Recursively add nodes from ``tree`` to ``graph``.

    Parameters
    ----------
    graph:
        Graphviz graph instance where nodes/edges are added.
    tree:
        The tree structure produced by ``MyDecisionTreeRegressor``.
    counter:
        ``itertools.count`` object used to generate unique node identifiers.

    Returns
    -------
    str
        Identifier of the current node in the graph.
    """

    node_id = str(next(counter))

    if not isinstance(tree, dict):
        # Leaf node
        graph.node(
            node_id,
            label=f"{tree:.3f}",
            shape="ellipse",
            style="filled",
            fillcolor="#FFECB3",
            color="#FBC02D",
        )
        return node_id

    question = next(iter(tree))
    graph.node(
        node_id,
        label=question,
        shape="box",
        style="filled,rounded",
        fillcolor="#BBDEFB",
        color="#1E88E5",
    )
    yes_branch, no_branch = tree[question]

    left_id = _add_nodes(graph, yes_branch, counter)
    right_id = _add_nodes(graph, no_branch, counter)

    graph.edge(node_id, left_id, label="True", color="#2E7D32")
    graph.edge(node_id, right_id, label="False", color="#C62828")

    return node_id


def visualise_tree(tree: dict[str, Any], output_path: str) -> None:
    """Create a visual representation of ``tree`` and write it to ``output_path``.

    Parameters
    ----------
    tree:
        A nested dictionary produced by ``MyDecisionTreeRegressor``.
    output_path:
        Path without extension where the image will be written. ``graphviz``
        appends ``.png`` automatically.
    """

    g = Digraph(format="png")
    g.attr("graph", bgcolor="white", rankdir="TB")
    g.attr("node", fontname="Helvetica")
    g.attr("edge", fontname="Helvetica")
    _add_nodes(g, tree, itertools.count())
    g.render(output_path, view=False, cleanup=True)


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training data from ``path``.

    The function mirrors the data preparation steps used in ``decision_tree``'s
    ``__main__`` block.
    """

    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(" ", "_")
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    df_wo_nulls = df.dropna()

    X = df_wo_nulls.drop(["Hotel_Name", "Price", "Description"], axis=1)
    y = np.log(df_wo_nulls["Price"])
    return X, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise custom decision tree")
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to CSV file with training data (default from config)",
    )
    parser.add_argument(
        "--output",
        default="tree",
        help="Output file path without extension (PNG will be produced)",
    )
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--min-samples-split", type=int, default=MIN_SAMPLES_SPLIT)
    parser.add_argument("--min-information-gain", type=float, default=MIN_INFORMATION_GAIN)
    parser.add_argument("--max-categories", type=int, default=MAX_CATEGORIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, y = load_data(args.data)

    tree = MyDecisionTreeRegressor(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_information_gain=args.min_information_gain,
        max_categories=args.max_categories,
    )
    tree.fit(X, y)

    if tree.tree is None:
        raise RuntimeError("The tree has not been fitted correctly")

    visualise_tree(tree.tree, args.output)
    print(f"Tree visualisation saved to {args.output}.png")


if __name__ == "__main__":
    main()

