# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""An application class for the Max-k-cut."""

from typing import List, Dict, Optional, Union
import networkx as nx
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgba
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .graph_optimization_application import GraphOptimizationApplication


class Maxkcut(GraphOptimizationApplication):
    """Optimization application for the "max-k-cut" [1] problem based on a NetworkX graph.

    References:
        [1]: Z. Tabi et al.,
             "Quantum Optimization for the Graph Coloring Problem with Space-Efficient Embedding"
             2020 IEEE International Conference on Quantum Computing and Engineering (QCE),
             2020, pp. 56-62, doi: 10.1109/QCE49297.2020.00018.,
             https://ieeexplore.ieee.org/document/9259934
    """

    def __init__(
        self,
        graph: Union[nx.Graph, np.ndarray, List],
        k: int,
        colors: Optional[Union[List[str], List[List[int]]]] = None
    ) -> None:
        """
        Args:
            graph: A graph representing a problem. It can be specified directly as a
                `NetworkX <https://networkx.org/>`_ graph,
                or as an array or list format suitable to build out a NetworkX graph.
            k: The number of colors
            colors: List of strings or list of colors in rgba lists to be assigned to each
                resulting subset, there must be as many colors as the number k
        """
        super().__init__(graph=graph)
        self._k_num = k
        self._colors = colors if colors and len(colors) >= k else None

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a Max-k-cut problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the Max-k-cut problem instance.
        """
        for w, v in self._graph.edges:
            self._graph.edges[w, v].setdefault("weight", 1)

        mdl = Model(name="Max-k-cut")
        n = self._graph.number_of_nodes()
        k = self._k_num
        x = {
            (v, i): mdl.binary_var(name=f"x_{0}_{1}".format(v, i))
            for v in range(n)
            for i in range(k)
        }
        first_penalty = mdl.sum_squares((1 - mdl.sum(x[v, i] for i in range(k)) for v in range(n)))
        second_penalty = mdl.sum(
            mdl.sum(self._graph.edges[v, w]["weight"] * x[v, i] * x[w, i] for i in range(k))
            for v, w in self._graph.edges
        )
        objective = first_penalty + second_penalty
        mdl.minimize(objective)

        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[List[int]]:
        """Interpret a result as k lists of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            k lists of node indices correspond to k node sets for the Max-k-cut
        """
        x = self._result_to_x(result)
        n = self._graph.number_of_nodes()
        cut = [[] for i in range(self._k_num)]  # type: List[List[int]]

        n_selected = x.reshape((n, self._k_num))
        for i in range(n):
            node_in_subset = np.where(n_selected[i] == 1)[0]  # one-hot encoding
            if len(node_in_subset) != 0:
                cut[node_in_subset[0]].append(i)

        return cut

    def _draw_result(
        self,
        result: Union[OptimizationResult, np.ndarray],
        pos: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Draw the result with colors

        Args:
            result : The calculated result for the problem
            pos: The positions of nodes
        """
        x = self._result_to_x(result)
        nx.draw(self._graph, node_color=self._node_color(x), pos=pos, with_labels=True)

    def _node_color(self, x: np.ndarray) -> List[List[int]]:
        # Return a list of colors for draw.

        n = self._graph.number_of_nodes()

        # k colors chosen from cm.rainbow, or from given color list
        colors = (
            cm.rainbow(np.linspace(0, 1, self._k_num)) if self._colors is None else self._colors
        )
        gray = to_rgba('lightgray')
        node_colors = np.full((n, len(gray)), gray)

        n_selected = x.reshape((n, self._k_num))
        for i in range(n):
            node_in_subset = np.where(n_selected[i] == 1)  # one-hot encoding
            if len(node_in_subset[0]) != 0:
                node_colors[i] = to_rgba(colors[node_in_subset[0][0]])

        return node_colors

    @property
    def k(self) -> int:
        """Getter of k

        Returns:
            The number of colors
        """
        return self._k_num

    @k.setter
    def k(self, k: int) -> None:
        """Setter of k

        Args:
            k: The number of colors
        """
        self._k_num = k
        self._colors = self._colors if self._colors and len(self._colors) >= k else None

    @property
    def colors(self) -> Union[List[str], List[List[int]]]:
        """Getter of colors list

        Returns:
            The k size color list
        """
        return self._colors

    @colors.setter
    def colors(self, colors: Union[List[str], List[List[int]]]) -> None:
        """Setter of colors list

        Args:
            colors: The k size color list
        """
        self._colors = colors if colors and len(colors) >= self._k_num else None
