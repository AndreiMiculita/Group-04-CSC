import sys
from typing import Tuple, List

import networkx as nx

try:
    # The insertion index should be 1 because index 0 is this file
    # This is messy but everyone seems to be doing it this way
    sys.path.insert(1, '/home/andrei/PycharmProjects/Group-04-CSC/sdopt-tearing-master')  # the type of path is string
    from grb_lazy import *
except (ModuleNotFoundError, ImportError) as e:
    print("{} fileure".format(type(e)))
else:
    print("Import succeeded")


def knapsack_comparisons(pairs: List[Tuple[int, int]]):
    """
    Generate an ordered set of projects to be given budgets. Build a graph based on the pairs, then compute a strict
    rank ordering by solving weighted Minimum Feedback Arc Set problem.
    :param pairs: a set of value-for-money pair rankings obtained from voters
    :return: result of computation
    """
    # Make a directed graph
    g = nx.DiGraph()
    # Add each of the pairwise comparisons as edges, increase/decrease weight by 1 if already present
    for pair in pairs:
        if pair in g.edges:
            g.edges[pair[0], pair[1]]['weight'] += 1
        elif (pair[1], pair[0]) in g.edges:
            g.edges[pair[1], pair[0]]['weight'] -= 1
        else:
            g.add_edge(pair[0], pair[1], weight=1)

    # Reverse all edges with negative weights, may make problem easier to solve
    # This temp list is to avoid having problems related to changing size during iteration
    edge_data = list(g.edges.data('weight'))
    for u, v, weight in edge_data:
        if (u, v) in g.edges and weight < 0:
            g.add_edge(v, u, weight=-weight)
            g.remove_edge(u, v)

    print(g.nodes)
    print(g.edges.data('weight'))

    # TODO get linear order from graph
    # Use this library
    # https://github.com/baharev/sdopt-tearing
    # https://www.mat.univie.ac.at/~neum/ms/minimum_feedback_arc_set.pdf

    print(solve_problem(g, (0, 0, 0)))

    return NotImplemented


def comparisons_test():
    """
    Testing the knapsack comparison function
    """
    pairs = [(1, 2), (1, 2), (2, 1), (2, 1), (2, 1), (1, 3), (1, 3), (1, 3), (3, 2)]
    knapsack_comparisons(pairs)


if __name__ == "__main__":
    comparisons_test()
