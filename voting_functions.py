# Last update Tuesday 18-02 22:35

import copy
import itertools

import networkx as nx
import numpy as np
from numpy import random
from typing import Tuple, List


def knapsack(A, ballot, max_cost, budget) -> dict:
    """
    Knapsack voting for participatory budgeting
    Needs a profile where each voter defines how much money they want to allocate to each of the choices
    """

    # Transpose the ballot: to see the votes per choice instead of per voter (alternatively, provide the ballot in
    # that form)
    ballot_t = np.transpose(ballot)

    # This will hold the final allocation per choice
    allocation = []

    # Go over each choice
    for i in range(0, len(ballot_t)):
        # Values per dollar: See paper "Knapsack Voting for Participatory Budgeting" for a description of this method
        # sack = np.zeros(max(ballot_t[i])).tolist()

        sack = np.zeros(max_cost[i]).tolist()
        for j in range(0, len(ballot_t[i])):
            vote = ballot_t[i][j]
            for k in range(0, vote):
                if k < len(sack):
                    sack[k] += 1

        # The final allocation is the maximum amount of values per dollar for each choice
        sack = [v for v in sack if v != 0]
        if len(set(sack)) != 1:
            allocation.append(len([v for v in sack if v != min(sack)]))
        else:
            allocation.append(len(sack))

    allocation_dict = {i: cost for i, cost in enumerate(allocation)}

    return allocation_dict


def average_vote(A, ballot, max_cost, budget) -> dict:
    """
    Average function: allocates the average of the allocated cost for each project
    """
    ballot_t = np.transpose(ballot)

    allocation = dict()

    for i in range(0, len(ballot_t)):
        allocation[i] = sum(ballot_t[i]) / np.size(ballot, 0)

    return allocation


def sequential_plurality(A, ballot, max_cost, budget) -> dict:
    # Specify how many elements you want in the social choice set
    # Sometimes the set will have more than k elements , when there are ties
    res = []

    old_A = A

    while len(res) < len(old_A):
        plurality_scores = {option: 0 for option in A}

        # Calculate plurality scores (how many times each option is first in someone's ballot)
        for vote in ballot:
            plurality_scores[vote[0]] += 1

        # Highest score
        max_score = max(plurality_scores.values())

        # Find the option(s) that has that highest score
        for option, score in plurality_scores.items():
            if score == max_score:
                res.append(option)
                break

        # Remove the maximum option from ballot and from list of options
        for i in range(len(ballot)):
            ballot[i] = [e for e in ballot[i] if e not in res]

        A = [option for option in A if option not in res]

    # Final allocation
    allocation = dict()

    for i, r in enumerate(res):
        cost = max_cost[old_A.index(r)]

        # Budget is exhausted
        if budget <= 0:
            break

        # Assign maximum cost of option OR the leftover budget
        allocation[i] = min(budget, cost)
        budget -= min(budget, cost)

    # Median kept as comments:
    # Calculate median per project
    # median = np.median(profile, axis=0)
    #
    # # Arrange it with respect to the social choice
    # for i in range(0, len(res)):
    #     allocation[i] = median[res[i] - 1]
    # '''

    return allocation


def dictatorship(A, ballot, max_cost, budget) -> dict:
    # Dictatorship possibilities are
    agent = random.randint(1, len(ballot))
    res = np.transpose(ballot)[:, agent]

    allocation = {i: alloc for i, alloc in enumerate(res)}

    return allocation



# Main
def main():
    for iteration in range(50):
        A_example = [1, 2, 3, 4]  # this can be changed to add more options in A
        ballot_example = np.array([np.random.permutation(A_example)])
        n = 4  # number of voters
        for idx in range(0, n):  # change second parameter for number of voters
            vote_example = np.random.permutation(A_example)
            ballot_example = np.append(ballot_example, [vote_example], axis=0)

        print(ballot_example)

        # functions
        ballot_list = [n.tolist() for n in ballot_example]
        ballot_copy = copy.deepcopy(ballot_list)  # created a copy to send to two different functions
        b_copy = copy.deepcopy(ballot_list)
        result = [plurality(A_example, ballot_example), condorcet(A_example, ballot_example),
                  borda(A_example, ballot_example), stv2(A_example, ballot_list),
                  sequential_plurality(A_example, ballot_copy, 2)]

        if len(result) == len(set(tuple(x) for x in result)):
            print('Iteration: ', iteration)
            print(ballot_example)
            print(result)
            break

    # Knapsack trial
    print("KNAPSACK")
    ballot_example = [[4, 5, 1], [3, 5, 2], [0, 0, 10]]
    print(knapsack(['a', 'b', 'c'], ballot_example, [5, 5, 10]))

    # Average vote trial
    print("Average function:", average_vote(['a', 'b', 'c'], ballot_example))

    # Allocation by cost trial
    A_example = [1, 2, 3, 4]
    max_cost_example = [4, 6, 10, 8]
    b_copy = [[4, 3, 1, 2], [3, 1, 4, 2], [2, 4, 3, 1], [4, 3, 2, 1]]
    result_example = sequential_plurality(A_example, b_copy, 4)
    print(result_example)
    budget_example = 10
    print("--------Allocation------")
    # print(aggregate_vote_to_cost(result_example, max_cost_example, budget_example, A_example))


if __name__ == "__main__":
    main()
