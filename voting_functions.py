# Last update Tuesday 18-02 22:35

import numpy as np
from numpy import random
import itertools


def dictatorship(A, n, ballot):
    print('------------DICTATORSHIP------------')
    # Dictatorship possibilities are
    column = ballot[:, 0]
    res = []
    [res.append(x) for x in column if x not in res]

    print('Possible choice for Dictatorship: ', res)

    return res


def plurality(A, n, ballot):
    print('------------PLURALITY------------')

    # Plurality
    column_occurrences = np.array([[0] * len(A)])
    for i in range(0, len(A)):
        # print(str(res[i]))
        column_occurrences = np.append(column_occurrences, [np.count_nonzero(ballot == str(A[i]), axis=0)], axis=0)

    print(column_occurrences)
    first_col = column_occurrences[:, 0]
    max_plu = np.where(first_col == np.amax(first_col))

    res = []
    for i in range(0, len(max_plu)):
        index = max_plu[i]
        for x in range(0, len(index)):
            print(A[index[x] - 1])
            res.append(A[index[x] - 1])

    return set(res)


# Borda
def borda(A, n, ballot):
    print('-----------BORDA------------')
    row_occurrences = np.array([[0] * len(A)])
    for i in range(0, len(A)):
        row_occurrences = np.append(row_occurrences, [np.count_nonzero(ballot == str(A[i]), axis=0)], axis=0)

    # print(row_occurrences)
    tem = np.array(np.arange(len(A)))[::-1]
    b = tem[:(len(A))]
    borda_column = row_occurrences * b

    borda_column_sum = np.sum(borda_column, axis=1)
    # print(borda_column)
    print(borda_column_sum)

    result = np.where(borda_column_sum == np.amax(borda_column_sum))

    res = []
    for i in range(0, len(result)):
        index = result[i]
        for x in range(0, len(index)):
            print(A[index[x] - 1])
            res.append(A[index[x] - 1])

    return set(res)


def condorcet(A, n, ballot):
    print('------------CONDORCET----------')

    graph = []
    for a, b in itertools.combinations(A, 2):
        defeats = 0
        for l in ballot:
            if list(l).index(a) < list(l).index(b):
                defeats += 1
            else:
                defeats -= 1

        graph.append(a + '' + b + str(defeats))
        graph.append(b + '' + a + str(-defeats))

    print(graph)

    tie = True
    for choice in A:
        winner = True
        for edge in graph:
            if edge.startswith(choice) and ('-' in edge or '0' in edge):
                winner = False

        if winner:
            print('Condorcet winner: ' + choice)
            tie = False
            return set([choice])

    return set(A)


def stv(A, n, ballot):
    print('------------STV------------')

    # Plurality
    column_occurrences = np.array([[0] * len(A)])
    for i in range(0, len(A)):
        # print(str(res[i]))
        column_occurrences = np.append(column_occurrences, [np.count_nonzero(ballot == str(A[i]), axis=0)], axis=0)

    # print(column_occurrences)
    first_col = column_occurrences[:, 0]
    max_plu = np.where(first_col == np.amax(first_col))

    for i in range(0, len(A) - 2):
        first_col = column_occurrences[:, i]
        min_plu = np.where(first_col == np.min(first_col[np.nonzero(first_col)]))

        for i in range(0, len(max_plu)):
            index = min_plu[i]
            print(index)
            for x in range(0, len(index)):
                print(A[index[x] - 1])

                if len(index) < 2:
                    flatten_ballot = ballot.flatten()
                    index_min = [i for i, v in enumerate(flatten_ballot) if A[index[x] - 1] in v]
                    print(index_min)
                    deleted_ballot = np.delete(flatten_ballot, index_min, axis=None)
                    column_occurrences = deleted_ballot
                    print(column_occurrences)

                else:
                    print('Tie so STV rule not applicable')
                    break;


def stv2(A, n, ballot, remove_first=True):
    print("STV2 - remove first=", remove_first)
    plurality_scores = {option: 0 for option in A}

    # Calculate plurality scores (how many times each option is first in someone's ballot)
    for vote in ballot:
        plurality_scores[vote[0]] += 1

    print(plurality_scores)

    # While there are scores lower than the biggest one (losers exist in the dictionary)
    while len(set(plurality_scores.values())) != 1:
        print("stv_iter")
        # find lowest score different from 0
        min_score = min(plurality_scores.values())

        # find the option that has that lowest score
        for option, score in plurality_scores.items():
            if score == min_score:
                min_option = option
                if remove_first:
                    break

        # for each voter, transfer loser's vote to backup and remove loser from vote
        for i in range(len(ballot)):
            if ballot[i].index(min_option) == 0:
                # transfer to backup
                plurality_scores[ballot[i][ballot[i].index(min_option) + 1]] += 1
            # remove loser
            ballot[i].remove(min_option)

        print("removing ", min_option, min_score)
        # Remove loser from scores dict
        plurality_scores.pop(min_option, None)
        print(plurality_scores)
        print('\n'.join(map(str, ballot)))

    print(plurality_scores)
    return set(plurality_scores.keys())


for j in range(50):
    A = ['a', 'b', 'c', 'd']  # this can be changed to add more options in A
    ballot = np.array([np.random.permutation(A)])
    n = 4  # number of voters
    for i in range(0, n):  # change second parameter for number of voters
        v = np.random.permutation(A)
        ballot = np.append(ballot, [v], axis=0)

    print(ballot)

    # functions
    ballot_list = [n.tolist() for n in ballot]
    res = [plurality(A, n, ballot), condorcet(A, n, ballot), borda(A, n, ballot), stv2(A, n, ballot_list)]
    if len(res) == len(set(tuple(x) for x in res)):
        print('Iteration: ', j)
        print(ballot)
        print(res)
        break





