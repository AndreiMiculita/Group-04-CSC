import copy
import argparse
import random as rn
from dataclasses import dataclass
from itertools import combinations
from scipy.stats import rankdata
from voting_functions import sequential_plurality, knapsack, average_vote, dictatorship
from tabulate import tabulate

import numpy as np

# Random number generator seed, set to None for true random
seed = None


@dataclass
class Agent:
    """
    Data class for an agent
    contains a vector indicating their preferences
    each dimension is in [0,1] -picture it as political orientation (left/right, traditionalist/progressivist etc)
    """

    id: int
    value_preferences: list


def generate_agents(number_of_agents: int = 100, value_dimensions: int = 3) -> list:
    """
    Use this function to generate the set of voting agents
    :param number_of_agents number of agents to generate
    :param value_dimensions how many dimensions the agents values have
    :return a set of voting agents
    """

    # N is going to be our set of agents (voters)
    N = list()

    for i in range(number_of_agents):
        # Generate a random set of values and normalize it, this are going to be the value preferences of the agent
        value_pr = [np.random.random() for a in range(value_dimensions)]
        value_pr_sum = sum(value_pr)
        value_pr_normalized = [(float(i) / value_pr_sum) for i in value_pr]

        N.append(Agent(id=i, value_preferences=value_pr_normalized))

    return N


def generate_projects(num_projects: int = 10, value_dimensions: int = 3) -> list:
    """
    Generate a set of projects
    :param num_projects: the number of projects presented to the agents
    :param value_dimensions: how many dimensions the agents values have
    :return: A set of all possible proposals (or projects)
    """

    # A is the set of all possible proposals (or projects)
    A = list()

    for i in range(num_projects):
        pref = rn.randrange(1, value_dimensions + 1, 1)
        project = np.zeros(value_dimensions)
        project[:pref] = 1
        project_f = np.random.permutation(project)

        A.append(project_f)

    return A


def generate_profile_preference(voter_set, max_costs: [int], budget: int = 100, num_projects: int = 10) -> np.ndarray:
    """
    Generate a profile from a set of voters
    :param voter_set: the set of all voter agents participating in the profile
    :param max_costs: a list containing the maximum cost that can be allocated to each project
    :param budget: the maximum budget that must be allocated
    :param num_projects: the number of projects presented to the agents
    :return: profile cost per project of each agent
    """
    # Create resources aray with probabilities directly proportional to profile[i] (and independent from size)
    profile = np.ndarray((len(voter_set), num_projects))

    r = np.concatenate([[i] * k for i, k in enumerate(max_costs)])

    for voter_idx, voter in enumerate(voter_set):
        projects_pref = generate_projects(num_projects, len(voter_set[0].value_preferences))

        projects_cost = np.multiply(voter.value_preferences, projects_pref)

        sum_proj_pref = np.sum(projects_cost, axis=1)
        norm_proj_pref = np.divide(sum_proj_pref, sum(sum_proj_pref))

        probs = np.concatenate([[norm_proj_pref[i] / k] * k for i, k in enumerate(max_costs)])

        # Sample resources
        sampled = np.random.choice(r, p=probs, replace=False, size=budget)

        # Convert back to expenses
        profile[voter_idx, :] = np.array([(sampled == i).sum() for i in range(len(norm_proj_pref))])

    profile = profile.astype(np.int)

    print("Spending: ", profile, " for a total of ", profile.sum())

    return profile


def generate_profile(voter_set, budget: int = 100) -> np.ndarray:
    """
    Generate a profile from a set of voters
    :param voter_set: the set of all voter agents participating in the profile
    :param budget: the maximum budget that must be allocated
    :return: profile cost per project of each agent
    """

    # Profile with cost per project
    profile = np.ndarray((len(voter_set), len(voter_set[0].value_preferences)))

    for voter in voter_set:
        cost_preference = np.multiply(voter.value_preferences, budget)
        profile[voter.id] = cost_preference

    return profile


def cost_to_order_profile(profile) -> np.ndarray:
    """
    Convert a cost preference profile to a linear order ballot
    :param profile: the cost preference profile
    :return: linear order ballot
    """

    ballot = np.ndarray(profile.shape)

    if len(profile.shape) > 1:
        for i in range(0, len(profile)):
            p = profile[i]
            ballot[i] = np.argsort(-1 * p) + 1
    else:
        ballot = np.argsort(-1 * profile) + 1

    ballot = ballot.astype(int)

    return ballot


def calculate_vote(profile, function, max_cost: [int], budget: int) -> np.array:
    """
    Calculates final ballot give a profile and a function to be used
    :param profile: the cost preference profile and function to be used
    :param function: the voting function
    :param max_cost: the maximum cost of the projects
    :param budget: total budget
    :return: final linear order
    """

    A_example = list(range(1, np.size(profile, 1) + 1, 1))

    ballot_list = [n.tolist() for n in profile]
    ballot_copy = copy.deepcopy(ballot_list)  # created a copy to send to two different functions

    attributes = function(A_example, ballot_copy, max_cost, budget)
    # Maybe the dict isn't sorted by keys
    result = np.array([attributes[key] for key in sorted(attributes.keys(), reverse=False)])

    # if function == 'knapsack':
    #     result = np.array(knapsack(A_example, ballot_copy, max_cost))
    #
    # elif function == 'average_vote':
    #     result = np.array(average_vote(A_example, ballot_copy, max_cost))
    #
    # elif function == 'sequential_plurality':
    #     result = np.array(sequential_plurality(A_example, ballot_copy, max_cost, budget))
    #
    # elif function == 'dictatorship':
    #     result = np.array(dictatorship(ballot_copy))
    #
    # else:
    #     return None

    return result


def abs_cost_difference(profile, final_cost) -> float:
    """
    Calculates the absolute cost difference between each agent profile and the final profile
    and returns the sum for all agents for all projects.
    :param profile: the cost preference ballot
    :param final_cost: the final cost preference obtained by the applcation of social choice function
    :return: integer  (absolute cost difference between each agent profile and the finel profile)
    """

    diff = abs(profile - final_cost)

    # Divide by number of agents, to have a metric that doesn't depend on it
    cost_abs = float(sum(sum(diff))) / len(profile)

    return cost_abs


def sum_kendalltau_dist(profile, final_linearorder) -> int:
    """
    Calculates the absolute cost difference between each agent profile and the final profile
    and returns the sum for all agents for all projects.
    :param profile: the cost preference ballot
    :param final_linearorder: the final linear order obtained by the applcation of social choice function
    :return: integer  (absolute cost difference between each agent profile and the finel profile)
    """

    agent_dist = []

    for i in profile:
        agent_dist.append(kendalltau_dist(i, final_linearorder))

    sum_agent_dist = sum(agent_dist)

    return sum_agent_dist


def kendalltau_dist(sigma_one, sigma_two) -> int:
    """
    Calculates the Kendall tau rank distance which is a metric that counts the number of pairwise disagreements
    between two ranking lists. The larger the distance, the more dissimilar the two lists are.
    :param sigma_one: the agent rank preference ballot
    :param sigma_two: the final rank preference obtained by the applcation of social choice function
    :return: integer  (number of pairwise disagreement)
    """

    tau = 0
    n_projects = len(sigma_one)

    for i, j in combinations(range(n_projects), 2):
        tau += (np.sign(sigma_one[i] - sigma_one[j]) == -np.sign(sigma_two[i] - sigma_two[j]))

    return tau


def generate_and_simulate(number_of_agents, value_dimensions, budget, num_projects, max_cost):
    max_cost = max_cost
    A_example = np.arange(1, num_projects + 1, 1).tolist()

    voter_set = generate_agents(number_of_agents, value_dimensions)
    profile_pref = generate_profile_preference(voter_set=voter_set, max_costs=max_cost, budget=budget,
                                               num_projects=num_projects)

    print("---Profile---")
    print(profile_pref)
    ballot = cost_to_order_profile(profile_pref)
    profile = [profile_pref, ballot]
    print('---Profile Order---')
    print(ballot)

    # Using dictionary to store the cost allocation and linear order result of each function
    voting = {}
    functions = [knapsack, average_vote, dictatorship, sequential_plurality]

    for f in functions:
        if f == sequential_plurality:
            allocation = calculate_vote(ballot, f, max_cost, budget)
            allocation_list = list(allocation)
            # Pad with zeros
            allocation_list += [0] * (num_projects - len(allocation_list))
            allocation = np.array(allocation_list)
            order = cost_to_order_profile(allocation)
        else:
            allocation = calculate_vote(profile_pref, f, max_cost, budget)
            order = cost_to_order_profile(allocation)

        voting[f.__name__] = {'allocation': allocation, 'order': order}

    # Older version in comments:
    # knapsack_cost = calculate_vote(profile_pref, 'knapsack', max_cost)
    # knapsack_order = cost_to_order_profile(knapsack_cost)
    # knapsack = [knapsack_cost, knapsack_order]
    #
    # print('_______knapsack cost_________')
    # print(knapsack_cost)
    # print('_______knapsack order_________')
    # print(knapsack_order)
    #
    # # Average vote trial
    # average_cost = calculate_vote(profile_pref, 'average_vote', max_cost)
    # average_order = cost_to_order_profile(average_cost)
    # average = [average_cost, average_order]
    #
    # print('_______average cost_________')
    # print(average_cost)
    # print('_______average order_________')
    # print(average_order)
    #
    # # Sequential plurality trial
    # sequential_order = calculate_vote(ballot, 'sequential_plurality', max_cost)
    # sequential_cost = aggregate_vote_to_cost(sequential_order, max_cost, budget, A_example)
    # sequential = [sequential_cost, sequential_order]
    #
    # print('_______sequential cost_________')
    # print(sequential_cost)
    # print('_______sequential order_________')
    # print(sequential_order)
    #
    # # Dictatorship trial
    # dictatorship_cost, n_agent = calculate_vote(profile_pref, 'dictatorship', max_cost)
    # dictatorship_order = ballot[n_agent, :]
    # dictatorship = [dictatorship_cost, dictatorship_order]
    #
    # print('_______dictatorship cost_________')
    # print(dictatorship_cost)
    # print('_______dictatorship order_________')
    # print(dictatorship_order)

    for x in voting:
        print(x)
        for y in voting[x]:
            print(y, ':', voting[x][y])

    return profile, voting  # knapsack, average, sequential, dictatorship, voting


def muliple_runs_evaluation(number_of_agents, value_dimensions, budget, num_projects, number_of_runs, max_cost):
    kendall = list()
    absv = list()

    print(f"The costs of the projects are: {max_cost}.")
    print(f"The total budget is {budget}.")

    for i in range(0, number_of_runs):
        print(f"_____RUN {i}_____")

        # Running multiple simulations individually
        profile, voting = generate_and_simulate(number_of_agents, value_dimensions, budget, num_projects, max_cost)

        order = []
        cost = []
        for f in voting.keys():
            order.append(sum_kendalltau_dist(profile[1], voting[f]['order']))
            cost.append(abs_cost_difference(profile[0], voting[f]['allocation']))

        kendall.append([order])
        absv.append([cost])

        # knap_kendall = sum_kendalltau_dist(profile[1], knapsack[1])
        # knap_abs = abs_cost_difference(profile[0], knapsack[0])

        # av_kendall = sum_kendalltau_dist(profile[1], average[1])
        # av_abs = abs_cost_difference(profile[0], average[0])

        # seq_kendall = sum_kendalltau_dist(profile[1], sequential[1])
        # seq_abs = abs_cost_difference(profile[0], sequential[0])

        # dict_kendall = sum_kendalltau_dist(profile[1], dictatorship[1])
        # dict_abs = abs_cost_difference(profile[0], dictatorship[0])

        # kendall.append([knap_kendall, av_kendall, seq_kendall, dict_kendall])
        # absv.append([knap_abs, av_abs, seq_abs, dict_abs])

    rank_kendall = np.sum([rankdata(item, method='min') for item in kendall], 0)
    rank_absv = np.sum([rankdata(item, method='min') for item in absv], 0)

    print(tabulate([["Kendall score"] + list(rank_kendall), ["Abs diff"] + list(rank_absv)], headers=['Ranking function', 'Knapsack', 'Average', 'Dictatorship', 'Sequential plurality']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('n_of_agents', type=int, default=10,
                        help='number of individuals in the simulation')
    parser.add_argument('n_of_dimensions', type=int, default=3,
                        help='number of personal opinions (dimensions) of the agents')
    parser.add_argument('budget', type=int, default=100,
                        help='Total budget that needs to be distributed over projects')
    parser.add_argument('num_projects', type=int, default=10,
                        help='Total number of projects over which the budget needs to be distributed')
    parser.add_argument('num_of_runs', type=int, default=10,
                        help='Total number of simulations to perform')

    args = parser.parse_args()

    # Remove this line to get truly random numbers
    # Generate max_costs, assuming no cost is greater than half the budget, and there are more low cost projects
    # Outputs at 1 are most frequent, falling off linearly towards budget/2
    max_costs = [0] * args.num_projects
    # This loop is very unlikely to run more than once
    while sum(max_costs) <= args.budget:
        max_costs = [int(np.floor(abs(rn.random() - rn.random()) * args.budget + 1)) for _ in range(args.num_projects)]

    # Evaluation method
    muliple_runs_evaluation(args.n_of_agents, args.n_of_dimensions, args.budget,
                            args.num_projects, args.num_of_runs, max_costs)
