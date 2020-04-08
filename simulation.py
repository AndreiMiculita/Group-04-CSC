import copy
import argparse
import random as rn
from dataclasses import dataclass
from itertools import combinations
from voting_functions import sequential_plurality, knapsack, average_vote

import numpy as np

# Random number generator seed, set to None for true random
seed = 51


@dataclass
class Agent:
    """
    Data class for an agent
    contains a vector indicating their preferences
    each dimension is in [0,1] - picture it as political orientation (left/right, traditionalist/progressivist etc)
    """
    id: int
    value_preferences: list


def generate_agents(number_of_agents: int = 100, value_dimensions: int = 3) -> list:
    """
    Use this function to generate the set of voting agents
    :param number_of_agents number of agents to generate
    :param value_dimensions how many dimensions the agents' values have
    :return a set of voting agents
    """
    list_of_agents = list()
    for i in range(number_of_agents):
        # Generate a random set of values and normalize it
        value_prefs = [np.random.random() for _ in range(value_dimensions)]
        value_prefs_sum = sum(value_prefs)
        value_prefs_normalized = [float(i) / value_prefs_sum for i in value_prefs]

        list_of_agents.append(Agent(id=i, value_preferences=value_prefs_normalized))
    return list_of_agents


def generate_projects(num_projects: int = 10, value_dimensions: int = 3) -> list:
    projects_pref = list()

    rn.seed(a=seed)

    for i in range(num_projects):
        pref = rn.randrange(0, value_dimensions + 1, 1)
        project = np.zeros(value_dimensions)
        project[:pref] = 1
        project_f = np.random.permutation(project)

        projects_pref.append(project_f)

    return projects_pref


def generate_profile_preference(voter_set, budget: int = 100, num_projects: int = 10) -> np.ndarray:
    profile = np.ndarray((len(voter_set), num_projects))
    for voter in voter_set:
        projects_pref = generate_projects(num_projects, len(voter_set[0].value_preferences))
        # projects_cost = [voter.value_preferences * p for p in projects_pref]
        projects_cost = np.multiply(voter.value_preferences, projects_pref)

        # print("------PROJECT COST----------")
        # print (projects_cost)

        # sum_proj_pref=[sum(p) for p in projects_cost]
        # norm_proj_pref= sum_proj_pref/sum(sum_proj_pref)
        sum_proj_pref = np.sum(projects_cost, axis=1)
        norm_proj_pref = np.divide(sum_proj_pref, sum(sum_proj_pref))

        # print(sum_proj_pref)
        # print("Then we normalize:")
        # print(norm_proj_pref)

        profile[voter.id] = norm_proj_pref * budget

    return profile


def generate_profile(voter_set, budget: int = 100) -> np.ndarray:
    # TODO
    """
    Generate a profile from a set of voters
    :param voter_set: the set of all voter agents participating in the profile
    :param budget: the maximum budget that must be allocated
    :return: profile: cost per project of each agent
    """
    # Profile with cost per project
    profile = np.ndarray((len(voter_set), len(voter_set[0].value_preferences)))
    for voter in voter_set:
        # cost_preference = [i*budget for i in voter.value_preferences]
        cost_preference = np.multiply(voter.value_preferences, budget)
        profile[voter.id] = cost_preference

    return profile


def cost_to_order_profile(profile) -> np.ndarray:
    """
    Convert a cost preference profile to a linear order ballot
    :param profile: the cost preference profile
    :return: linear order ballot
    """
    print(len(profile.shape))
    ballot = np.ndarray(profile.shape)

    if len(profile.shape)>1:
        for i in range(0, len(profile)):
            p = profile[i]
            print(p)
            ballot[i] = np.argsort(-1 * p) + 1
            print(ballot[i])
    else: 
        ballot = np.argsort(-1 * profile) + 1

    return ballot


def calculate_vote(profile) -> list:

    A_example = [1, 2, 3, 4,5]

    ballot_list = [n.tolist() for n in profile]
    ballot_copy = copy.deepcopy(ballot_list)  # created a copy to send to two different functions
    b_copy = copy.deepcopy(ballot_list)
    result = np.array(average_vote(A_example, ballot_copy))

    #print(result)

    return result

def abs_cost_difference(profile, final_cost)-> int:

    diff= abs(profile-final_cost)

    cost_abs=sum(sum(diff))

    return cost_abs

def sum_kemeny_distance(profile, final_linearorder) -> int:

    agent_dist=[]

    for i in profile:

        agent_dist.append(kendalltau_dist(profile, final_linearorder))

    sum_agent_dist=sum(agent_dist)

    print(agent_dist)

    print(sum_agent_dist)

    return sum_agent_dist


def kendalltau_dist(rank_a, rank_b):
    tau = 0
    n_candidates = len(rank_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(rank_a[i] - rank_a[j]) ==
                -np.sign(rank_b[i] - rank_b[j]))
    return tau


def generate_and_simulate(number_of_agents, value_dimensions, budget, num_projects):
    voter_set = generate_agents(number_of_agents, value_dimensions)
    profile = generate_profile(voter_set=voter_set, budget=budget)
    profile_pref = generate_profile_preference(voter_set=voter_set, budget=budget, num_projects=num_projects)
    #print("---Prajakta's profile----")
    #print(profile)
    print("---Francesca's profile----")
    print(profile_pref)
    ballot = cost_to_order_profile(profile_pref)
    print(ballot)
    result= calculate_vote(profile_pref)
    print(result)
    result_order=cost_to_order_profile(result)
    print(result_order)
    cost_abs=abs_cost_difference(profile_pref, result)

    kemeny_dis=sum_kemeny_distance(ballot, result_order)
    #print(cost_abs)
    print(kemeny_dis)



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

    args = parser.parse_args()

    generate_and_simulate(args.n_of_agents, args.n_of_dimensions, args.budget, args.num_projects)
