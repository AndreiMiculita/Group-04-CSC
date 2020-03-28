from dataclasses import dataclass
import numpy as np
import sys
import argparse


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


def generate_profile(voter_set,budget: int=100) -> np.ndarray:
    #TODO
    """
    Generate a profile from a set of voters
    :param voter_set: the set of all voter agents participating in the profile
    :return: profile: cost per project of each agent
    """
    # Profile with cost per project
    profile = np.ndarray((len(voter_set),len(voter_set[1].value_preferences)))
    for voter in voter_set:
        cost_preference = [i*budget for i in voter.value_preferences]
        profile[voter.id] = cost_preference

    return profile


def calculate_vote(profile) -> list:
    # TODO
    """
    Calculate a list of winners, given a profile
    :param profile: the ballot of voters
    :return: a list of the options, from most to least important
    """
    return NotImplemented


def generate_and_simulate(number_of_agents, value_dimensions, budget):
    voter_set = generate_agents(number_of_agents, value_dimensions)
    profile = generate_profile(voter_set=voter_set,budget=budget)
    print(profile)
    calculate_vote(profile=profile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('n_of_agents',type=int, default=10, 
                    help='number of individuals in the simulation')
    parser.add_argument('n_of_dimensions',type=int, default=3, 
                    help='number of personal opinions (dimensions) of the agents')
    parser.add_argument('budget',type=int,default=100,
                             help='Total budget that needs to be distributed over projects')

    args = parser.parse_args()

    generate_and_simulate(args.n_of_agents, args.n_of_dimensions,args.budget)

