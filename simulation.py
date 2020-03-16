from dataclasses import dataclass
import numpy as np


@dataclass
class Agent:
    """
    Data class for an agent
    contains a vector indicating their preferences
    each dimension is in [0,1] - picture it as political orientation (left/right, traditionalist/progressivist etc)
    """
    id: int
    value_preferences: list


def generate_agents(number_of_agents: int = 100, value_dimensions: int = 3) -> set:
    """
    Use this function to generate the set of voting agents
    :param number_of_agents number of agents to generate
    :param value_dimensions how many dimensions the agents' values have
    :return a set of voting agents
    """
    set_of_agents = set()
    for i in range(number_of_agents):
        # Generate a random set of values and normalize it
        value_prefs = [np.random.random() for _ in range(value_dimensions)]
        value_prefs_sum = sum(value_prefs)
        value_prefs_normalized = [float(i) / value_prefs_sum for i in value_prefs]

        set_of_agents.add(Agent(id=i, value_preferences=value_prefs_normalized))
    return set_of_agents


def generate_profile(voter_set) -> np.ndarray:
    """
    Generate a profile from a set of voters
    :param voter_set: the set of all voter agents participating in the profile
    :return:
    """
    return NotImplemented


def calculate_vote(profile) -> list:
    """
    Calculate a list of winners, given a profile
    :param profile: the ballot of voters
    :return: a list of the options, from most to least important
    """
    return NotImplemented


def generate_and_simulate():
    voter_set = generate_agents(number_of_agents=10)
    profile = generate_profile(voter_set=voter_set)
    calculate_vote(profile=profile)


if __name__ == "main":
    generate_and_simulate()
