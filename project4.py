import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import tqdm as tm
import json
import pickle
import matplotlib.pyplot as plt
import copy


def get_name():  # name function
    name = "Itay lorberboym"
    print(name)
    return name


def get_id():  # id function
    id = "314977596"
    print(id)
    return id


# ------------------------ 1 ---------------------- Epidemic Spread

# -- a --

def get_infected_nodes(network):
    infectious = 0
    for n, data in network.nodes(data=True):
        if data['status'] == 'i':
            infectious += 1

    return infectious

def update_initial_infection_time(network, infection_time):
    for n in network:
        if network.nodes[n]['status'] == 'i':
            network.nodes[n]['infection_time'] = infection_time

def check_if_infection_occurred(network, n1, n2, edge_attribute, p, infections):
    if network.nodes[n1]['status'] == 'i' and network.nodes[n2][
        'status'] == 's':  # if n1 is sick and n2 is not sick
        for _ in range(edge_attribute['contacts']):
            if (np.random.rand()) < p: # if probability is in accept range
                infections['infected_in_epoche'].add(n2)
                break
        return infections
    if network.nodes[n1]['status'] == 's' and network.nodes[n2]['status'] == 'i':  # if n2 is sick and n1 is not sick
        for _ in range(edge_attribute['contacts']):
            if np.random.rand() < p * edge_attribute['contacts']:
                infections['infected_in_epoche'].add(n1)
                break
        return infections

    return infections

def update_mortality_total(network):
    nodes_to_remove = []
    for n in network:
        if (network.nodes[n]['status'] == 'i') and (np.random.rand()) < (network.nodes[n][
            'mortalitylikelihood']):  # if sick and the probability to die in current epoche is in the accepptance area
            nodes_to_remove.append(n)  # add to list of nodes that have died
    network.remove_nodes_from(nodes_to_remove)  # remove nodes that have died from the network

    return len(nodes_to_remove)

def reduce_infection_time(network):  # reduce infection time for every sick node in every epoche
    for n in network:
        if network.nodes[n]['status'] == 'i':
            network.nodes[n]['infection_time'] -= 1

def change_node_state(network, state, infectious_current):
    for n in network:
        if network.nodes[n]['status'] == 'i' and network.nodes[n][
            'infection_time'] == 0:  # if sick and infection is over
            network.nodes[n]['status'] = state  # change state
            infectious_current -= 1  # reduce the current infected nodes amount

    return infectious_current

def update_infected_nodes(network, infection_time, infections):
    for n in infections['infected_in_epoche']:
        network.nodes[n]['status'] = 'i'  # change node status to infected
        network.nodes[n]['infection_time'] = infection_time  # update node infection time
        infections['infections_total'] += 1
        infections['infectious_current'] += 1
    infections['infected_in_epoche'].clear()

    return infections

def is_sick_nodes_in_network(network):
    for n in network:
        if network.nodes[n]['status'] == 'i':
            return True
    return False

def calculate_R0(network, p):
    # R0 is calculated by all nodes that can get infected (in "s" state) and connected to sick nodes divided by the sick nodes (in "i" state) multiply by p
    if is_sick_nodes_in_network(network):
        total_healthy_nodes = 0
        num_of_sick_nodes = 0
        for n in network:
            if network.nodes[n]['status'] == 'i':  # get all nodes in status "i"
                num_of_sick_nodes += 1
                for n2 in network.neighbors(n):
                    if network.nodes[n2][
                        'status'] == 's':  # get all nodes in status "s" that connected to nodes in status "i"
                        total_healthy_nodes += 1
        k = total_healthy_nodes / num_of_sick_nodes
        return p * k
    else:  # if there are not sick nodes in network R0 will be 0
        return 0

def epidmeic_analysis(network, model_type='SIS', infection_time=2, p=0.05, epochs=20, seed=314977596):
    ans = {}
    G = copy.deepcopy(network)
    np.random.seed(seed=seed)
    mortality_total = 0
    R0 = 0
    initial_infected_nodes = get_infected_nodes(G)  # check for infected nodes at the beginning
    infections = {'infections_total': initial_infected_nodes,
                  'infectious_current': initial_infected_nodes,
                  'infected_in_epoche':set()}

    if (initial_infected_nodes != 0):
        update_initial_infection_time(G, infection_time)  # update the infection time of the initial infected nodes
        for t in range(epochs):  # iterating over epoches
            for n1, n2, edge_attribute in G.edges(data=True):  # iterating over edegs to check if nodes are infected
                infections = check_if_infection_occurred(G, n1, n2, edge_attribute, p, infections)
            died_during_epoche = update_mortality_total(G)
            mortality_total += died_during_epoche  # update network if nodes have died.
            infections['infectious_current'] -= died_during_epoche
            reduce_infection_time(G)  # reduce infection time after each epoche
            infections = update_infected_nodes(G, infection_time,
                                               infections)  # adding to infectious current the nodes that got infected during current epoche
            if model_type == 'SIS':  # if we use the SIS model
                infections['infectious_current'] = change_node_state(G, 's', infections[
                    'infectious_current'])  # we change the node state back to susceptible
            else:  # if we use SIR model
                infections['infectious_current'] = change_node_state(G, 'r', infections[
                    'infectious_current'])  # we change the node state to recovered
        R0 = calculate_R0(G, p)  # calculate R0

    ans['infections_total'] = infections['infections_total']
    ans['infectious_current'] = infections['infectious_current']
    ans['mortality_total'] = mortality_total
    ans['r_0'] = R0

    return ans


# ------------------------ 2 ---------------------- Vaccination Policy

# -- a --

def get_random_nodes_lst(network, vaccines):
    rand_set = set()
    while vaccines > 0: #while we still have vaccines left
        for n in nx.nodes(network):
            if np.random.rand() < 0.5:
                rand_set.add(n)
                vaccines -= 1
            if vaccines <= 0: # if there are not vaccines left
                return rand_set

def get_vaccined_nodes(lst, vaccines):
    vaccined_nodes = []
    for i in range(vaccines):
        vaccined_nodes.append(lst[i][0])

    return vaccined_nodes

def get_ordered_mortality_lst(network):
    mortality_lst = []

    for n in network:
        mortality_lst.append((n, network.nodes[n]['mortalitylikelihood']))
    ordered_mortality_lst = sorted(mortality_lst, key=lambda tup: tup[1], reverse=True)

    return ordered_mortality_lst

def vaccination_analysis(network, model_type='SIR', infection_time=2, p=0.05, epochs=10, seed=314977596, vaccines=50,
                         policy='rand'):
    np.random.seed(seed)
    G = copy.deepcopy(network)
    vaccined_nodes = []

    if policy == 'rand':  # random pick policty
        vaccined_nodes = get_random_nodes_lst(G, vaccines)

    elif policy == 'betweenness':  # highest betweenness policty
        betweenness_dict = nx.betweenness_centrality(G)
        betweenness_centrality_lst = [v for v in
                                      sorted(betweenness_dict.items(), key=lambda item: item[1], reverse=True)]
        vaccined_nodes = get_vaccined_nodes(betweenness_centrality_lst, vaccines)

    elif policy == 'degree':  # highest degree policty
        degree_centrality_dict = nx.degree_centrality(G)
        degree_centrality_lst = [v for v in
                                 sorted(degree_centrality_dict.items(), key=lambda item: item[1], reverse=True)]
        vaccined_nodes = get_vaccined_nodes(degree_centrality_lst, vaccines)

    elif policy == 'mortality':  # highest mortality rate policty
        ordered_mortality_lst = get_ordered_mortality_lst(G)
        vaccined_nodes = get_vaccined_nodes(ordered_mortality_lst, vaccines)

    G.remove_nodes_from(vaccined_nodes)  # removing the nodes that received vaccines

    return epidmeic_analysis(network=G, model_type=model_type, infection_time=infection_time, p=p, epochs=epochs,
                             seed=seed)  # run epidemic simulation without vaccined nodes.







