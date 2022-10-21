#Spyros Tzallas 4502
#Georgios Lypopoylos 4411

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
import networkx.algorithms.community as nx_comm
from networkx import edge_betweenness_centrality as betweenness, edge_betweenness_centrality, \
    edge_betweenness_centrality_subset
import random


def my_graph_plot_routine(G, fb_nodes_colors, fb_links_colors, fb_links_styles, graph_layout, node_positions):
    plt.figure(figsize=(10, 10))

    if len(node_positions) == 0:
        if graph_layout == 'circular':
            node_positions = nx.circular_layout(G)
        elif graph_layout == 'random':
            node_positions = nx.random_layout(G, seed=50)
        elif graph_layout == 'planar':
            node_positions = nx.planar_layout(G)
        elif graph_layout == 'shell':
            node_positions = nx.shell_layout(G)
        else:  # DEFAULT VALUE == spring
            node_positions = nx.spring_layout(G)

    nx.draw(G,
            with_labels=True,  # indicator variable for showing the nodes' ID-labels
            style=fb_links_styles,  # edge-list of link styles, or a single default style for all edges
            edge_color=fb_links_colors,  # edge-list of link colors, or a single default color for all edges
            pos=node_positions,  # node-indexed dictionary, with position-values of the nodes in the plane
            node_color=fb_nodes_colors,  # either a node-list of colors, or a single default color for all nodes
            node_size=100,  # node-circle radius
            alpha=0.9,  # fill-transparency
            width=0.5  # edge-width
            )
    plt.show()

    return (node_positions)


############################## READ FROM CSV ######################################################################

def STUDENT_AM_read_graph_from_csv():
    fb_links = pd.read_csv("fb-pages-food.edges")
    fb_links_df = fb_links.head(MAX_NUM_LINKS)
    fb_links_loopless_df = fb_links_df[fb_links_df["node_1"] != fb_links_df["node_2"]]

    for i in range(0, MAX_NUM_LINKS):
        if (fb_links_df["node_1"][i] not in node_name_list):
            node_name_list.append(fb_links_df["node_1"][i])

        if (fb_links_df["node_2"][i] not in node_name_list):
            node_name_list.append(fb_links_df["node_2"][i])

    G = nx.from_pandas_edgelist(fb_links_loopless_df, "node_1", "node_2", create_using=nx.Graph())
    nodesPositions = my_graph_plot_routine(G, 'red', 'blue', 'solid', graph_layout, node_positions)
    return G, nodesPositions, node_name_list


###################################### RANDOM EDGE ##################################################

def STUDENT_AM_add_random_edges_to_graph(G, node_name_list, NUM_RANDOM_EDGES, EDGE_ADDITION_PROBABILITY):
    for i in node_name_list:
        neighbors = ([n for n in G.neighbors(i)])
        for j in node_name_list:
            if (j not in neighbors and j != i):
                probability = random.uniform(0, 1)
                if (probability >= EDGE_ADDITION_PROBABILITY and NUM_RANDOM_EDGES != 0):
                    G.add_edge(i, j)
                    NUM_RANDOM_EDGES = NUM_RANDOM_EDGES - 1

    my_graph_plot_routine(G, 'red', 'blue', 'solid', graph_layout, node_positions)


################################### HAMILTON #####################################################

def STUDENT_AM_add_hamilton_cycle_to_graph(G, node_name_list):
    for i in range(0, len(node_name_list) - 1):
        G.add_edge(node_name_list[i], node_name_list[i + 1])

    my_graph_plot_routine(G, 'red', 'blue', 'solid', graph_layout, node_positions)


################################### Girvan_Newman #####################################################

def STUDENT_AM_use_nx_girvan_newman_for_communities(G, graph_layout, node_positions):
    communities = girvan_newman(G, most_valuable_edge=None)
    community_tuples = tuple(c for c in next(communities))

    return community_tuples


################################### My Girvan_Newman #####################################################
def most_central_edge_1(G):
    centrality = edge_betweenness_centrality(G, weight="weight")
    return max(centrality, key=centrality.get)


def most_central_edge_2(G, sources, targets):
    centrality = edge_betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None)
    return max(centrality, key=centrality.get)


def STUDENT_AM_one_shot_girvan_newman_for_communities(G, graph_layout, node_positions, Edge_Betweenness_Choice):
    targets = []
    community_tuples = list(nx.connected_components(G))
    LC = max(nx.connected_components(G), key=len)

    S = [G.subgraph(LC).copy()]
    source = list(S[0].nodes)
    stat = int(len(source) / 10)
    for i in range(0, stat):
        targets.append(source[i])

    while (nx.number_connected_components(S[0]) == 1):
        if (Edge_Betweenness_Choice == 1):
            most_valuable_edge = most_central_edge_1(S[0])
            S[0].remove_edge(int(most_valuable_edge[0]), int(most_valuable_edge[1]))

        else:
            most_valuable_edge = most_central_edge_2(S[0], source, targets)
            S[0].remove_edge(int(most_valuable_edge[0]), int(most_valuable_edge[1]))

    G.remove_edge(most_valuable_edge[0], most_valuable_edge[1])

    split_community_tuples = list(nx.connected_components(S[0]))
    community_tuples.remove(LC)
    community_tuples.append(split_community_tuples[0])
    community_tuples.append(split_community_tuples[1])

    my_graph_plot_routine(G, 'red', 'blue', 'solid', graph_layout, node_positions)

    return community_tuples, split_community_tuples[0], split_community_tuples[1]


################################### Divisive Community #####################################################

def STUDENT_AM_divisive_community_detection(G, number_of_divisions, graph_layout, node_positions):
    CUR_PARTITION = []
    while True:
        try:
            Edge_Betweenness_Choice = int(input(
                "Press <<1>> for choosing the exact edge betweenness number or <<2>> for approximately choosing it using a small portion of the edges in the graph: "))
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")

    HIERARCHY = STUDENT_AM_one_shot_girvan_newman_for_communities(G, graph_layout, node_positions,
                                                                  Edge_Betweenness_Choice)

    if (Edge_Betweenness_Choice == 1):
        while (nx.number_connected_components(G) < number_of_divisions):
            LC = max(nx.connected_components(G), key=len)

            HIERARCHY, LC1, LC2 = STUDENT_AM_one_shot_girvan_newman_for_communities(G, graph_layout, node_positions,
                                                                                    Edge_Betweenness_Choice)
            CUR_PARTITION.append([LC, LC1, LC2])

        print(CUR_PARTITION)
        print(nx.number_connected_components(G))

    else:
        while (nx.number_connected_components(G) < number_of_divisions):
            LC = max(nx.connected_components(G), key=len)

            HIERARCHY, LC1, LC2 = STUDENT_AM_one_shot_girvan_newman_for_communities(G, graph_layout, node_positions,
                                                                                    Edge_Betweenness_Choice)
            CUR_PARTITION.append([LC, LC1, LC2])

        print(CUR_PARTITION)
        print(nx.number_connected_components(G))

    return HIERARCHY


def STUDENT_AM_visualize_communities(G, community_tuples, graph_layout, node_positions):
    print()


def STUDENT_AM_determine_opt_community_structure(G, hierarchy_of_community_tuples):
    min_mod = 0
    min_partition = []
    all_mod = []
    print(hierarchy_of_community_tuples)
    for comm in hierarchy_of_community_tuples:

        current_mod = nx_comm.modularity(G,comm)
        all_mod.append(current_mod)

        if (current_mod < min_mod):
            min_mod = current_mod
            min_partition = comm

    plt.stem(hierarchy_of_community_tuples, all_mod, use_line_collection=True)

    return min_mod, min_partition


if __name__ == "__main__":
    number_of_divisions = 0
    node_name_list = []
    community_tuples = []
    visited = []
    HIERARCHY = []
    node_positions = {}
    graph_layout = 'spring'

    print("(1) Create graph from fb-food data set (fb-pages-food.nodes and fb-pages-food.nodes)\n")
    while True:
        try:
            MAX_NUM_LINKS = int(input("Give a num for links: "))
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    G, node_positions, node_name_list = STUDENT_AM_read_graph_from_csv()

    breakWhileLoop = False

    while not breakWhileLoop:

        print("(2) Add random edges from each node")
        print("(3) Add hamilton cycle (if graph is not connected)")
        print("(4) Compute communities with GIRVAN-NEWMAN")
        print("(5) Compute a binary hierarchy of communities")
        print("(6) Compute modularity-values for all community partitions")
        print("(7) Visualize the communities of the graph")
        print("\n\n")

        while True:
            try:
                my_option_list = int(input('Provide your (case-sensitive) option: '))
                break
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")

        if (my_option_list == 2):
            number_of_Edge = int(input("Give the number of random Edge: "))
            prob = float(input("Give the probability of random Edge: "))
            STUDENT_AM_add_random_edges_to_graph(G, node_name_list, number_of_Edge, prob)

        elif (my_option_list == 3):
            STUDENT_AM_add_hamilton_cycle_to_graph(G, node_name_list)

        elif (my_option_list == 4):
            community_tuples1 = STUDENT_AM_use_nx_girvan_newman_for_communities(G, graph_layout, node_positions)
            community_tuples1 = STUDENT_AM_one_shot_girvan_newman_for_communities(G, graph_layout, node_positions, 1)

        elif (my_option_list == 5):
            while True:
                try:
                    number_of_divisions = int(input("Give the number of Divisions: "))

                    while (number_of_divisions < nx.number_connected_components(G) or (
                            number_of_divisions > min(G.number_of_nodes(), 10 * nx.number_connected_components(G)))):
                        number_of_divisions = int(input("Give the number of Divisions: "))

                    break

                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

            HIERARCHY = STUDENT_AM_divisive_community_detection(G, number_of_divisions, graph_layout, node_positions)

        elif (my_option_list == 6):
            print()



        elif (my_option_list == 7):
            STUDENT_AM_determine_opt_community_structure(G, HIERARCHY)

        else:
            exit(-1)
