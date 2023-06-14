import networkx as nx
import csv
import pathlib
import os
import numpy as np

import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go

DIR_LOC = pathlib.Path(__file__).parents[1] # /Research Directory
GEN_STRUCTURES_FILE = os.path.join(DIR_LOC, "generated_microstructures", "FeatureData_FakeMatl_0.csv")
maxWeight = 1

def checkHeader(line):
    for a in line:
        return not a.isdigit() and not a.replace('.','',1).isdigit() and a.count('.') < 2
    return False

def float_or_int(string):
    if string.isdigit():
        return int(string)
    elif string.replace('.','',1).isdigit() and string.count('.') < 2:
        return float(string)
    return string

def network_plot_3D(G, angle, save=False):

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    size = nx.get_node_attributes(G, 'size')
    # Get number of nodes
    n = G.nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in n])

    # Define color range proportional to number of edges adjacent to a single node
    colors = {i: plt.cm.plasma(G.degree(i)/edge_max) for i in n}

    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

    layout = go.Layout(
        title="Network of grains in microstructure (3D visualization)",
    )

    fig = go.Figure(layout=layout)

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    for key, value in pos.items():
        xi = value[0]
        yi = value[1]
        zi = value[2]

        # Scatter plot
        fig.add_trace(go.Scatter3d(
            x=[xi],
            y=[yi],
            z=[zi],
            mode ='markers',
            marker = dict(
                symbol='circle',
                size=size[key]/100,
                color=colors[key],
                opacity=0.7
            ),
            name='featureID ' + key,
            legendgroup="Grains"
        ))

    # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
    # Those two points are the extrema of the line to be plotted

    for j in G.edges.data("weight", default=1):
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color='black', width=j[2]/maxWeight*10),
            hoverinfo='none',
            opacity=0.2,
            name='edge from ' + str(j[0]) + ' to ' + str(j[1]),
            legendgroup="Edges"
        ))

    plot(fig)

def gen_graph(input_file):

    # Data:
    headers = {
        0: "Feature_ID"
    }
    adjacency_list = {}

    # opening the CSV file
    with open(input_file, mode ='r') as file:

        # reading the CSV file
        csvFile = csv.reader(file)

        # displaying the contents of the CSV file
        i = 0
        for line in csvFile:
            if checkHeader(line):
                i+=1
                headers[i] = line
            else:
                f = headers[i].index("Feature_ID")
                featureID = line[f]
                if featureID not in adjacency_list:
                    adjacency_list[featureID] = {}
                if i == 1:
                    c0 = headers[i].index("Centroids_0")
                    c1 = headers[i].index("Centroids_1")
                    c2 = headers[i].index("Centroids_2")
                    featureID = line[f]
                    pos = (float_or_int(line[c0]), float_or_int(line[c1]), float_or_int(line[c2]))
                    adjacency_list[featureID]["positionList"] = pos
                    adjacency_list[featureID]["size"] = float_or_int(line[headers[i].index("Volumes")])
                if i == 2:
                    neighborList = line[2:]
                    adjacency_list[featureID]["neighborList"] = neighborList
                if i == 3:
                    weightsList = [float_or_int(w) for w in line[2:]]
                    global maxWeight
                    maxWeight = max(max(weightsList), maxWeight)
                    adjacency_list[featureID]["weightsList"] = weightsList


    nx_graph = nx.Graph()

    for node, properties in adjacency_list.items():
        nx_graph.add_node(node, pos=adjacency_list[node]["positionList"], size=adjacency_list[node]["size"])

        weightsList = adjacency_list[node]["weightsList"]
        neighborList = adjacency_list[node]["neighborList"]
        for i in range(len(weightsList)):
            nx_graph.add_edge(node, neighborList[i], weight=weightsList[i])

    return nx_graph

def main():
    nx_graph = gen_graph(GEN_STRUCTURES_FILE)
    network_plot_3D(nx_graph, 0)

if __name__ == "__main__":
    main()
