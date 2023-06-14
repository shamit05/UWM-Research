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

def add_cube_scatter(fig, x=0, y=0, z=0, rot_x=0, rot_y=0, rot_z=0, size=10, color='black', featureID=''):
    # Define the vertices of the cube
    vertices = (np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1]
    ]) * size) - (size/2)

    # Apply rotation transformations
    rotation_matrix = np.array([
        [np.cos(rot_z)*np.cos(rot_y), np.cos(rot_z)*np.sin(rot_y)*np.sin(rot_x) - np.sin(rot_z)*np.cos(rot_x),
         np.cos(rot_z)*np.sin(rot_y)*np.cos(rot_x) + np.sin(rot_z)*np.sin(rot_x)],
        [np.sin(rot_z)*np.cos(rot_y), np.sin(rot_z)*np.sin(rot_y)*np.sin(rot_x) + np.cos(rot_z)*np.cos(rot_x),
         np.sin(rot_z)*np.sin(rot_y)*np.cos(rot_x) - np.cos(rot_z)*np.sin(rot_x)],
        [-np.sin(rot_y), np.cos(rot_y)*np.sin(rot_x), np.cos(rot_y)*np.cos(rot_x)]
    ])

    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    # Apply translation
    translated_vertices = rotated_vertices + (np.array([x, y, z]))

    fig.add_trace(
        go.Mesh3d(
            # Update the vertices with the rotated and translated coordinates
            x=translated_vertices[:, 0],
            y=translated_vertices[:, 1],
            z=translated_vertices[:, 2],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            showlegend=True,
            legendgroup="Grains",
            hovertemplate=f"""
                x: {x},\n
                y: {y},\n
                z: {z}
            """,
            color= str(color),
            opacity=0.7,
            name='featureID ' + featureID
        )
    )

    return fig

def network_plot_3D(G, angle, save=False):

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    rot = nx.get_node_attributes(G, 'rot')
    size = nx.get_node_attributes(G, 'size')
    surfaceFeature = nx.get_node_attributes(G, 'surfaceFeature')
    # Get number of nodes
    n = G.nodes()

    # Define color range proportional to whether grain is a surface feature or not
    colors = {i: "blue" if surfaceFeature[i] else "red" for i in n}

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
        rx = rot[key][0]
        ry = rot[key][1]
        rz = rot[key][2]

        # Scatter plot
        # fig.add_trace(go.Scatter3d(
        #     x=[xi],
        #     y=[yi],
        #     z=[zi],
        #     mode ='markers',
        #     marker = dict(
        #         symbol='circle',
        #         size=size[key]/2000,
        #         color=colors[key],
        #         opacity=0.7
        #     ),
        #     name='featureID ' + key
        # ))

        # Plot mesh cube in scatter plot
        fig = add_cube_scatter(fig, x=xi, y=yi, z=zi, rot_x=rx, rot_y=ry, rot_z=rz, size=size[key]/2000, color=colors[key], featureID=key)

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

    return fig

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
            # If header, then its the next set of data
            if checkHeader(line):
                i+=1
                headers[i] = line
            else:
                # Gets feature ID to start building adjacency list with networkx
                f = headers[i].index("Feature_ID")
                featureID = line[f]
                if featureID not in adjacency_list:
                    adjacency_list[featureID] = {}
                if i == 1:
                    c0 = headers[i].index("Centroids_0")
                    c1 = headers[i].index("Centroids_1")
                    c2 = headers[i].index("Centroids_2")
                    r0 = headers[i].index("EulerAngles_0")
                    r1 = headers[i].index("EulerAngles_1")
                    r2 = headers[i].index("EulerAngles_2")
                    sf = headers[i].index("SurfaceFeatures")
                    featureID = line[f]
                    pos = (float_or_int(line[c0]), float_or_int(line[c1]), float_or_int(line[c2]))
                    rot = (float_or_int(line[r0]), float_or_int(line[r1]), float_or_int(line[r2]))
                    adjacency_list[featureID]["positionList"] = pos
                    adjacency_list[featureID]["rotationList"] = rot
                    adjacency_list[featureID]["size"] = float_or_int(line[headers[i].index("Volumes")])
                    adjacency_list[featureID]["surfaceFeature"] = bool(float_or_int(line[sf]))
                if i == 2:
                    neighborList = line[2:]
                    adjacency_list[featureID]["neighborList"] = neighborList
                if i == 3:
                    weightsList = [float_or_int(w) for w in line[2:]]
                    global maxWeight
                    maxWeight = max(max(weightsList), maxWeight)
                    adjacency_list[featureID]["weightsList"] = weightsList


    nx_graph = nx.Graph()

    # Iterate through adjacency list to add each node and edge to networkx
    for node, properties in adjacency_list.items():
        nx_graph.add_node(node,
            pos=adjacency_list[node]["positionList"],
            rot=adjacency_list[node]["rotationList"],
            size=adjacency_list[node]["size"],
            surfaceFeature=adjacency_list[node]["surfaceFeature"]
        )

        weightsList = adjacency_list[node]["weightsList"]
        neighborList = adjacency_list[node]["neighborList"]
        for i in range(len(weightsList)):
            nx_graph.add_edge(node, neighborList[i], weight=weightsList[i])

    return nx_graph

def main():
    nx_graph = gen_graph(GEN_STRUCTURES_FILE)
    plot(network_plot_3D(nx_graph, 0))


if __name__ == "__main__":
    main()
