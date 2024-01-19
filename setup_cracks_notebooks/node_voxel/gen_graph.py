import networkx as nx
import csv
import pathlib
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from plotly.offline import iplot, plot
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.colors as mcolors

DIR_LOC = os.path.dirname(os.getcwd()) # /Research Directory
GEN_STRUCTURES_FILE = os.path.join(DIR_LOC, "stress_sim", "CellData_FakeMatl_0.csv")
DIR_LOC, GEN_STRUCTURES_FILE

# Function to convert colors to hex
def convert_colors_to_hex(rgba_colors):
    """Convert a list of RGBA colors to hex format."""
    return [mcolors.to_hex(color) for color in rgba_colors]

def float_or_int(string):
    if string.isdigit():
        return int(string)
    elif string.replace('.','',1).isdigit() and string.count('.') < 2:
        return float(string)
    return string

def calculate_von_mises(stresses):
    """Compute von Mises stress given the stress tensor components."""
    s11, s22, s33, s12, s13, s23 = stresses
    return np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s11 - s33)**2 + 6*(s12**2 + s13**2 + s23**2)))

def is_neighbor(centroid1, centroid2):
    """Check if two nodes are neighbors based on their centroids."""
    return np.all(np.abs(centroid1 - centroid2) == [2.5, 0, 0]) or \
           np.all(np.abs(centroid1 - centroid2) == [0, 2.5, 0]) or \
           np.all(np.abs(centroid1 - centroid2) == [0, 0, 2.5])

def add_cube_scatter(fig, x=0, y=0, z=0, rot_x=0, rot_y=0, rot_z=0, size=2.5, color='black', alpha=0.7, featureID=''):
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
            opacity=alpha,
            name='featureID ' + str(featureID)
        )
    )

    return fig

def network_plot_3D(G, colors):

    # Get node positions
    pos = nx.get_node_attributes(G, 'centroid')
    rot = nx.get_node_attributes(G, 'eulerangles')
    featureid = nx.get_node_attributes(G, 'featureid')
    # Get number of nodes
    n = list(G.nodes())
    # Define color range proportional to grain ID
    alphas = {}
    for i in n:
        alphas[i] = 0.2

    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

    layout = go.Layout(
        title="Network of grains in microstructure (3D visualization)",
        scene=dict(
                 aspectmode='data'
         )
    )

    fig = go.Figure(layout=layout)

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    for i in n:
        key, value = list(pos.items())[i]
        xi = value[0]
        yi = value[1]
        zi = value[2]
        rx = rot[key][0]
        ry = rot[key][1]
        rz = rot[key][2]
        feature = featureid[key]

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
        if xi < 0.1:
            continue
        fig = add_cube_scatter(fig, x=xi, y=yi, z=zi, rot_x=rx, rot_y=ry, rot_z=rz, size=1, color=colors[key], alpha=alphas[key], featureID=feature)
        print("plotting node", key, end="\r")
        if i > 100:
            break

    # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
    # Those two points are the extrema of the line to be plotted
    # for j in G.edges.data("diffNeighbour", default=0):
    #     x = np.array((pos[j[0]][0], pos[j[1]][0]))
    #     y = np.array((pos[j[0]][1], pos[j[1]][1]))
    #     z = np.array((pos[j[0]][2], pos[j[1]][2]))
    #
    #     fig.add_trace(go.Scatter3d(
    #         x=x,
    #         y=y,
    #         z=z,
    #         mode='lines',
    #         line=dict(color='red' if j[2] else 'black'),
    #         hoverinfo='none',
    #         opacity=0.2,
    #         name='edge from ' + str(j[0]) + ' to ' + str(j[1]),
    #         legendgroup="Edges"
    #     ))

    return fig

def gen_graph(input_file):

    # Load the data
    df = pd.read_csv(input_file)

    # Create a graph from the dataframe
    G = nx.Graph()

    # Add nodes to the graph
    for index, row in df.iterrows():
        G.add_node(index, centroid=row[['X', 'Y', 'Z']].values,
                   stress=row[[' S11', ' S22', ' S33', ' S12', ' S13', ' S23']].values,
                   strain=row[[' E11', ' E22', ' E33', ' E12', ' E13', ' E23']].values,
                   boundarycell=row['BoundaryCells'],
                   eulerangles=row[['EulerAngles_0', 'EulerAngles_1', 'EulerAngles_2']].values,
                   featureid=row['FeatureIds'])
        print("Adding node", index, end="\r")

    # Use hashmap for efficient lookups
    centroid_to_node = {tuple(data['centroid']): node for node, data in G.nodes(data=True)}

    # Define possible neighbor offsets for a voxel
    offsets = [(2.5, 0, 0), (-2.5, 0, 0),
               (0, 2.5, 0), (0, -2.5, 0),
               (0, 0, 2.5), (0, 0, -2.5)]

    # Add edges to the graph based on the hashmap
    i = 0
    for node, data in G.nodes(data=True):
        for offset in offsets:
            neighbor_centroid = tuple(data['centroid'] + np.array(offset))
            if neighbor_centroid in centroid_to_node:
                neighbor_node = centroid_to_node[neighbor_centroid]
                if node < neighbor_node:  # Ensure unique edges
                    diffNeighbour = 0 if data['featureid'] == G.nodes[neighbor_node]['featureid'] else 1
                    G.add_edge(node, neighbor_node, diffNeighbour=diffNeighbour)
                    print("Adding edge", i, end="\r")
                    i+=1

    # Calculate von Mises stresses
    von_mises_stresses = [calculate_von_mises(data['stress']) for _, data in G.nodes(data=True)]

    # Map the von Mises stresses to colors
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=min(von_mises_stresses), vmax=max(von_mises_stresses))
    colors = [cmap(norm(stress)) for stress in von_mises_stresses]
    print(G)

    return G, convert_colors_to_hex(colors)

nx_graph, colors = gen_graph(GEN_STRUCTURES_FILE)
plot(network_plot_3D(nx_graph, colors))