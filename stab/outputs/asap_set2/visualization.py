from pyvis.network import Network
import json

# used for each essay offset in classification_predictions array
c_index = 0
#0->17
#18
#19-->26

essay = "2978.txt"

relations_file = "./optimized_relations.json"
classification_file = "./classification_predictions.json"
essay_file = f"./classification/{essay}.json"

type_dict = {0: "Claim", 1:"Claim", 2:"Premise"}
color_dict = {0: "#FF0000", 1:"#FF0000", 2:"#0000FF"}
nodes = [] # component indices for this essay
labels = [] # claim or premise for this essay
colors = [] # colors of each node for this essay
edges = [] # [(1, 7)]
hover_text = [] # component_text for this essay

nodes_with_outgoing = []

def get_components(essay):
    # Essay file: {0: ..., index, component_text, ...; 1, ...}
    with open(essay) as e_file:
        components = json.load(e_file)
    with open(classification_file) as c_file:
        classifications = json.load(c_file)
    for i in range(len(components)):
        nodes.append(components[i]["index"])
        hover_text.append(components[i]["component_text"])
        labels.append(str(components[i]["index"]) + type_dict.get(classifications[i + c_index]))
        colors.append(color_dict.get(classifications[i + c_index]))

def get_relations():
    with open(relations_file) as r_file:
        data = json.load(r_file)
    for str in data[essay]["relations"].keys():
        s, d = map(int, str.split(","))
        edges.append((s, d))
        nodes_with_outgoing.append(s)

def main():
    get_relations()
    get_components(essay_file)
    # get all the claims
    claims = set(nodes)
    print(claims)
    print(nodes_with_outgoing)
    claims -= set(nodes_with_outgoing)
    print(claims)
    for x in claims:
        colors[x-1] = "#FF0000"
        labels[x-1] = str(x) + "Claim"
    
    print(colors)
    print(labels)
    net = Network(directed=True)
    net.add_nodes(nodes, label=labels, color=colors, title=hover_text)
    net.add_edges(edges)
    net.show("visualize.html", notebook=False)


if __name__ == "__main__":
    main()