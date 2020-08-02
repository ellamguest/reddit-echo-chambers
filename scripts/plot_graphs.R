library(igraph)
library(ggraph)
library(graphlayouts)
setwd("~/GitHub/thesis/")


# COMMUNITY GRAPH
el = read.csv("output/2019_01/community_edges.csv", row.names = 'X')
nl = read.csv("tables/2019_01/comm_size.csv")
net <- graph_from_data_frame(d = el, directed=F, vertices=nl)
E(net)$log_weight = log(E(net)$weight)
V(net)$log_size = log(V(net)$size)

subgraph = induced.subgraph(net, (V(net)$size > 5))
subgraph = induced.subgraph(subgraph, (E(net)$weight > 5))

width = 10
height = width / 1.62
res=100

png('figures/2019_01/community_graph.png', width=width*res, height=height*res, res=res)

ggraph(subgraph,layout = "manual",node.positions = data.frame(x = V(subgraph)$x, y = V(subgraph)$y)) + 
    geom_edge_fan(aes(alpha = weight,width = weight),
                edge_colour = "#6B6868",n = 2) + 
  scale_edge_width(range = c(0.3,5)) + 
  scale_edge_alpha(range = c(0.5,1)) + 
  geom_node_point(aes(fill = labels, size = size*.75),
                  colour = "#FCFCFC",
                  shape = 21, stroke = 0.3) + 
  scale_fill_brewer(palette = "Pastel1", na.value = "gray53") + 
  scale_size(range = c(10,30)) + 
  geom_node_text(aes(label = labels), colour = "#000000", size = 8, family = "sans") + 
  theme_graph() + 
  theme(legend.position = "none")

dev.off()

# styling
  geom_edge_fan(aes(alpha = weight,width = weight),
                edge_colour = "#6B6868",n = 2) + 
  scale_edge_width(range = c(0.3,5)) + 
  scale_edge_alpha(range = c(0.5,1)) + 
  geom_node_point(aes(fill = labels, size = size*.75),
                  colour = "#FCFCFC",
                  shape = 21, stroke = 0.3) + 
  scale_fill_brewer(palette = "Pastel1", na.value = "gray53") + 
  scale_size(range = c(10,30)) + 
  geom_node_text(aes(label = labels), colour = "#000000", size = 8, family = "sans") + 
  theme_graph() + 
  theme(legend.position = "none")

png('figures/2019_01/community_graph.png', width=width*res, height=height*res, res=res)

ggraph(subgraph,layout = "manual",node.positions = data.frame(x = V(subgraph)$x, y = V(subgraph)$y)) + 
	 geom_edge_fan(aes(alpha = log_weight,width = log_weight),
edge_colour = "#787676",n = 2) + 
	 scale_edge_width(range = c(0.3,1.2)) + 
	 scale_edge_alpha(range = c(0.1,1)) + 
	 geom_node_point(aes(fill = labels, size = log_size),
colour = "#F7F2F2",
shape = 21, stroke = 0.3) + 
	 scale_fill_brewer(palette = "Set1", na.value = "gray53") + 
	 scale_size(range = c(3,20)) + 
	 geom_node_text(aes(label = labels), colour = "#000000", size = 6, family = "sans", repel = TRUE,segment.alpha=0) + 
	 theme_graph() + 
	 theme(legend.position = "none")

dev.off()

# POLITICAL GRAPH

edges = read.csv("output/2019_01/pol_edges.csv", row.names = 'X', stringsAsFactors=T)
nodes = read.csv("output/2019_01/pol_nodes.csv", stringsAsFactors=T)
pol_net <- graph_from_data_frame(d = edges, directed=F, vertices=nodes)

width = 10
height = width / 1.62
res=100

png('figures/2019_01/community_graph.png', width=width*res, height=height*res, res=res)
ggraph(pol_net,layout = "manual",node.positions = data.frame(x = V(pol_net)$x, y = V(pol_net)$y)) + 
	 geom_edge_fan(edge_colour = "#A8A8A8",edge_width = 0.8,edge_alpha = 1,n = 2) + 
	 geom_node_point(aes(fill = topic),
colour = "#000000",
size = 6,
shape = 21, stroke = 0.3) + 
	 scale_fill_brewer(palette = "Dark2", na.value = "gray53") + 
	 geom_node_text(aes(label = name), colour = "#000000", size = 5, family = "sans") + 
	 theme_graph() + 
	 theme(legend.position = "none")

dev.off()
