import numpy as np
import networkx as nx
import random


class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(
						cur_nbrs[alias_draw(alias_nodes[cur][0], 
											alias_nodes[cur][1])
						]
						"""
							结合cur_nbrs = sorted(G.neighbor(cur)) 和 alias_nodes/alias_edges的序号，
							才能确定节点的ID。
							所以路径上的每个节点在确定下一个节点是哪个的时候，都要经过sorted(G.neighbors(cur))这一步。
						"""
					)
				else:
					prev = walk[-2]
					next = cur_nbrs[
									alias_draw(alias_edges[(prev, cur)][0], 
												alias_edges[(prev, cur)][1])
									]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print 'Walk iteration:'
		for walk_iter in range(num_walks):
			print str(walk_iter+1), '/', str(num_walks)
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		"""
			对边的权重，变化后的权重，再进行归一化了。
			生成的normalized_probs，是对一条边的终点的下一步节点转移的归一化概率的列表
		"""

		return alias_setup(normalized_probs)
		"""
			返回的参数，第一个返回值是别名列表，第二个返回值是到各个节点的转移概率。
		"""

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			"""
				G.neighbors(node)得到一个邻接节点列表
				这里面sorted(G.neighbors(node))的用法，非常重要
				unnormailized_probs是一个列表，列表的每个元素是边上的权重
			"""
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			"""
				这样就得到了归一化的、到各点的转移概率
			"""
			alias_nodes[node] = alias_setup(normalized_probs)
			"""
				alias_nodes[node]是一个字典，字典的value，又是一个包含两个元素的元组(tuple): 第一个元素是别名列表，第二个元素是到各个节点的转移概率。

			"""

		alias_edges = {}
		"""
			alias_edges是字典
		"""
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				"""
					在alias_edges字典中加入项

					alias_edges[edge]是一个字典，字典的value，又是一个包含两个元素的元组(tuple): 
					第一个元素是边的终点的邻接节点转移概率对应的别名列表(sorted的前后序号，不是节点序号)，第二个元素是边的终点，到各个节点的转移概率。
				"""
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges
		"""
			这里面alias_nodes和alias_edges都是字典，字典的元素组成都是 '节点序号':'(邻接节点转移概率对应的别名列表(sorted的前后序号，不是节点序号), 到邻接节点的转移概率)' 的形式。
			不同的是，alias_edges中，'到邻接节点的转移概率'这个取值，有p或q的影响。
		"""
		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details

	the input list probs, in this using context, is the normalized weight of nodes already sorted in ID order.
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)
	"""
		J is the alias array, recording the alias's corresponding discrete integer
	"""

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)
	"""
		this part of code, needs to prove that, the alias array will eventually form and exit the loop.
		however, see the code of line 147 and line 148, that makes most of the alias method outstanding strength.
	"""
	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]