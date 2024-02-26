### GNN Model
#### Neural Message Passing
##### Message passing framework
For each message passing iteration in a GNN, a hidden embedding $h_u^{(k)}$ correspond to each node $u$ is updated according to information aggregrated from $u$'s neighbors. The update rule can be formulated as:
$$\begin{align*}
h_u^{(k+1)} &= \text{UPDATE}^{(k)}(h_u^{(k)}, \text{AGGREGATE}^{(k)}(\{h_v^{(k)}: v \in \mathcal{N}(u)\})) \\
&= \text{UPDATE}^{(k)}(h_u^{(k)}, m_{\mathcal{N}(u)}^{(k)}).
\end{align*}$$
where UPDATE and AGGREGATE are differentiable functions and $m_{\mathcal{N}(u)}^{(k)}$ is the messaged that is aggregated from $u$'s neighbors. 

At each iteration, the AGGREGATE function takes as input the set of embeddings of the nodes in the neighborhood of $u$, $\{h_v^{(k)}: v \in \mathcal{N}(u)\}$, and produces a message $m_{\mathcal{N}(u)}^{(k)}$ that is then used by UPDATE with the previous embedding, $h_u^{(k)}$, to produce the new embedding $h_u^{(k+1)}$. After $K$ iterations, we can use the output of the final layer to define the embeddings for each node. Note that the GNNs defined in this way are permutation invariant, i.e., the embeddings of the nodes are invariant to the order of the nodes in the input graph.

##### Intuition
At each iteration, every node aggregates information from its local
neighborhood, and as these iterations progress each node embedding contains
more and more information from further reaches of the graph. 

These embeddings encode structural information about the graph. For example, after $k$ iterations of message passing, the embedding of node $u$ might encode information about the degrees of all the neighbors in $u$'s $k$-hop neighborhood.
The embeddings can also capture feature-based information. After $k$ iterations of message passing, the embedding of node $u$ might encode information about the features of all the nodes in $u$'s $k$-hop neighborhood. This is analogous to the behavior of a convolutional layer in a CNN.

##### Basic GNN
The basic GNN message passing can be defined as:
$$
h_u^{(k)} = \sigma\left(W_{self}^{(k)} h_u^{(k-1)} + W_{neigh}^{(k)} \sum_{v \in \mathcal{N}(u)} h_v^{(k-1)} + b^{k} \right),
$$
where $W_{self}^{(k)}$ and $W_{neigh}^{(k)}$ are learnable weight matrices, $b^{(k)}$ is a learnable bias, and $\sigma$ is an element wise non-linear function.

This can be equivalently define the basic GNN through UPDATE and AGGREGATE:
$$\begin{align*}
&m_{\mathcal{N}(u)}^{(k)} = \sum_{v \in \mathcal{N}(u)} h_v = \text{AGGREGATE}^{(k)}({h_v^{(k)}, \forall v \in \mathcal{N}(u)}),\\
&\text{UPDATE}(h_u, m_{\mathcal{N}(u)}) = \sigma\left(W_{self} h_u + W_{neigh} m_{\mathcal{N}(u)} + b\right).
\end{align*}$$
In this case, the aggregation function is a simple sum and the update function is a simple neural network layer.
Described above are the node-level equations, for graph-level, we can define the graph-level embedding as:
$$ 
H^{(t)} = \sigma \left( AH^{(k - 1)}W_{neigh}^{(k)} + H^{(k-1)}W_{self}^{(k)} \right)
$$
where $H^{(k)} \in \mathbb{R}^{|V| \times d}$ denotes the matrix of node representation at layer $t$, $A$ is the adjacency matrix.

To simplify the message passing approach, it is common to add self-loops to omit the update step. We define the message passing simply as 
$$
h_u^{(k)} = \text{AGGREGATE}^{(k)}(\{h_v^{(k-1)}: v \in \mathcal{N}(u) \cup \{u\} \}).
$$

#### Generalized Neighborhood Aggregation
