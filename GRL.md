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
##### Neighborhood Normalization
The basic neighborhood aggregation operation of summing the embeddings can be unstable and sensitive to node degrees. For example, if node $u$ has 100x as many neighbors as node $u'$ then we can expect that 
$$
\left\lVert \sum_{v \in \mathcal{N}(u)} h_v \right\rVert \gg \left\lVert \sum_{v' \in \mathcal{N}(u')} h_v' \right\rVert.
$$
since $|\mathcal{N}(u)| \gg |\mathcal{N}(u')|$. This can lead to numerical instability and slow convergence. To address this, we can normalize the aggregated messages by the degrees of the nodes:
$$
m_{\mathcal{N}(u)} = \frac{1}{|\mathcal{N}(u)|} \sum_{v \in \mathcal{N}(u)} h_v.
$$
This is the same as averaging the embeddings. One other successful normalization factor is the symmetric normalization:
$$
m_{\mathcal{N}(u)} = \frac{\sum_{v \in \mathcal{N}(u)} h_v}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}}.
$$
This method is used in the Graph Convolutional Network (GCN).
It is important to note that normalization can lead to loss of information. For example, after normalization, it can be hard (or even impossible) to use the learned embeddings to distinguish between nodes of different degrees, and various other structural graph features can be obscured by normalization. Usually, normalization is most helpful in tasks where node feature information is far more useful than structural information, or where there is a very wide range of node degrees that can lead to instabilities during optimization.

##### Set Aggregators
The AGGREGATE function can be more complex than simple sum or average. Since the input is a set without any orderings, any aggregation function must be permutation invariant. 
###### Set pooling
One principled approach to define an aggregation function is based on the theory of permutation invariant neural networks. Any permutation invariant function that maps a set of embeddings to a single embedding can be approximated by the model
$$
m_{\mathcal{N}(u)} = \text{MLP}_\theta\left(\sum_{v \in \mathcal{N}(u)} \text{MLP}_\phi(h_v^{(k-1)})\right).
$$
This means that we can directly learn the aggregation function from the data by using MLPs. 

##### Neighborhood Attention



### Theories
A general convolution operation can be defined as 
\[ (f \star g)(x) = \int_{\mathbb{R}^d} f(y)h(x - y)dy = F^{-1}(F(f(x)) \circ F(f(x)) ), \]
where $F$ is the Fourier transform. We can view the discrete convolution $f \star h$ as a filtering operation of the series $f(x_i)$ by a filter $h$. 

For a chain graph, let $A$ be an adjacency matrix and $L$ the normalized Laplacian matrix. Then 
\[ (Af)[t] = f[(t+1) \mod n] \]
and
\[ (Lf)[t] = f[t] - f[(t + 1) \mod n].  \]
Multiplying a signal by the Laplacian computes the difference between the signal at a node and its neighbors.

For arbitrary graph with adjacency matrix $A$, we can represent convolutional filters as matrices of the form
\[ Q_h = \sum_{i=0}^N \alpha_i A^i  \]