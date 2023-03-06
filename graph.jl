using SparseArrays
using DataStructures
using LinkCutTrees
using Laplacians
using Graphs

struct Graph
  n::Int
  m::Int
  EdgeList::Matrix{Int}
  IncidenceMatrix::Matrix{Int}
  AdjacencyMatrix::Matrix{Int}
end

function ToStandard(g::Graph)::DiGraph
  G = DiGraph()
  add_vertices!(G, g.n)
  for e = 1:g.m
    (p, q) = g.EdgeList[e, :]
    add_edge!(G, p, q)
  end
  return G
end


function FromEdgeList(n::Int, E::AbstractMatrix{Int})
  m = size(E, 1)
  @assert size(E) == (m, 2)
  return Graph(n, m, E, MakeIncidenceMatrix(n, E), MakeAdjacencyMatrix(n, E, ones(m)))
end

function MakeIncidenceMatrix(n::Int, E::AbstractMatrix{Int})
  m = size(E, 1)
  @assert size(E) == (m, 2)
  A = zeros(Int, n, m)
  for (i, (u, v)) in enumerate(eachrow(E))
    A[[u, v], i] = [-1, 1]
  end
  return sparse(A)
end

function MakeAdjacencyMatrix(n::Int, E::AbstractMatrix{Int}, w::AbstractVector{<:Number})
  m = size(E, 1)
  @assert size(E) == (m, 2)
  Adj = zeros(eltype(w), n, n)
  for (i, (u, v)) in enumerate(eachrow(E))
    Adj[u, v] = w[i]
  end
  return sparse(Adj)
end

function MakeSymmetricAdjacencyMatrix(
  n::Int,
  E::AbstractMatrix{Int},
  w::AbstractVector{<:Number} = ones(Int, size(E, 1)),
)
  Adj = zeros(eltype(w), n, n)
  for (i, (u, v)) in enumerate(eachrow(E))
    Adj[u, v] = w[i]
    Adj[v, u] = w[i]
  end
  return sparse(Adj)
end

function ComputeTreeSolution(
  n::Int,
  EdgeList::AbstractMatrix{Int},
  demand::AbstractVector{<:Number},
)
  T = prim(MakeSymmetricAdjacencyMatrix(n, EdgeList))

  z = zeros(n, n)
  vis = zeros(Int, n)

  stk = Stack{Int}()
  parent = zeros(Int, n)
  b = deepcopy(demand)

  # make 1 root
  push!(stk, 1)
  while !isempty(stk)
    top = pop!(stk)
    if top > 0
      if vis[top] > 0
        continue
      end
      vis[top] = 1
      # first time we're seeing top
      push!(stk, -top)
      for nei in rowvals(T[top, :])
        # hope that we don't have a cycle
        if vis[nei] > 0
          continue
        end
        parent[nei] = top
        push!(stk, nei)
      end
    else
      top = -top
      vis[top] = 2
      # unmet demand so far must be met by parent
      if parent[top] == 0
        # except for root, because it has no parent and is supposed to not have any unmet dependency by now.
        continue
      end
      ub = b[top]
      b[parent[top]] += ub
      b[top] = 0
      if ub > 0
        z[parent[top], top] = abs(ub)
      else
        z[top, parent[top]] = abs(ub)
      end
    end
  end
  @assert all(b .< 1e-16)

  return sparse(z), T
end

struct McfpNet
  G::Graph
  Cost::AbstractVector{Int}
  Cap::AbstractVector{Int}
  Demand::AbstractVector{Int}
end

struct AuxiliaryNet
  G::Graph
  Cost::AbstractVector{Int}
  Demand::AbstractVector{Int}
end

struct AuxiliarySolutions
  # arc flow
  x::AbstractVector{<:Number}
  # node potential
  y::AbstractVector{<:Number}
  # arc slack
  s::AbstractVector{<:Number}
end

#=
"Round" a fractional flow to a rounded integral flow with equal or lower cost.
Assumes that flow are balanced, i.e. Ax = b. Returns a balanced integral flow regardless.
ref: https://arxiv.org/pdf/1507.08139.pdf (section 4)
=#
function ComputeIntegralFlow(netw::McfpNet, x::AbstractVector{<:Number})
  G = netw.G
  EPS = 1e-10  # 1/m ??
  aG = [make_tree(Int, Int, Int, i) for i = 1:G.n]

  # move a certain node to the root
  splay = function (u)
    par = edge_label(aG[u])
    if par == nothing
      return
    end
    v, w = G.EdgeList[par, :]
    @assert u == v || u == w
    if u == w
      v, w = w, v
    end
    cut!(aG[u])
    splay(w)
    link!(aG[w], aG[u], par, netw.Cost[par])
  end
  # find path to the root in the tree
  rootPath = function (u)
    anc = []
    while u != nothing
      append!(anc, label(u))
      u = LinkCutTrees.parent(u)
    end
    return anc
  end
  # find path between two nodes in the tree
  treePath = function (j, k)
    jAnc = reverse(rootPath(j))
    kAnc = reverse(rootPath(k))
    lca = 1
    while lca < length(jAnc) && lca < length(kAnc) && jAnc[lca+1] == kAnc[lca+1]
      lca += 1
    end
    path = vcat(reverse(jAnc[lca:length(jAnc)]), kAnc[lca+1:length(kAnc)])
    lca = length(jAnc) - lca + 1
    return lca, path
  end
  # find cost and minimum availability along the directed path
  cost_avail = function (lca, path)
    cost, avail = 0, Inf
    for i = 1:(lca-1)
      j = edge_label(aG[path[i]])
      u, v = G.EdgeList[j, :]
      @assert u == path[i] || v == path[i]
      if u == path[i]
        cost += netw.Cost[j]
        avail = min(avail, ceil(x[j]) - x[j])
      else
        cost -= netw.Cost[j]
        avail = min(avail, x[j] - floor(x[j]))
      end
    end
    for i = lca+1:length(path)
      j = edge_label(aG[path[i]])
      u, v = G.EdgeList[j, :]
      @assert u == path[i] || v == path[i]
      if v == path[i]
        cost += netw.Cost[j]
        avail = min(avail, ceil(x[j]) - x[j])
      else
        cost -= netw.Cost[j]
        avail = min(avail, x[j] - floor(x[j]))
      end
    end
    return cost, avail
  end
  # cancel fractional flow
  cancel = function (lca, path, avail)
    cuts = []
    for i = 1:(lca-1)
      j = edge_label(aG[path[i]])
      u, v = G.EdgeList[j, :]
      @assert u == path[i] || v == path[i]
      if u == path[i]
        x[j] += avail
        if abs(x[j] - ceil(x[j])) < EPS
          append!(cuts, path[i])
        end
      else
        x[i] -= avail
        if abs(x[j] - floor(x[j])) < EPS
          append!(cuts, path[i])
        end
      end
    end
    for i = lca+1:length(path)
      j = edge_label(aG[path[i]])
      u, v = G.EdgeList[j, :]
      @assert u == path[i] || v == path[i]
      if v == path[i]
        x[j] += avail
        if abs(x[j] - ceil(x[j])) < EPS
          append!(cuts, path[i])
        end
      else
        x[i] -= avail
        if abs(x[j] - floor(x[j])) < EPS
          append!(cuts, path[i])
        end
      end
    end
    return cuts
  end

  for i = 1:G.m
    if abs(x[i] - round(x[i])) < EPS
      x[i] = round(x[i])
      continue
    end

    u, v = G.EdgeList[i, :]
    if LinkCutTrees.find_root(aG[u]) === LinkCutTrees.find_root(aG[v])
      lca, path = treePath(aG[u], aG[v])
      cost, avail = cost_avail(lca, path)
      cost -= netw.Cost[i]
      avail = min(avail, x[i] - floor(x[i]))
      flipped = true
      if cost > 0
        lca = length(path) - lca + 1
        path = reverse(path)
        cost, avail = cost_avail(lca, path)
        cost += netw.Cost[i]
        avail = min(avail, ceil(x[i]) - x[i])
        flipped = false
      end
      cuts = cancel(lca, path, avail)
      if flipped
        x[i] -= avail
      else
        x[i] += avail
      end
      if length(cuts) == 0
        @show x[i]
        @assert abs(x[i] - round(x[i])) < EPS
      else
        for k in cuts
          cut!(aG[k])
        end
      end
    end

    if abs(x[i] - round(x[i])) < EPS
      x[i] = round(x[i])
    else
      splay(u)
      link!(aG[u], aG[v], i, netw.Cost[i])
    end
  end
  return Int.(round.(x))
end
