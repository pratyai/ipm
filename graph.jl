module Graphs

using FromFile
using SparseArrays
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

end  # module Graph
