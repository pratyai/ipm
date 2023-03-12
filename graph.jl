module Graphs

using FromFile
using SparseArrays
using Graphs

struct Graph
  n::Int
  m::Int
  EdgeList::Matrix{Int}
  IncidenceMatrix::AbstractMatrix{Int}
  AdjacencyMatrix::AbstractMatrix{Int}
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
  I = vcat(E[1:m, 1], E[1:m, 2])
  J = vcat(1:m, 1:m)
  V = vcat(-ones(m), ones(m))
  A = sparse(I, J, V)
  return A
end

function MakeAdjacencyMatrix(n::Int, E::AbstractMatrix{Int}, w::AbstractVector{<:Number})
  m = size(E, 1)
  @assert size(E) == (m, 2)
  @assert size(w) == (m,)
  I = E[1:m, 1]
  J = E[1:m, 2]
  V = w
  Adj = sparse(I, J, V)
  return Adj
end

function MakeSymmetricAdjacencyMatrix(
  n::Int,
  E::AbstractMatrix{Int},
  w::AbstractVector{<:Number} = ones(Int, size(E, 1)),
)
  m = size(E, 1)
  @assert size(E) == (m, 2)
  @assert size(w) == (m,)
  I = vcat(E[1:m, 1], E[1:m, 2])
  J = vcat(E[1:m, 2], E[1:m, 1])
  V = vcat(w, w)
  Adj = sparse(I, J, V)
  return Adj
end

struct McfpNet
  G::Graph
  Cost::AbstractVector{Int}
  Cap::AbstractVector{Int}
  Demand::AbstractVector{Int}
end

end  # module Graph
