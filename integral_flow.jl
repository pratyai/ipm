#=
"Round" a fractional flow to a rounded integral flow with equal or lower cost.
Assumes that flow are perfectly balanced, i.e. Ax = b. Returns a balanced integral flow.
ref: https://arxiv.org/pdf/1507.08139.pdf (section 4)
=#

using FromFile

@from "graph.jl" import Graphs.McfpNet

import Base.@kwdef
using Setfield
using DataStructures
using LinkCutTrees

@kwdef struct Solver
  netw::McfpNet
  nodes::AbstractVector{LinkCutTreeNode}
  x::AbstractVector{<:Number}
end

# Find if `u` and `v` are already on the same tree in the forest.
function are_connected(s::Solver, u::Int, v::Int)
  nodes = s.nodes
  return LinkCutTrees.find_root(nodes[u]) === LinkCutTrees.find_root(nodes[v])
end

# Move the node `u` to the root of the tree.
function make_root(s::Solver, u::Int)
  netw, nodes = s.netw, s.nodes
  G = netw.G

  pe = edge_label(nodes[u])  # edge toward parent
  if pe == nothing
    # already in root.
    return
  end

  v, w = G.EdgeList[pe, :]
  @assert u == v || u == w
  if u == v
    v, w = w, v
  end
  # now we're taking about `(u, v)` edge.

  # disconnect from parent
  cut!(nodes[u])
  # make sure `v` is root in its own tree.
  make_root(s, v)
  # link `v` and its subtree under `u`.
  link!(nodes[v], nodes[u], pe, netw.Cost[pe])
end

# Find the sequence of vertices on the path toward the root.
function root_path(s::Solver, u::Int)
  u = s.nodes[u]

  path = []
  while u != nothing
    append!(path, label(u))
    u = LinkCutTrees.parent(u)
  end
  return path
end

# Find the sequence of vertices on the path from `u` to `v`.
function tree_path(s::Solver, u::Int, v::Int)
  make_root(s, v)
  path = root_path(s, u)
  @assert last(path) == v
  return path
end

# Find cost and lowest availability along the directed path from `u` to `v`.
function cost_avail_on_path(s::Solver, u::Int, v::Int, avail_uv::Number)
  netw, nodes, x = s.netw, s.nodes, s.x
  G = netw.G

  cost, avail = 0, avail_uv
  make_root(s, v)
  u = nodes[u]
  while LinkCutTrees.parent(u) != nothing
    e = edge_label(u)
    p, q = G.EdgeList[e, :]
    @assert p == label(u) || q == label(u)
    if p == label(u)
      # can send more flow
      cost += netw.Cost[e]
      avail = min(avail, ceil(x[e]) - x[e])
    else
      # can send less flow
      cost -= netw.Cost[e]
      avail = min(avail, x[e] - floor(x[e]))
    end
    u = LinkCutTrees.parent(u)
  end

  return cost, avail
end

function very_integral(x::Number, EPS::Number = 1e-9)   # EPS = 1/m ?
  return abs(round(x) - x) < EPS
end

# Cancel fractional flow on the path between `u` and `v`, whichever direction is cheaper.
function cancel_fractional_flow_between(
  s::Solver,
  u::Int,
  v::Int,
  avail_uv::Number,
  cost_uv::Int,
)
  netw, nodes, x = s.netw, s.nodes, s.x
  G = netw.G

  u, v, by = let
    cuv, auv = cost_avail_on_path(s, u, v, avail_uv)
    cuv += cost_uv
    cvu, avu = cost_avail_on_path(s, v, u, 1 - avail_uv)
    cvu -= cost_uv
    cuv <= cvu ? (u, v, auv) : (v, u, avu)
  end

  let u = u, v = v, by = by
    make_root(s, v)
    u = nodes[u]
    while LinkCutTrees.parent(u) != nothing
      up = LinkCutTrees.parent(u)
      e = edge_label(u)
      p, q = G.EdgeList[e, :]
      @assert p == label(u) || q == label(u)
      if p == label(u)
        # send more flow
        x[e] += by
      else
        # send less flow
        x[e] -= by
      end
      @assert x[e] >= 0
      @assert x[e] <= netw.Cap[e]
      if very_integral(x[e])
        cut!(u)
      end
      u = up
    end
  end

  return u, v, by
end

function compute_integral_flow(netw::McfpNet, x::AbstractVector{<:Number})
  G = netw.G
  s = Solver(netw = netw, nodes = [make_tree(Int, Int, Int, i) for i = 1:G.n], x = x)
  nodes = s.nodes

  for i = 1:G.m
    if very_integral(x[i])
      continue
    end

    u, v = G.EdgeList[i, :]
    if are_connected(s, u, v)
      # can send on u-v tree path only what can be send on this arc in reverse.
      p, q, by = cancel_fractional_flow_between(s, u, v, x[i] - floor(x[i]), -netw.Cost[i])
      @assert (u, v) == (p, q) || (u, v) == (q, p)
      if (u, v) == (p, q)
        x[i] -= by
      else
        x[i] += by
      end
      @assert x[i] >= 0
      @assert x[i] <= netw.Cap[i]
    end

    if !very_integral(x[i])
      @assert !are_connected(s, u, v)
      # make sure `v` is root in its own tree.
      make_root(s, v)
      # link `v` and its subtree under `u`.
      link!(nodes[v], nodes[u], i, netw.Cost[i])
    end
  end
  return Int.(round.(x))
end
