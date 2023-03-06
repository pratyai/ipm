include("graph.jl")

using LinearAlgebra

mutable struct Solver
  netw::McfpNet
  x::AbstractVector{<:Number}
  y::AbstractVector{<:Number}
  z_l::AbstractVector{<:Number}
  z_u::AbstractVector{<:Number}
  eps::Number
end

function objective(s::Solver)
  return s.netw.Cost' * s.x
end

function primal_dual_search_dir(s::Solver)
  netw = s.netw
  G = netw.G
  A = G.IncidenceMatrix
  c = netw.Cost
  b = netw.Demand
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps = s.eps

  eta = -[(-x .- eps); (x - netw.Cap .- eps)]' * [z_l; z_u]  # beware of pointwise ops
  mu = 1e-1 * eta / G.m
  @show mu

  S_l = diagm(z_l ./ (x .+ eps))  # beware of pointwise ops
  S_u = diagm(z_u ./ (netw.Cap .+ eps - x))  # beware of pointwise ops
  S = S_l + S_u
  KKT = [
    S A'
    A 0I
  ]
  rd = c - z_l + z_u + A' * y
  rp = A * x - b
  rc_l = (x .+ eps) .* z_l .- mu  # beware of pointwise ops
  rc_u = (netw.Cap .+ eps - x) .* z_u .- mu  # beware of pointwise ops
  rdc = rd + rc_l ./ (x .+ eps) + rc_u ./ (netw.Cap .+ eps - x)  # beware of pointwise ops
  r = -[rdc; rp]

  @show KKT r
  #=
  L = A * A'
  Li = pinv(L)
  dy = Li * (rp - A * (S \ rdc))
  dx = S \ (-rdc - A' * y)
  =#
  dxy = pinv(KKT) * r
  dx, dy = dxy[1:G.m], dxy[G.m+1:G.m+G.n]
  dz_l = -(rc_l + dx .* z_l) ./ (x .+ eps)  # beware of pointwise ops
  dz_u = (-rc_u + dx .* z_u) ./ (netw.Cap .+ eps - x)  # beware of pointwise ops
  return dx, dy, dz_l, dz_u
end

function decide_steplength(
  s::Solver,
  dx::AbstractVector{<:Number},
  dy::AbstractVector{<:Number},
  dz_l::AbstractVector{<:Number},
  dz_u::AbstractVector{<:Number},
)
  netw = s.netw
  G = netw.G
  A = G.IncidenceMatrix
  c = netw.Cost
  b = netw.Demand
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u

  #=
  @show G.IncidenceMatrix * x
  @show netw.Cost' * x
  @show G.IncidenceMatrix * (x + dx)
  @show netw.Cost' * (x + dx)
  =#

  ax = minimum([dx[i] > 0 ? netw.Cap[i] - x[i] : x[i] for i = 1:G.m] ./ abs.(dx))
  ay = 1.0
  az_l = minimum([dz_l[i] < 0 ? z_l[i] : Inf for i = 1:G.m] ./ abs.(dz_l))
  az_u = minimum([dz_u[i] < 0 ? z_u[i] : Inf for i = 1:G.m] ./ abs.(dz_u))
  @show ax ay az_l az_u
  ax, ay, az_l, az_u = min(1, ax), min(1, ay), min(1, az_l), min(1, az_u)
  return ax, ay, az_l, az_u
end
