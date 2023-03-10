module MehrotraPredictorCorrector

using FromFile

@from "graph.jl" import Graphs.McfpNet
using LinearAlgebra
using Statistics
import Base.@kwdef
using Setfield

@kwdef struct Solver
  # fixed parameters for the problem
  A::AbstractMatrix{<:Number}
  b::AbstractVector{<:Number}
  c::AbstractVector{<:Number}

  # iterates
  x::AbstractVector{<:Number}
  y::AbstractVector{<:Number}  # \lambda
  s::AbstractVector{<:Number}
end

function kkt_residual(s::Solver)
  rd = s.A' * s.y + s.s - s.c  # dual
  rp = s.A * s.x - s.b  # primal
  rc = s.x .* s.s  # center
  return rd, rp, rc
end

function big_kkt_solve(
  s::Solver,
  rd::AbstractVector{<:Number},
  rp::AbstractVector{<:Number},
  rc::AbstractVector{<:Number},
)
  n, m = length(rd), length(rp)  # var, con
  S, X = diagm(s.s), diagm(s.x)
  Z = zeros(m, n)
  KKT = [
    0I s.A' I
    s.A 0I Z
    S Z' X
  ]
  r = -[rd; rp; rc]
  dxys = pinv(KKT) * r
  @debug "[kkt error]" (KKT * dxys - r)
  dx, dy, ds = dxys[1:n], dxys[n+1:n+m], dxys[n+m+1:n+m+n]
  return dx, dy, ds
end

function small_kkt_solve(
  s::Solver,
  rd::AbstractVector{<:Number},
  rp::AbstractVector{<:Number},
  rc::AbstractVector{<:Number},
)
  n, m = length(rd), length(rp)  # var, con
  S, X = diagm(s.s), diagm(s.x)
  Sg = diagm(s.s ./ s.x)
  Z = zeros(m, n)
  KKT = [
    Sg s.A'
    s.A 0I
  ]
  rdc = rd + rc ./ s.x
  r = -[rdc; rp]
  dxy = pinv(KKT) * r
  @debug "[kkt error]" (KKT * dxy - r)
  dx, dy = dxy[1:n], dxy[n+1:n+m]
  ds = -(rc + dx .* s.s) ./ s.x
  return dx, dy, ds
end

function single_step(s::Solver)
  rd, rp, rc = kkt_residual(s)
  dxf, dyf, dsf = big_kkt_solve(s, rd, rp, rc)

  n = length(s.x)
  ap = minimum([-s.x[i] / dxf[i] for i = 1:n if dxf[i] < 0]; init = 1)
  ad = minimum([-s.s[i] / dsf[i] for i = 1:n if dsf[i] < 0]; init = 1)
  muf = mean((s.x + ap * dxf) .* (s.s + ad * dsf))
  mu = mean(s.x .* s.s)
  sigma = (muf / mu)^3
  rc += (dxf .* dsf) .- (sigma * mu)

  dx, dy, ds = big_kkt_solve(s, rd, rp, rc)
  ap = minimum([-s.x[i] / dx[i] for i = 1:n if dx[i] < 0]; init = 1)
  ad = minimum([-s.s[i] / ds[i] for i = 1:n if ds[i] < 0]; init = 1)
  ap, ad = 0.9 * ap, 0.9 * ad
  @debug "[iter]" ap ad dx dy ds

  s = @set s.x += ap * dx
  s = @set s.y += ad * dy
  s = @set s.s += ad * ds
  return s
end

function from_netw(netw::McfpNet, start::Union{Solver,Nothing} = nothing)
  A = netw.G.IncidenceMatrix
  b = netw.Demand
  u = netw.Cap
  c = netw.Cost

  n, m = netw.G.m * 2, netw.G.n + netw.G.m  # var, con
  x, y, s =
    start == nothing ? (0.1 * ones(n), zeros(m), 0.1 * ones(n)) :
    (start.x, start.y, start.s)

  S = Solver(A = [A 0A; I I], b = [b; u], c = [c; 0c], x = x, y = y, s = s)

  return S
end

end  # module MehrotraPredictorCorrector
