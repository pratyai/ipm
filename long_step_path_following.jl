include("graph.jl")

using LinearAlgebra
using Statistics
import Base.@kwdef
using Setfield

@kwdef struct LongStepPathFollower
  # tuning parameters
  gamma::Number = 1e-3
  sigma_min::Number = 0.5
  sigma_max::Number = 0.9

  # fixed parameters for the problem
  A::AbstractMatrix{<:Number}
  b::AbstractVector{<:Number}
  c::AbstractVector{<:Number}

  # iterates
  x::AbstractVector{<:Number}
  y::AbstractVector{<:Number}  # \lambda
  s::AbstractVector{<:Number}
end

function kkt_residual(s::LongStepPathFollower)
  rd = s.A' * s.y + s.s - s.c  # dual
  rp = s.A * s.x - s.b  # primal
  rc = s.x .* s.s  # center
  return rd, rp, rc
end

function big_kkt_solve(
  s::LongStepPathFollower,
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
  dx, dy, ds = dxys[1:n], dxys[n+1:n+m], dxys[n+m+1:n+m+n]
  return dx, dy, ds
end

function search_dir(s::LongStepPathFollower, sigma::Number)
  return big_kkt_solve(s, rd, rp, rc)
end

function is_good_neighbourhood(
  x::AbstractVector{<:Number},
  y::AbstractVector{<:Number},
  s::AbstractVector{<:Number},
  gamma::Number,
)
  mu = mean(x .* s)
  return all((x .* s) .>= gamma * mu)
end

function find_good_neighbourhood(
  s::LongStepPathFollower,
  dx::AbstractVector{<:Number},
  dy::AbstractVector{<:Number},
  ds::AbstractVector{<:Number},
)
  n, m = length(dx), length(dy)  # var, con
  EPS = 1e-6
  lo, hi = 0.0, 1.0
  mu = mean(s.x .* s.s)
  while hi - lo > EPS
    a = (lo + hi) / 2
    nx, ny, ns = s.x + a * dx, s.y + a * dy, s.s + a * ds
    if is_good_neighbourhood(nx, ny, ns, s.gamma)
      lo = a
    else
      hi = a
    end
  end
  return lo
end

function single_step(s::LongStepPathFollower)
  # select some sigma in (0,1)
  sigma = s.sigma_max
  mu = mean(s.x .* s.s)

  rd, rp, rc = kkt_residual(s)
  rc .-= sigma * mu

  dx, dy, ds = big_kkt_solve(s, rd, rp, rc)
  a = find_good_neighbourhood(s, dx, dy, ds)
  @debug "[iter]" a dx dy ds
  s = @set s.x += a * dx
  s = @set s.y += a * dy
  s = @set s.s += a * ds
  return s
end

function from_netw(netw::McfpNet)
  A = netw.G.IncidenceMatrix
  b = netw.Demand
  u = netw.Cap
  c = netw.Cost

  n, m = netw.G.m * 2, netw.G.n + netw.G.m  # var, con
  x = 0.1 * ones(n)
  y = zeros(m)
  s = 0.1 * ones(n)

  S = LongStepPathFollower(A = [A 0A; I I], b = [b; u], c = [c; 0c], x = x, y = y, s = s)
  @assert is_good_neighbourhood(x, y, s, S.gamma)

  return S
end
