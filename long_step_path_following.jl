module LongStepPathFollowing

using FromFile

@from "graph.jl" import Graphs.McfpNet
@from "nocedal_kkt.jl" import NocedalKKT as kkt
using LinearAlgebra
using Statistics
import Base.@kwdef
using Setfield
using SparseArrays

@kwdef struct Solver
  # tuning parameters
  gamma::Number = 1e-3
  sigma_min::Number = 0.5
  sigma_max::Number = 0.95
  mu_tol::Number = 1e-6

  kkt::kkt.Solver
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
  S::Solver,
  dx::AbstractVector{<:Number},
  dy::AbstractVector{<:Number},
  ds::AbstractVector{<:Number},
)
  s, gamma = S.kkt, S.gamma
  n, m = length(dx), length(dy)  # var, con
  EPS = 1e-6
  lo, hi = 0.0, 1.0
  mu = mean(s.x .* s.s)
  while hi - lo > EPS
    a = (lo + hi) / 2
    nx, ny, ns = s.x + a * dx, s.y + a * dy, s.s + a * ds
    if is_good_neighbourhood(nx, ny, ns, gamma)
      lo = a
    else
      hi = a
    end
  end
  return lo
end

function single_step(S::Solver; solver_fn = kkt.approx_kkt_solve)
  s = S.kkt
  # select some sigma in (0,1)
  gamma, sigma = S.gamma, S.sigma_max
  mu = mean(s.x .* s.s)

  rd, rp, rc = kkt.kkt_residual(s)
  rc .-= sigma * mu

  dx, dy, ds = solver_fn(s, rd, rp, rc)
  a = find_good_neighbourhood(S, dx, dy, ds)
  @debug "[iter]" a dx dy ds
  s = @set s.x += a * dx
  s = @set s.y += a * dy
  s = @set s.s += a * ds
  @assert is_good_neighbourhood(s.x, s.y, s.s, gamma)
  S = @set S.kkt = s

  return S
end

function from_netw(netw::McfpNet, start::Union{Solver,Nothing} = nothing)
  A = netw.G.IncidenceMatrix
  b = netw.Demand
  u = netw.Cap
  c = netw.Cost

  n, m = netw.G.m * 2, netw.G.n + netw.G.m  # var, con
  x, y, s =
    start == nothing ? (0.1 * ones(n), zeros(m), 0.1 * ones(n)) :
    (start.kkt.x, start.kkt.y, start.kkt.s)

  S = Solver(
    kkt = kkt.Solver(
      A = sparse([A 0A; I I]),
      b = [b; u],
      c = [c; 0c],
      x = x,
      y = y,
      s = s,
      Ag = sparse(A),
    ),
  )
  @assert is_good_neighbourhood(x, y, s, S.gamma)

  return S
end

function set_flow(S::Solver, netw::McfpNet, x::AbstractVector)
  s = S.kkt
  @assert length(x) == netw.G.m
  @assert length(s.x) == 2 * netw.G.m
  xu = netw.Cap - x
  EPS = 1e-3
  for i = 1:netw.G.m
    if x[i] <= EPS
      x[i], xu[i] = 0.1, netw.Cap[i] - 0.1
    elseif xu[i] <= EPS
      xu[i], x[i] = 0.1, netw.Cap[i] - 0.1
    end
  end
  s = @set s.x = [x; xu]
  S = @set S.kkt = s
  return S
end

function expected_iteration_count(S::Solver)
  s = S.kkt
  n = length(s.x)
  gamma, sigma_min, sigma_max, mu_tol = S.gamma, S.sigma_min, S.sigma_max, S.mu_tol
  delta =
    (2^1.5 * gamma * (1 - gamma) / (1 + gamma)) *
    min(sigma_min * (1 - sigma_min), sigma_max * (1 - sigma_max))
  mu = mean(s.x .* s.s)
  log_epsilon = abs(log(mu_tol / mu))
  niters = Int.(ceil(n * log_epsilon / delta))

  if niters <= 1 || niters > 100
    @debug "[lpf]" n delta mu log_epsilon niters
  end

  return niters
end

end  # module LongStepPathFollowing
