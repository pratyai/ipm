using FromFile

@from "graph.jl" import Graphs

using LinearAlgebra
import Base.@kwdef
using Setfield

@kwdef struct LongStepPathFollowingData
  gamma::Number
  sigma_min::Number
  sigma_max::Number
end

@kwdef struct Solver
  netw::McfpNet

  x::AbstractVector{<:Number}
  y::AbstractVector{<:Number}
  z_l::AbstractVector{<:Number}
  z_u::AbstractVector{<:Number}

  eps::Number
  sigma::Number

  primal_eps::Number
  dual_eps::Number
  gap_eps::Number

  long_step_path_following_data::Union{LongStepPathFollowingData,Nothing} = nothing
end

function with_solution(
  s::Solver,
  x::AbstractVector{<:Number},
  y::AbstractVector{<:Number},
  z_l::AbstractVector{<:Number},
  z_u::AbstractVector{<:Number},
)
  s = @set s.x = x
  s = @set s.y = y
  s = @set s.z_l = z_l
  s = @set s.z_u = z_u
  return s
end

function with_cost(s::Solver, cost::AbstractVector{Int})
  s = @set s.netw.Cost = cost
  return s
end

function objective(s::Solver)
  return s.netw.Cost' * s.x
end

# ref: https://link.springer.com/book/10.1007/978-0-387-40065-5 (ch. 14.2)
function starting_point(netw::McfpNet)
  G = netw.G
  A, b, c = G.IncidenceMatrix, netw.Demand, netw.Cost
  L = A * A'

  x = A' * pinv(L) * b
  dx = max(0, -1.5 * minimum(x))
  x = x .+ dx
  y = pinv(L) * A * c
  z = c - A'y
  dz = max(0, -1.5 * minimum(z))
  z = z .+ dz
  dx = 0.5 * (x' * z) / sum(z)
  dz = 0.5 * (x' * z) / sum(x)
  x = x .+ dx
  z = z .+ dz
  z_u = z * 0.5
  z_l = z * 1.5

  return x, y, z_l, z_u
end

function kkt_residuals(s::Solver)
  netw, G = s.netw, s.netw.G
  A, b, c = G.IncidenceMatrix, netw.Demand, netw.Cost
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps = s.eps

  rd = c - z_l + z_u + A' * y
  rp = A * x - b
  rc_l = (x .+ eps) .* z_l
  rc_u = (netw.Cap .+ eps - x) .* z_u
  return rd, rp, rc_l, rc_u
end

function kkt_solve(
  s::Solver,
  rd::AbstractVector{<:Number},
  rp::AbstractVector{<:Number},
  rc_l::AbstractVector{<:Number},
  rc_u::AbstractVector{<:Number},
)
  netw, G = s.netw, s.netw.G
  A = G.IncidenceMatrix
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps = s.eps

  S_l = diagm(z_l ./ (x .+ eps))
  S_u = diagm(z_u ./ (netw.Cap .+ eps - x))
  S = S_l + S_u
  KKT = [
    S A'
    A 0I
  ]

  rdc = rd + rc_l ./ (x .+ eps) - rc_u ./ (netw.Cap .+ eps - x)
  r = -[rdc; rp]

  dxy = pinv(KKT) * r
  @debug "[kkt solver]" KKT * dxy - r
  dx, dy = dxy[1:G.m], dxy[G.m+1:G.m+G.n]

  dz_l = -(rc_l + dx .* z_l) ./ (x .+ eps)
  dz_u = (-rc_u + dx .* z_u) ./ (netw.Cap .+ eps - x)

  return dx, dy, dz_l, dz_u
end

function surrogate_duality_gap(s::Solver)
  netw, eps = s.netw, s.eps
  x, z_l, z_u = s.x, s.z_l, s.z_u
  gap = -[(-x .- eps); (x - netw.Cap .- eps)]' * [z_l; z_u]
  return gap
end

# ref: https://en.wikipedia.org/wiki/Mehrotra_predictor%E2%80%93corrector_method
function mehrotra_search_dir(s::Solver)
  netw, G = s.netw, s.netw.G
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps, sigma = s.eps, s.sigma

  gap = surrogate_duality_gap(s)
  mu = gap / G.m

  # predictor step
  rd, rp, rc_l, rc_u = kkt_residuals(s)
  dx_aff, dy_aff, dz_laff, dz_uaff = kkt_solve(s, rd, rp, rc_l, rc_u)

  # centering + corrector step
  @debug "[mehrotra]" mu sigma
  rd, rp, rc_l, rc_u = kkt_residuals(s)
  rc_l, rc_u =
    rc_l .- (sigma * mu) + (dx_aff .* dz_laff), rc_u .- (sigma * mu) + (dx_aff .* dz_uaff)
  dx, dy, dz_l, dz_u = kkt_solve(s, rd, rp, rc_l, rc_u)

  return dx, dy, dz_l, dz_u
end

# ref: https://link.springer.com/book/10.1007/978-0-387-40065-5 (alg. 14.2)
function long_step_path_following_search_dir(s::Solver)
  data = s.long_step_path_following_data
  sigma = (data.sigma_min + data.sigma_max) / 2
  gap = surrogate_duality_gap(s)
  mu = gap / s.netw.G.m
  @debug "[path following]" mu sigma

  rd, rp, rc_l, rc_u = kkt_residuals(s)
  rc_l, rc_u = rc_l .- sigma * mu, rc_u .- sigma * mu
  dx, dy, dz_l, dz_u = kkt_solve(s, rd, rp, rc_l, rc_u)

  return dx, dy, dz_l, dz_u
end

# ref: https://link.springer.com/book/10.1007/978-0-387-40065-5 (alg. 14.2)
function long_step_path_following_steplength(
  s::Solver,
  dx::AbstractVector{<:Number},
  dy::AbstractVector{<:Number},
  dz_l::AbstractVector{<:Number},
  dz_u::AbstractVector{<:Number},
)
  netw, eps, data = s.netw, s.eps, s.long_step_path_following_data
  gamma = data.gamma
  gap = surrogate_duality_gap(s)
  mu = gap / s.netw.G.m

  a = 1
  for i = 1:20
    nx, nz_l, nz_u = s.x + a * dx, s.z_l + a * dz_l, s.z_u + a * dz_u
    xz_l = nz_l .* (nx .+ eps)
    xz_u = nz_u .* (netw.Cap .+ eps - nx)
    in_good_neighbourhood = all(vcat(xz_l .>= gamma * mu, xz_u .>= gamma * mu))
    if in_good_neighbourhood
      return a
    end
    a = a * 0.9
  end
  return 0
end

# ref: https://link.springer.com/book/10.1007/978-0-387-40065-5 (alg. 14.2)
function long_step_path_following_iteration(s::Solver)
  netw, eps, data = s.netw, s.eps, s.long_step_path_following_data
  gamma = data.gamma
  gap = surrogate_duality_gap(s)
  mu = gap / s.netw.G.m
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u

  xz_l = z_l .* (x .+ eps)
  xz_u = z_u .* (netw.Cap .+ eps - x)
  in_good_neighbourhood = all(vcat(xz_l .>= gamma * mu, xz_u .>= gamma * mu))
  @debug "[iter]" in_good_neighbourhood

  dx, dy, dz_l, dz_u = long_step_path_following_search_dir(s)
  a = long_step_path_following_steplength(s, dx, dy, dz_l, dz_u)
  @debug "[iter]" dx dy dz_l dz_u a
  return x + a * dx, y + a * dy, z_l + a * dz_l, z_u + a * dz_u
end

# ref: https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
function path_following_search_dir(s::Solver)
  netw, G = s.netw, s.netw.G
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps, sigma = s.eps, s.sigma

  gap = surrogate_duality_gap(s)
  mu = sigma * gap / G.m
  @debug "[path following]" mu

  rd, rp, rc_l, rc_u = kkt_residuals(s)
  rc_l, rc_u = rc_l .- mu, rc_u .- mu
  dx, dy, dz_l, dz_u = kkt_solve(s, rd, rp, rc_l, rc_u)

  return dx, dy, dz_l, dz_u
end

function steplength_inside_box(
  x::AbstractVector{<:Number},
  dx::AbstractVector{<:Number},
  cap::Union{AbstractVector{<:Number},Nothing} = nothing,
)
  n = length(x)
  return minimum(
    vcat(
      [-x[i] / dx[i] for i = 1:n if dx[i] < 0],
      (cap != nothing ? [(cap[i] - x[i]) / dx[i] for i = 1:n if dx[i] > 0] : []),
    ),
    init = 1,
  )
end

# ref: https://link.springer.com/book/10.1007/978-0-387-40065-5 (ch. 14.2)
function decide_steplength(
  s::Solver,
  dx::AbstractVector{<:Number},
  dy::AbstractVector{<:Number},
  dz_l::AbstractVector{<:Number},
  dz_u::AbstractVector{<:Number},
)
  netw, G, c = s.netw, s.netw.G, s.netw.Cost
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u

  a_primal = steplength_inside_box(x, dx, s.netw.Cap)
  a_dual = min(steplength_inside_box(z_l, dz_l), steplength_inside_box(z_u, dz_u))

  return a_primal, a_dual
end

function is_optimal_enough(s::Solver)
  netw = s.netw
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps = s.eps

  gap = surrogate_duality_gap(s)
  rd, rp, rc_l, rc_u = kkt_residuals(s)

  @debug "[optimality check]" rp rd rc_l rc_u gap

  return norm(rp, 2) <= s.primal_eps && norm(rd, 2) <= s.dual_eps && gap <= s.gap_eps
end

function minimize(s::Solver, max_iter::Int)
  # return (current solver state, is optimal, number of iterations completed)
  @info "[minimizing]" max_iter objective(s)
  for t = 1:max_iter
    optimal = is_optimal_enough(s)
    @debug "[iter]" t optimal objective(s)
    @debug "[iter]" s.x s.y s.z_l s.z_u
    if optimal
      @info "[minimizing]" optimal t objective(s)
      return s, optimal, t - 1
    end
    old_obj, old_gap = objective(s), surrogate_duality_gap(s)

    #=
    dx, dy, dz_l, dz_u = path_following_search_dir(s)
    @debug "[iter]" dx dy dz_l dz_u
    a_primal, a_dual = decide_steplength(s, dx, dy, dz_l, dz_u)
    @debug "[iter]" a_primal a_dual
    ... =  s.x + a_primal * dx, s.y + a_dual * dy, s.z_l + a_dual * dz_l, s.z_u + a_dual * dz_u
    =#

    nx, ny, nz_l, nz_u = long_step_path_following_iteration(s)
    @debug "[iter]" (s.netw.G.IncidenceMatrix * nx - s.netw.Demand)

    s = with_solution(s, nx, ny, nz_l, nz_u)

    new_obj, new_gap = objective(s), surrogate_duality_gap(s)
    del_obj, del_gap = old_obj - new_obj, old_gap - new_gap
    @debug "[iter]" del_obj del_gap
  end
  @debug "[iter]" s.x s.y s.z_l s.z_u
  optimal = false
  @info "[minimizing]" optimal max_iter objective(s)
  return s, false, max_iter
end
