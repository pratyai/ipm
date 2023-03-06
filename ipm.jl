include("graph.jl")

using LinearAlgebra
import Base.@kwdef

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

  phase::Int
end

function with_solution(
  s::Solver,
  x::AbstractVector{<:Number},
  y::AbstractVector{<:Number},
  z_l::AbstractVector{<:Number},
  z_u::AbstractVector{<:Number},
)
  return Solver(
    netw = s.netw,
    x = x,
    y = y,
    z_l = z_l,
    z_u = z_u,
    eps = s.eps,
    sigma = s.sigma,
    primal_eps = s.primal_eps,
    dual_eps = s.dual_eps,
    gap_eps = s.gap_eps,
    phase = s.phase,
  )
end

function with_phase(s::Solver, phase::Int)
  return Solver(
    netw = s.netw,
    x = s.x,
    y = s.y,
    z_l = s.z_l,
    z_u = s.z_u,
    eps = s.eps,
    sigma = s.sigma,
    primal_eps = s.primal_eps,
    dual_eps = s.dual_eps,
    gap_eps = s.gap_eps,
    phase = phase,
  )
end

function objective(s::Solver)
  if s.phase == 1
    return 0
  else
    return s.netw.Cost' * s.x
  end
end

# ref: https://link.springer.com/book/10.1007/978-0-387-40065-5 (ch. 14.2)
function starting_point(netw::McfpNet)
  G = netw.G
  A, b, c = G.IncidenceMatrix, netw.Demand, netw.Cost
  L = A * A'

  x = A' * pinv(L) * b
  dx = max(0, -1.5 * minimum(x))
  x = x .+ dx  # beware of pointwise ops
  y = pinv(L) * A * c
  z = c - A'y
  dz = max(0, -1.5 * minimum(z))
  z = z .+ dz  # beware of pointwise ops
  dx = 0.5 * (x' * z) / sum(z)
  dz = 0.5 * (x' * z) / sum(x)
  x = x .+ dx  # beware of pointwise ops
  z = z .+ dz  # beware of pointwise ops
  z_u = z * 0.5
  z_l = z * 1.5

  return x, y, z_l, z_u
end

function kkt_residuals(s::Solver)
  netw, G = s.netw, s.netw.G
  A, b = G.IncidenceMatrix, netw.Demand
  c = (s.phase == 1 ? zeros(netw.G.m) : netw.Cost)
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps = s.eps

  rd = c - z_l + z_u + A' * y
  rp = A * x - b
  rc_l = (x .+ eps) .* z_l  # beware of pointwise ops
  rc_u = (netw.Cap .+ eps - x) .* z_u  # beware of pointwise ops
  return rd, rp, rc_l, rc_u
end

function box_project(s::Solver, dx::AbstractVector{<:Number})
  netw, x = s.netw, s.x
  for i = 1:netw.G.m
    if dx[i] < 0 && x[i] <= 0
      dx[i] = 0
    elseif dx[i] > 0 && x[i] >= netw.Cap[i]
      dx[i] = 0
    end
  end
  return dx
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

  S_l = diagm(z_l ./ (x .+ eps))  # beware of pointwise ops
  S_u = diagm(z_u ./ (netw.Cap .+ eps - x))  # beware of pointwise ops
  S = S_l + S_u
  KKT = [
    S A'
    A 0I
  ]

  rdc = rd + rc_l ./ (x .+ eps) + rc_u ./ (netw.Cap .+ eps - x)  # beware of pointwise ops
  r = -[rdc; rp]

  dxy = pinv(KKT) * r
  @show KKT * dxy - r
  dx, dy = dxy[1:G.m], dxy[G.m+1:G.m+G.n]

  dz_l = -(rc_l + dx .* z_l) ./ (x .+ eps)  # beware of pointwise ops
  dz_u = (-rc_u + dx .* z_u) ./ (netw.Cap .+ eps - x)  # beware of pointwise ops

  return dx, dy, dz_l, dz_u
end

# ref: https://en.wikipedia.org/wiki/Mehrotra_predictor%E2%80%93corrector_method
function mehrotra_search_dir(s::Solver)
  netw, G = s.netw, s.netw.G
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps, sigma = s.eps, s.sigma

  # surrogate duality gap
  gap = -[(-x .- eps); (x - netw.Cap .- eps)]' * [z_l; z_u]  # beware of pointwise ops
  mu = sigma * gap / G.m
  @show mu

  # predictor step
  rd, rp, rc_l, rc_u = kkt_residuals(s)
  dx_aff, dy_aff, dz_laff, dz_uaff = kkt_solve(s, rd, rp, rc_l, rc_u)

  # centering + corrector step
  rd, rp, rc_l, rc_u = kkt_residuals(s)
  rc_l, rc_u = rc_l .- mu + (dx_aff .* dz_laff), rc_u .- mu + (dx_aff .* dz_uaff)
  dx, dy, dz_l, dz_u = kkt_solve(s, rd, rp, rc_l, rc_u)

  return dx, dy, dz_l, dz_u
end

# ref: https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
function primal_dual_search_dir(s::Solver)
  netw, G = s.netw, s.netw.G
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps, sigma = s.eps, s.sigma

  # surrogate duality gap
  gap = -[(-x .- eps); (x - netw.Cap .- eps)]' * [z_l; z_u]  # beware of pointwise ops
  mu = sigma * gap / G.m
  @show mu

  rd, rp, rc_l, rc_u = kkt_residuals(s)
  rc_l, rc_u = rc_l .- mu, rc_u .- mu
  dx, dy, dz_l, dz_u = kkt_solve(s, rd, rp, rc_l, rc_u)

  return dx, dy, dz_l, dz_u
end

# ref: https://link.springer.com/book/10.1007/978-0-387-40065-5 (ch. 14.2)
function decide_steplength(
  s::Solver,
  dx::AbstractVector{<:Number},
  dy::AbstractVector{<:Number},
  dz_l::AbstractVector{<:Number},
  dz_u::AbstractVector{<:Number},
)
  netw, G = s.netw, s.netw.G
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u

  a_primal = minimum(
    vcat(
      [-x[i] / dx[i] for i = 1:G.m if dx[i] < 0],
      [(netw.Cap[i] - x[i]) / dx[i] for i = 1:G.m if dx[i] > 0],
    ),
    init = 1,
  )
  a_dual = minimum(
    vcat(
      [-z_l[i] / dz_l[i] for i = 1:G.m if dz_l[i] < 0],
      [-z_u[i] / dz_u[i] for i = 1:G.m if dz_u[i] < 0],
    ),
    init = 1,
  )

  return a_primal, a_dual
end

function is_optimal_enough(s::Solver)
  netw = s.netw
  x, y, z_l, z_u = s.x, s.y, s.z_l, s.z_u
  eps = s.eps

  # surrogate duality gap
  gap = -[(-x .- eps); (x - netw.Cap .- eps)]' * [z_l; z_u]  # beware of pointwise ops
  rd, rp, rc_l, rc_u = kkt_residuals(s)

  @show rp rd gap

  return norm(rp, 2) <= s.primal_eps && norm(rd, 2) <= s.dual_eps && gap <= s.gap_eps
end
