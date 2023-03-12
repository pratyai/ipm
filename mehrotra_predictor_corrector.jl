module MehrotraPredictorCorrector

using FromFile

@from "graph.jl" import Graphs.McfpNet
@from "nocedal_kkt.jl" import NocedalKKT as kkt
using LinearAlgebra
using Statistics
import Base.@kwdef
using Setfield
using SparseArrays

@kwdef struct Solver
  kkt::kkt.Solver
end

function single_step(S::Solver; solver_fn = kkt.approx_kkt_solve)
  s = S.kkt
  rd, rp, rc = kkt.kkt_residual(s)
  dxf, dyf, dsf = solver_fn(s, rd, rp, rc)

  n = length(s.x)
  ap = minimum([-s.x[i] / dxf[i] for i = 1:n if dxf[i] < 0]; init = 1)
  ad = minimum([-s.s[i] / dsf[i] for i = 1:n if dsf[i] < 0]; init = 1)
  muf = mean((s.x + ap * dxf) .* (s.s + ad * dsf))
  mu = mean(s.x .* s.s)
  sigma = (muf / mu)^3
  rc += (dxf .* dsf) .- (sigma * mu)

  dx, dy, ds = solver_fn(s, rd, rp, rc)
  ap = minimum([-s.x[i] / dx[i] for i = 1:n if dx[i] < 0]; init = 1)
  ad = minimum([-s.s[i] / ds[i] for i = 1:n if ds[i] < 0]; init = 1)
  ap, ad = 0.95 * ap, 0.95 * ad
  @debug "[iter]" ap ad dx dy ds

  s = @set s.x += ap * dx
  s = @set s.y += ad * dy
  s = @set s.s += ad * ds
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

end  # module MehrotraPredictorCorrector
