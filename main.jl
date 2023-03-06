using ArgParse
using LinearAlgebra

include("dimacs.jl")
include("ipm.jl")

function parse_cmdargs()
  s = ArgParseSettings()
  @add_arg_table s begin
    "-i"
    help = "input dimacs file path"
    arg_type = String
    required = true
  end
  return parse_args(s)
end

function augment(netw::McfpNet)
  G = netw.G
  n = G.n + 1
  E = mapreduce(permutedims, vcat, [netw.Demand[i] < 0 ? [i, n] : [n, i] for i = 1:G.n])
  E = vcat(G.EdgeList, E)
  C = vcat(netw.Cost, sum(netw.Cost) * ones(G.n))
  U = vcat(netw.Cap, [Int.(sum(abs.(netw.Demand)) / 2) for i = 1:G.n])
  B = vcat(netw.Demand, 0)
  x = vcat(zeros(G.m), abs.(netw.Demand))
  return McfpNet(FromEdgeList(n, E), C, U, B), x
end

function main()
  args = parse_cmdargs()
  netw = ReadDimacs(args["i"])
  netw, x = augment(netw)
  @show netw.G netw.Cost netw.Cap netw.Demand
  @show netw.G.IncidenceMatrix * x - netw.Demand
  # x = zeros(netw.G.m)

  s = Solver(
    netw = netw,
    x = x,
    y = zeros(netw.G.n),
    z_l = ones(netw.G.m),
    z_u = ones(netw.G.m),
    eps = 1e-1,
    sigma = 1e-1,
    primal_eps = 1e-1,
    dual_eps = 1e-1,
    gap_eps = 1e-1,
  )
  @show s.x objective(s)
  for t = 1:100
    @show t is_optimal_enough(s)
    dx, dy, dz_l, dz_u = mehrotra_search_dir(s)
    @show dx dy dz_l dz_u
    a_primal, a_dual = decide_steplength(s, dx, dy, dz_l, dz_u)
    @show a_primal a_dual
    if norm([a_primal * dx; a_dual * dy; a_dual * dz_l; a_dual * dz_u], Inf) < 1e-9
      break
    end
    nx = s.x + a_primal * dx
    @show nx
    @show netw.G.IncidenceMatrix * nx - netw.Demand
    # nx = ComputeIntegralFlow(netw, nx)
    s = with(s, nx, s.y + a_dual * dy, s.z_l + a_dual * dz_l, s.z_u + a_dual * dz_u)
    @show s.x s.y s.z_l s.z_u objective(s)
  end
  @show s.x objective(s)
end

LinearAlgebra.BLAS.set_num_threads(2)
main()
