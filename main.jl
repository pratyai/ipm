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
  # netw, x = augment(netw)
  x, y, z_l, z_u = starting_point(netw)
  @show netw.G netw.Cost netw.Cap netw.Demand
  @show netw.G.IncidenceMatrix * x - netw.Demand
  # x = zeros(netw.G.m)

  #=
  nx = [0.14309825144033528, 0.17616890857498807, 0.29065460208646204, 0.5662471464732046, 0.03307065713465443, 0.5331764893385502]
  @show nx
  @show netw.G.IncidenceMatrix * nx - netw.Demand
  nx = ComputeIntegralFlow(netw, nx)
  @show nx
  @show netw.G.IncidenceMatrix * nx - netw.Demand
  return
  =#

  s = Solver(
    netw = netw,
    x = x,
    y = y,
    z_l = z_l,
    z_u = z_u,
    eps = 1e-1,
    sigma = 1e-1,
    primal_eps = 1e-1,
    dual_eps = 1e-1,
    gap_eps = 1e-1,
    phase = 1,
  )

  for phase in [1, 2]
    @show phase
    s = with_phase(s, phase)
    @show s.x objective(s)
    for t = 1:100
      @show t is_optimal_enough(s)
      if is_optimal_enough(s)
        break
      end
      dx, dy, dz_l, dz_u = mehrotra_search_dir(s)
      @show dx dy dz_l dz_u
      a_primal, a_dual = decide_steplength(s, dx, dy, dz_l, dz_u)
      @show a_primal a_dual
      nx = s.x + a_primal * dx
      @show nx
      @show netw.G.IncidenceMatrix * nx - netw.Demand
      #=
      nx = ComputeIntegralFlow(netw, nx)
      @show nx
      @show netw.G.IncidenceMatrix * nx - netw.Demand
      =#
      s = with_solution(
        s,
        nx,
        s.y + a_dual * dy,
        s.z_l + a_dual * dz_l,
        s.z_u + a_dual * dz_u,
      )
      @show s.x s.y s.z_l s.z_u objective(s)
    end
  end
  @show s.x objective(s)
end

LinearAlgebra.BLAS.set_num_threads(2)
main()
