using ArgParse
using LinearAlgebra
using Logging

include("dimacs.jl")
include("graph.jl")
include("ipm.jl")
include("augmentations.jl")

function parse_cmdargs()
  s = ArgParseSettings()
  @add_arg_table s begin
    "-i"
    help = "input dimacs file path"
    arg_type = String
    required = true
    "-l"
    help = "log file path"
    arg_type = String
    required = false
  end
  return parse_args(s)
end

function setup_logging(logpath::String)
  @printf "logging into: %s\n" logpath
  io = open(logpath, "w")
  global_logger(SimpleLogger(io, Logging.Debug))
  return io
end

function main()
  args = parse_cmdargs()

  logio = nothing
  if args["l"] != nothing
    if args["l"] == "?"
      args["l"] = "logs/" * basename(args["i"]) * ".log"
    end
    logio = setup_logging(args["l"])
  end

  @info "[cmdline args]" args
  netw = ReadDimacs(args["i"])
  netw, x = add_a_star_spanning_tree(netw, sum(netw.Cost))
  _, y, z_l, z_u = starting_point(netw)
  @debug netw.G netw.Cost netw.Cap netw.Demand
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
  )

  s_1 = with_cost(s, zeros(Int, s.netw.G.m))
  s_1, optimal, iters = minimize(s_1, 20)
  @info "[phase 1]" optimal iters
  s_2 = with_solution(s, s_1.x, s_1.y, s_1.z_l, s_1.z_u)
  s_2, optimal, iters = minimize(s_2, 50)
  @info "[phase 2]" optimal iters

  if logio != nothing
    close(logio)
  end
end

LinearAlgebra.BLAS.set_num_threads(2)
main()
