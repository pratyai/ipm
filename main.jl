using ArgParse
using LinearAlgebra
using Logging

include("dimacs.jl")
include("graph.jl")
include("ipm.jl")
include("augmentations.jl")
include("long_step_path_following.jl")

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

function do_netw(netw::McfpNet)
  TOL = 1e-1

  s = from_netw(netw)
  @debug "[init]" s

  @debug "[iter start]" s.x s.y s.s
  optimal, niters = false, 0
  for t = 1:20
    @debug "[iter]" t s.x s.y s.s
    mu = mean(s.x .* s.s)
    rd, rp, rc = kkt_residual(s)
    @debug "[iter]" mu rd rp rc

    if mu < TOL && norm(rd, Inf) < TOL && norm(rp, Inf) < TOL
      optimal = true
      break
    end

    s = single_step(s)
    niters += 1
  end
  @debug "[iter end]" s.x s.y s.s optimal niters
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
  @debug netw.G netw.Cost netw.Cap netw.Demand
  do_netw(netw)


  if logio != nothing
    close(logio)
  end
end

LinearAlgebra.BLAS.set_num_threads(2)
main()
