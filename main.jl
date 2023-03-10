using FromFile

using ArgParse
using LinearAlgebra
using Logging
using Setfield
using Printf
using Statistics

@from "graph.jl" import Graphs.McfpNet
@from "dimacs.jl" import Dimacs.ReadDimacs
@from "long_step_path_following.jl" import LongStepPathFollowing as lpf
@from "mehrotra_predictor_corrector.jl" import MehrotraPredictorCorrector as mpc

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

function do_netw(netw::McfpNet; start = nothing, flow = nothing)
  TOL = 1e-1

  alg = mpc

  s = alg.from_netw(netw, start)
  if flow != nothing
    s = alg.set_flow(s, netw, flow)
  end
  if alg == lpf
    s = @set s.mu_tol = TOL
  end
  @debug "[init]" s

  optimal, niters = false, 0
  for t = 1:100
    expected_niters = alg == lpf ? lpf.expected_iteration_count(s) : nothing
    @debug "[iter]" t s.x s.y s.s expected_niters
    mu = mean(s.x .* s.s)
    rd, rp, rc = alg.kkt_residual(s)
    @debug "[iter]" mu rd rp rc norm(rd, Inf) norm(rp, Inf)

    if mu < TOL && norm(rd, Inf) < TOL && norm(rp, Inf) < TOL
      optimal = true
      break
    end

    s = alg.single_step(s)
    niters += 1

    del_mu = mu - mean(s.x .* s.s)
    @debug "[iter]" del_mu
  end
  @debug "[iter end]" s.x s.y s.s optimal niters
  if !optimal
    mu = mean(s.x .* s.s)
    rd, rp, rc = alg.kkt_residual(s)
    @debug "[iter]" mu rd rp rc
  end
  return s
end

function do_phase_1(netw::McfpNet, flow = nothing)
  @debug "[phase 1]"
  netw = @set netw.Cost = zeros(netw.G.m)
  return do_netw(netw; flow = flow)
end

function do_phase_2(netw::McfpNet, p1s = nothing)
  @debug "[phase 2]"
  return do_netw(netw; start = p1s)
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
  @debug "[mcfp]" netw.G netw.Cost netw.Cap netw.Demand

  p1s = do_phase_1(netw, x)
  p2s = do_phase_2(netw, p1s)


  if logio != nothing
    close(logio)
  end
end

LinearAlgebra.BLAS.set_num_threads(2)
main()
