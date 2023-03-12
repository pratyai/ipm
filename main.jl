using FromFile

using ArgParse
using LinearAlgebra
using Logging
using Setfield
using Printf
using Statistics

using Profile
using FileIO
using PProf
using Serialization

@from "graph.jl" import Graphs.McfpNet
@from "dimacs.jl" import Dimacs.ReadDimacs
@from "long_step_path_following.jl" import LongStepPathFollowing as lpf
@from "mehrotra_predictor_corrector.jl" import MehrotraPredictorCorrector as mpc
@from "nocedal_kkt.jl" import NocedalKKT as kkt

include("augmentations.jl")
include("integral_flow.jl")

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
  global_logger(SimpleLogger(io, Logging.Info))
  return io
end

function do_netw(netw::McfpNet; start = nothing, flow = nothing)
  TOL = 1e-2

  alg = mpc

  S = alg.from_netw(netw, start)
  if flow != nothing
    S = alg.set_flow(S, netw, flow)
  end
  if alg == lpf
    S = @set S.mu_tol = TOL
  end
  @debug "[init]" S

  optimal, niters = false, 0
  for t = 1:100
    s = S.kkt
    expected_niters = alg == lpf ? lpf.expected_iteration_count(S) : nothing
    @info "[iter]" t
    @debug "[iter]" s.x s.y s.s expected_niters
    mu = mean(s.x .* s.s)
    rd, rp, rc = kkt.kkt_residual(s)
    @info "[iter]" mu norm(rd, Inf) norm(rp, Inf)
    @debug "[iter]" rd rp rc

    if mu < TOL && norm(rd, Inf) < TOL && norm(rp, Inf) < TOL
      optimal = true
      break
    end

    S = alg.single_step(S; solver_fn = kkt.approx_kkt_solve)
    niters += 1

    s = S.kkt
    del_mu = mu - mean(s.x .* s.s)
    @info "[iter]" del_mu
  end

  s = S.kkt
  @info "[iter end]" optimal niters
  @debug "[iter end]" s.x s.y s.s optimal niters
  if !optimal
    mu = mean(s.x .* s.s)
    rd, rp, rc = kkt.kkt_residual(s)
    @debug "[iter]" mu rd rp rc
  end
  return S
end

function do_phase_1(netw::McfpNet, flow = nothing)
  @info "[phase 1]"
  netw = @set netw.Cost = zeros(netw.G.m)
  return do_netw(netw; flow = flow)
end

function do_phase_2(netw::McfpNet, p1s = nothing)
  @info "[phase 2]"
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
  onetw = netw
  @info "[mcfp]" onetw.G onetw.Cost onetw.Cap onetw.Demand

  netw, x = add_a_star_spanning_tree(netw, sum(netw.Cost))
  @info "[mcfp]" netw.G netw.G.n netw.G.m
  @info "[mcfp]" netw.Cost netw.Cap netw.Demand

  p1s = do_phase_1(netw, x)
  p2s = @profile do_phase_2(netw, p1s)
  open("profiles/" * basename(args["i"]) * ".bean", "w") do f
    serialize(f, Profile.retrieve())
  end

  x = p2s.kkt.x[1:netw.G.m]
  @info "[flow]" x

  x = compute_integral_flow(netw, x)
  @info "[integral flow]" x

  ox = x[1:onetw.G.m]
  @info "[integral flow]" ox


  if logio != nothing
    close(logio)
  end
end

LinearAlgebra.BLAS.set_num_threads(2)
main()
