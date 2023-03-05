using ArgParse

include("dimacs.jl")

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

function main()
  args = parse_cmdargs()
  netw = ReadDimacs(args["i"])
  @show netw
  x = [0.5, 0.5, 0.5 + 1e-12]
  @show netw.Cost' * x
  x = ComputeIntegralFlow(netw, x)
  @show x
  @show netw.Cost' * x
end

main()
