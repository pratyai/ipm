using ArgParse
using Serialization
using ProfileView

function parse_cmdargs()
  s = ArgParseSettings()
  @add_arg_table s begin
    "-i"
    help = "input profile file path (.bean)"
    arg_type = String
    required = true
  end
  return parse_args(s)
end

function main()
  args = parse_cmdargs()

  f = open(args["i"])
  r = deserialize(f)
  ProfileView.view(r[1], lidict = r[2])
  readline()
end

main()
