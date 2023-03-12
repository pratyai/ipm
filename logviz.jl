module LogViz

import Base.@kwdef
using Setfield
using DataStructures
using Plots

gr()  # plot backend

const EVENT_CLASS_RE = r"^┌ (?<level>[^:]+): (?:\[(?<tag>[a-zA-Z0-9 ]*)\])?.*$"ms
const EVENT_PAYLOAD_RE = r"^│[ ]*(?<key>.*) = (?<val>.*)$"m

@kwdef struct Event
  level::String
  tag::String
  payload::Any
end

function from_str(e::AbstractString)::Union{Event,Nothing}
  level, tag = match(EVENT_CLASS_RE, e)
  if level != "Info"
    return nothing
  end
  try
    payload = [(m["key"], m["val"]) for m in eachmatch(EVENT_PAYLOAD_RE, e)]
    return Event(level = level, tag = tag, payload = payload)
  catch e
    @show e
    return nothing
  end
end

function ReadLogEvents(path::String)
  lines = readlines(path)
  events = String[]
  ce = nothing
  for l in lines
    if length(l) >= 1000
      # skip massive lines
      continue
    end
    if startswith(l, "┌")
      if ce != nothing
        append!(events, [ce])
      end
      ce = l
    else
      ce = ce * "\n" * l
    end
  end
  if ce != nothing
    append!(events, [ce])
  end

  events = [from_str(e) for e in events]
  events = [e for e in events if e != nothing]
  return events
end

function ByTime(events::AbstractVector{Event})
  dict = DefaultDict{Int,AbstractVector{Event}}(() -> Event[])
  t = nothing
  for e in events
    for (k, v) in e.payload
      if k == "t"
        t = parse(Int, v)
      end
    end
    if t == nothing
      continue
    end
    append!(dict[t], [e])
  end
  return dict
end

function ByPhase(events::AbstractVector{Event})
  dict = DefaultDict{AbstractString,AbstractVector{Event}}(() -> Event[])
  phase = nothing
  for e in events
    if startswith(e.tag, "phase ")
      phase = e.tag
      continue
    end
    if phase == nothing
      continue
    end
    append!(dict[phase], [e])
  end
  return dict
end

function NeatlyGroup(events::AbstractVector{Event})
  events = ByPhase(events)
  events = Dict(p => ByTime(es) for (p, es) in events)
  return events
end

function PlotMu(events, phase)
  events = events[phase]
  n = length(events)
  t = 1:n
  mu = zeros(n)
  rd = zeros(n)
  rp = zeros(n)
  for (t, es) in events
    for e in es
      if e.tag != "iter"
        continue
      end
      for (k, v) in e.payload
        if k == "mu"
          mu[t] = parse(Float64, v)
        elseif k == "norm(rd, Inf)"
          rd[t] = parse(Float64, v)
        elseif k == "norm(rp, Inf)"
          rp[t] = parse(Float64, v)
        end
      end
    end
  end

  viz = function (var, name)
    ind = var .> 0
    p = plot!(
      t[ind],
      var[ind],
      yscale = :log10,
      label = name * " (" * phase * ")",
      primary = true,
    )
    p = scatter!(t[ind], var[ind], yscale = :log10, primary = false)
    return p
  end

  p = viz(mu, "mu")
  p = viz(rd, "rd")
  p = viz(rp, "rp")

  return p
end

function PlotNiters(stat)
  sort!(stat; by = x -> x[1] + x[2])

  n_m = [n + m for (n, m, p1ic, p2ic) in stat]
  p1ic = [p1ic for (n, m, p1ic, p2ic) in stat]
  p2ic = [p2ic for (n, m, p1ic, p2ic) in stat]

  viz = function (var, name)
    p = plot!(n_m, var, xscale = :log10, label = name, primary = true)
    p = scatter!(n_m, var, xscale = :log10, primary = false)
    return p
  end

  p = viz(p1ic, "phase 1 #iters")
  p = viz(p2ic, "phase 2 #iters")
  p = viz(p1ic + p2ic, "net #iters")
  hline!(p, [100], label = "cutoff")

  return p
end

end  # module LogViz

using Plots
using ArgParse

function parse_cmdargs()
  s = ArgParseSettings()
  @add_arg_table s begin
    "-i"
    help = "input log file path (or directory, depending on the context)"
    arg_type = String
    required = true
    "-o"
    help = "image file directory"
    arg_type = String
    required = false
    "-g"
    help = "compute aggregate plots"
    action = :store_true
  end
  return parse_args(s)
end

function do_single_log(f::String)
  events = LogViz.ReadLogEvents(f)
  events = LogViz.NeatlyGroup(events)

  p = plot(dpi = 300, size = (800, 640), legend = :outerbottom, legend_columns = 3)
  p = LogViz.PlotMu(events, "phase 1")
  p = LogViz.PlotMu(events, "phase 2")
  xlabel!("t")
end

function do_aggregate_log(d::String)
  logs = readdir(d, join = true)
  @show logs

  stat_lg = function (lg)
    events = LogViz.ReadLogEvents(lg)

    n, m = nothing, nothing
    for e in events
      if e.tag != "mcfp"
        continue
      end
      for (k, v) in e.payload
        if k == "netw.G.n"
          n = parse(Int, v)
        elseif k == "netw.G.m"
          m = parse(Int, v)
        end
      end
    end

    events = LogViz.NeatlyGroup(events)

    niters = function (evs)
      for (t, es) in evs
        for e in es
          if e.tag != "iter end"
            continue
          end
          for (k, v) in e.payload
            if k == "niters"
              return parse(Int, v)
            end
          end
        end
      end
      return nothing
    end
    p1_niters, p2_niters = niters(events["phase 1"]), niters(events["phase 2"])

    return n, m, p1_niters, p2_niters
  end

  stat = [stat_lg(lg) for lg in logs]

  p = plot(dpi = 300, size = (800, 640), legend = :outerbottom)
  p = LogViz.PlotNiters(stat)
  xlabel!("n + m")
end

function main()
  args = parse_cmdargs()

  if args["o"] != nothing
    if args["o"] == "?"
      args["o"] = "plots"
    end
  end

  if args["g"]
    do_aggregate_log(args["i"])
  else
    do_single_log(args["i"])
  end

  if args["o"] != nothing
    Plots.svg(args["o"] * "/" * basename(args["i"]))
    Plots.png(args["o"] * "/" * basename(args["i"]))
  else
    display(p)
    readline()
  end
end

main()
