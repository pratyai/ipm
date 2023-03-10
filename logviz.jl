module LogViz

import Base.@kwdef
using Setfield
using DataStructures
using Plots

const EVENT_BASIC_RE = r"^┌[^└]*└[^\n]*$"ms
const EVENT_CLASS_RE = r"^┌ (?<level>[^:]+): (?:\[(?<tag>[a-zA-Z0-9 ]*)\])?.*$"ms
const EVENT_PAYLOAD_RE = r"^│[ ]*(?<key>.*) = (?<val>.*)$"m

@kwdef struct Event
  level::String
  tag::String
  payload::Any
end

function from_str(e::AbstractString)::Event
  level, tag = match(EVENT_CLASS_RE, e)
  payload = [(m["key"], m["val"]) for m in eachmatch(EVENT_PAYLOAD_RE, e)]
  return Event(level = level, tag = tag, payload = payload)
end

function ReadLogEvents(path::String)
  content = read(path, String)

  events = [from_str(m.match) for m in eachmatch(EVENT_BASIC_RE, content)]
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
      legend = :outerbottom,
      legend_columns=3
    )
    p = scatter!(t[ind], var[ind], yscale = :log10, primary = false)
    return p
  end

  p = viz(mu, "mu")
  p = viz(rd, "rd")
  p = viz(rp, "rp")

  return p
end

end  # module LogViz

using Plots
using ArgParse

function parse_cmdargs()
  s = ArgParseSettings()
  @add_arg_table s begin
    "-i"
    help = "input log file path"
    arg_type = String
    required = true
  end
  return parse_args(s)
end

function main()
  args = parse_cmdargs()

  events = LogViz.ReadLogEvents(args["i"])
  events = LogViz.NeatlyGroup(events)
  p = LogViz.PlotMu(events, "phase 1")
  p = LogViz.PlotMu(events, "phase 2")
  xlabel!("t")
  display(p)
end

main()
readline()
