include("graph.jl")

using Scanf
using Printf
using SparseArrays

function ReadDimacs(path::String)::McfpNet
  n, m, E = nothing, nothing, nothing
  C, U, B = nothing, nothing, nothing

  f = open(path)
  while !eof(f)
    r, c = @scanf(f, "%s", String)
    if c == "c"
      c = readline(f)
    elseif c == "p"
      r, dir, n, m = @scanf(f, "%s %d %d", String, Int, Int)
      E, C, U = Int[], Int[], Int[]
      B = zeros(Int, n)
    elseif c == "n"
      r, v, b = @scanf(f, "%d %d", Int, Int)
      # we adopted the opposite convention :(
      B[v] = -b
    elseif c == "a"
      r, i, j, l, u, c = @scanf(f, "%d %d %d %d %d", Int, Int, Int, Int, Int)
      E = isempty(E) ? [i j] : [E; [i j]]
      append!(U, u)
      append!(C, c)
    end
  end
  return McfpNet(FromEdgeList(n, E), C, U, B)
end

function WriteDimacs(path::String, G::McfpNet, flow::AbstractVector{Int})
  open(path, "w") do f
    @printf(f, "p min %d %d\n", G.G.n, G.G.m)
    for i = 1:G.G.n
      @printf(f, "n %d %d\n", i, -G.Demand[i])
    end
    for i = 1:G.G.m
      u, v = G.G.EdgeList[i, :]
      @printf(f, "a %d %d 0 %d %d\n", u, v, G.Cap[i], G.Cost[i])
    end
    for i = 1:G.G.m
      @printf(f, "f %d %d\n", i, flow[i])
    end
  end
end
