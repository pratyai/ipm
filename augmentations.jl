using FromFile

@from "graph.jl" import Graphs

function add_a_star_spanning_tree(netw::McfpNet, cost::Int)
  #=
  Add a new node, with 0 demand (called star).
  Connect all old nodes with non-zero demand to the star.
  Each star-arc either goes in or out of the star depending on the other node's demand.
  =#
  G = netw.G
  star = G.n + 1
  starE = mapreduce(
    permutedims,
    vcat,
    [
      netw.Demand[i] < 0 ? [i, star, cost, -netw.Demand[i]] :
      [star, i, cost, netw.Demand[i]] for i = 1:G.n if netw.Demand[i] != 0
    ],
  )
  E = starE[:, 1:2]
  E = vcat(G.EdgeList, E)
  C = starE[:, 3]
  C = vcat(netw.Cost, C)
  U = starE[:, 4]
  U = vcat(netw.Cap, U)
  B = vcat(netw.Demand, 0)
  starnet = McfpNet(FromEdgeList(G.n + 1, E), C, U, B)

  # propose a flow: no flow anywhere else, but star-arcs get full flow.
  starU = starE[:, 4]
  x = vcat(zeros(G.m), starU)

  return starnet, x
end
