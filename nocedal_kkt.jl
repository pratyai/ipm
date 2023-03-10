module NocedalKKT

using LinearAlgebra
using Statistics
import Base.@kwdef
using Setfield
using Laplacians
using SparseArrays

@kwdef struct Solver
  # fixed parameters for the problem
  A::AbstractMatrix{<:Number}
  b::AbstractVector{<:Number}
  c::AbstractVector{<:Number}

  # iterates
  x::AbstractVector{<:Number}
  y::AbstractVector{<:Number}  # \lambda
  s::AbstractVector{<:Number}

  # cache & perf stuff
  Ag::AbstractMatrix{<:Number}  # incidence matrix
end

function kkt_residual(s::Solver)
  rd = s.A' * s.y + s.s - s.c  # dual
  rp = s.A * s.x - s.b  # primal
  rc = s.x .* s.s  # center
  return rd, rp, rc
end

function big_kkt_solve(
  s::Solver,
  rd::AbstractVector{<:Number},
  rp::AbstractVector{<:Number},
  rc::AbstractVector{<:Number},
)
  n, m = length(rd), length(rp)  # var, con
  S, X = diagm(s.s), diagm(s.x)
  Z = zeros(m, n)
  KKT = [
    0I s.A' I
    s.A 0I Z
    S Z' X
  ]
  r = -[rd; rp; rc]
  dxys = pinv(KKT) * r
  dx, dy, ds = dxys[1:n], dxys[n+1:n+m], dxys[n+m+1:n+m+n]

  @debug begin
    KKT = [
      0I s.A' I
      s.A 0I Z
      S Z' X
    ]
    "[kkt error]"
  end (KKT * [dx; dy; ds] + [rd; rp; rc])
  return dx, dy, ds
end

function small_kkt_solve(
  s::Solver,
  rd::AbstractVector{<:Number},
  rp::AbstractVector{<:Number},
  rc::AbstractVector{<:Number},
)
  n, m = length(rd), length(rp)  # var, con
  D = diagm(s.s ./ s.x)
  KKT = [
    -D s.A'
    s.A 0I
  ]
  rdc = rd - rc ./ s.x
  r = -[rdc; rp]
  dxy = pinv(KKT) * r
  dx, dy = dxy[1:n], dxy[n+1:n+m]
  ds = -(rc + s.s .* dx) ./ s.x

  @debug begin
    S, X = diagm(s.s), diagm(s.x)
    Z = zeros(m, n)
    KKT = [
      0I s.A' I
      s.A 0I Z
      S Z' X
    ]
    "[kkt error]"
  end (KKT * [dx; dy; ds] + [rd; rp; rc])
  return dx, dy, ds
end

function no_kkt_solve(
  s::Solver,
  rd::AbstractVector{<:Number},
  rp::AbstractVector{<:Number},
  rc::AbstractVector{<:Number},
)
  M = s.A * diagm(s.x ./ s.s) * s.A'
  dy = pinv(M) * (-rp - s.A * (rd .* s.x ./ s.s) + s.A * (rc ./ s.s))
  dx = -(rc - s.x .* (rd + s.A' * dy)) ./ s.s
  ds = -(rc + s.s .* dx) ./ s.x

  @debug begin
    n, m = length(rd), length(rp)  # var, con
    S, X = diagm(s.s), diagm(s.x)
    Z = zeros(m, n)
    KKT = [
      0I s.A' I
      s.A 0I Z
      S Z' X
    ]
    "[kkt error]"
  end (KKT * [dx; dy; ds] + [rd; rp; rc])
  return dx, dy, ds
end

function approx_kkt_solve(
  s::Solver,
  rd::AbstractVector{<:Number},
  rp::AbstractVector{<:Number},
  rc::AbstractVector{<:Number},
  tol::Number = 1e-6,
)
  dy = let s = s, rd = rd, rp = rp, rc = rc
    A = s.Ag
    n, m = size(A)
    x, xu = s.x[1:m], s.x[m+1:m+m]
    rd, rdu = rd[1:m], rd[m+1:m+m]
    y, yu = s.y[1:n], s.y[n+1:n+m]
    rp, rpu = rp[1:n], rp[n+1:n+m]
    s, su = s.s[1:m], s.s[m+1:m+m]
    rc, rcu = rc[1:m], rc[m+1:m+m]

    k = x ./ s + xu ./ su
    rq = -rpu - (x .* rd - rc) ./ s - (xu .* rdu - rcu) ./ su
    d = x .* (1 .- x ./ (k .* s)) ./ s
    rs = -rp - A * ((x .* (rd + rq ./ k) - rc) ./ s)

    L = A * spdiagm(d) * A'
    ainv = function (lap)
      adj = sparse(spdiagm(diag(lap)) - lap)
      return approxchol_lap(adj; tol = tol, params = ApproxCholParams())
    end
    Li = ainv(L)
    # Li = r -> pinv(L) * r

    dy = Li(rs)
    dyu = (rq - (A' * dy) .* x ./ s) ./ k

    [dy; dyu]
  end
  dx = -(rc - s.x .* (rd + s.A' * dy)) ./ s.s
  ds = -(rc + s.s .* dx) ./ s.x

  @debug begin
    n, m = length(rd), length(rp)  # var, con
    S, X = diagm(s.s), diagm(s.x)
    Z = zeros(m, n)
    KKT = [
      0I s.A' I
      s.A 0I Z
      S Z' X
    ]
    "[kkt error]"
  end (KKT * [dx; dy; ds] + [rd; rp; rc])

  return dx, dy, ds
end

end  # module NocedalKKT
