import pynetgen as ng
import random as rng

if __name__ == '__main__':
    rng.seed(2)
    for n in [200, 300, 500]:
        try:
            seed = rng.randint(0, 100)
            arcs = rng.randint(n, n*(n-1))
            nsrc = rng.randint(1, round(n/3))
            nsnk = rng.randint(1, round((n - nsrc)/2))

            fname = f'ng_{n}_{arcs}_{nsrc}_{nsnk}_{seed}.dimacs'
            # type=0 is for generating min-cost flow problems.
            print(f'generating {fname}')
            ng.netgen_generate(seed=seed, nodes=n, sources=nsrc, sinks=nsnk,
                               density=arcs, type=0, fname=fname)
        except Exception as e:
            print(f'there was some error: {e}')