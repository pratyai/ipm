nice:
	julia -e 'using JuliaFormatter; format(pwd(), indent=2);'

test:
	MKL_NUM_THREADS=16 JULIA_NUM_THREADS=16 julia main.jl -i data/ng_4_6_1_1_41.dimacs
