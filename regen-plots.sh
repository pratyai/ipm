set -x

for f in $(ls logs | sort -t _ -k 2 -g); do
  julia logviz.jl -o '?' -i "logs/${f}"
done