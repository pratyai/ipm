set -x

for f in $(ls logs | sort -t _ -k 2 -g); do
  julia main.jl -l '?' -i "data/${f%.log}"
done