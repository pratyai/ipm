set -x

for f in $(ls logs); do
  julia main.jl -l '?' -i "data/${f%.log}"
done