import cProfile, pstats, io
from simulation import run_simulation

pr = cProfile.Profile()
pr.enable()

run_simulation(num_days=2, seed=42, progress=False)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())
