build:
	mpic++ convex_hull.cpp -o main

run:
	mpirun --use-hwthread-cpus -n $(n) ./main $(i)
	
clean:
	rm main