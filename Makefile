build:
	mpic++ convex_hull.cpp -o main

run:
	mpirun --use-hwthread-cpus -n 4  ./main 1000_1.txt
	
clean:
	rm main