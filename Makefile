SRC =  colouringCC.c mmio.c converter.c
TARGET = seirial

compile:
	gcc -O3 -fopenmp colouringCC_OpenMP.c mmio.c converter.c -o parallel_MP
	mpicc -O3 colouringCC.c mmio.c converter.c -o parallel -lm
	
run: compile

	@echo "===== OPENMP ====="
	./parallel_MP ./data/com-Orkut.mtx

	@echo "===== MPI ====="
	mpirun -np 4 ./parallel ./data/com-Orkut.mtx

clean:
	rm -f parallel_MP
	rm -f parallel