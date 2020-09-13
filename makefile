build:
	mpicc -c ShearSort.c -lm
	mpicc -o exec ShearSort.o -lm

clean:
	rm *.o exec

run25cmd:
	mpiexec -np 25 exec input.txt

run25stdin:
	mpiexec -np 25 exec
