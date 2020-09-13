/*
 ============================================================================
 Name        : ShearSort.c
 Author      : Daniel Sasson
 Version     : 1.0
 Copyright   : MIT
 Description : An implementation of Shear Sort algorithm with MPI.
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define ROOT 0
#define MAX_INPUT_SIZE 1000
#define MAX_FILE_NAME_LENGTH 20
#define TRIO_SIZE 3
#define DIM 2
#define ROWS_DIM 1
#define COLS_DIM 0
#define DISP 1

// A struct for a sequence of three integers.
typedef struct
{
	int x;
	int y;
	int z;
}Trio;

MPI_Comm create_cart_comm(int rank, int n, int *coords);
int read_from_file(char* file_name, int *data);
int get_perfect_square(int n);
void create_trio_datatype(MPI_Datatype* trio_type);
void create_trio_mat(Trio *trios, int *data, int num_of_trios);
void print_trio_array(Trio *trios, int n);
int compare_trio(Trio *t1, Trio *t2);
void shear_sort(MPI_Comm comm, MPI_Datatype trio_type, Trio* my_trio, int *coords, int n);
void odd_even_sort(MPI_Comm comm, MPI_Datatype trio_type, Trio* my_trio, int *coords ,int dim, int n);
void compare_and_exchange(MPI_Comm comm, MPI_Datatype trio_type, Trio* my_trio, int *coords, int dim, int doEven);
void set_trio(Trio* trio, int x, int y, int z);

int main(int argc, char *argv[])
{
    int rank, num_of_procs;
    int data[MAX_INPUT_SIZE], num_of_trios=0, n=0;
    char* file_name;
    Trio *trios, my_trio;
    MPI_Datatype trio_type;
    MPI_Comm comm;

    int my_coord[DIM];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs);
    create_trio_datatype(&trio_type);

    if (rank == 0)
    {
    	if (argc > 1) // File name delivered as a command line argument.
    	{
    		file_name = argv[1];
    	}
	else // Read file name from stdin
	{
		file_name = (char*)calloc(MAX_FILE_NAME_LENGTH, sizeof(char));
		printf("Enter file name: ");fflush(stdout);
		scanf("%s", file_name);
	}

	printf("\nfile name: %s\n", file_name);

    	num_of_trios = read_from_file(file_name, data); // Read and count trios from file.

    	// Get the perfect square of number of trios, or -1 if such one doesn't exists.
    	n = get_perfect_square(num_of_trios);

    	// Check that the number of trios has a perfect square, and is equal to the number of processes.
    	// Abort otherwise.
    	if (n == -1 || num_of_procs != n*n)
    	{
    		printf("Number of trios must be equal to number of processes, and must have an integer square.\n");
    		MPI_Abort(MPI_COMM_WORLD, 2);
    	}

    	trios = (Trio*) calloc(num_of_trios, sizeof(Trio)); // Initialize trios array.
    	create_trio_mat(trios, data, num_of_trios); // Populate the trios array.

    	printf("Unsorted array:\n");
    	print_trio_array(trios, n);
    }

    MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD); // Broadcast the perfect square of number of trios (n).

    comm = create_cart_comm(rank, n, my_coord); // Create a 2 dimensional Cartesian topology.

    MPI_Scatter(trios, 1, trio_type, &my_trio, 1, trio_type, ROOT, comm); // Send a trio for each one of the processes.

    shear_sort(comm, trio_type, &my_trio, my_coord, n); // Shear sort the trios.

    MPI_Gather(&my_trio, 1, trio_type, trios, 1, trio_type, ROOT, comm); // Receive sorted trios from all processes.

    // Print the sorted trios array.
    if (rank == ROOT)
    {
    	printf("Sorted array:\n");
    	print_trio_array(trios, n);
    }

    free(trios);

    MPI_Finalize();
    return 0;
}

void set_trio(Trio* trio, int x, int y, int z)
{
	trio->x = x;
	trio->y = y;
	trio->z = z;
}

void compare_and_exchange(MPI_Comm comm, MPI_Datatype trio_type, Trio* my_trio, int *coords, int dim, int doEven)
{
	MPI_Status status;
	Trio nbr_trio;
	int res, source, dest;

	// Get the processes rank from their coordinates, from the chosen dimension (rows or columns).
	MPI_Cart_shift(comm, dim, DISP, &source, &dest);

	// Odd on odd step / even on even step positioned processes, first sending their trio, then receiving their
	// neighbor's trio.
	if (coords[dim] % 2 == doEven)
	{
		if (dest != MPI_PROC_NULL)
		{
			MPI_Send(my_trio, 1, trio_type, dest, 0, comm);
			MPI_Recv(&nbr_trio, 1, trio_type, dest, 0, comm, &status);
		}
		else // For process with no neighbor to compare and exchange.
		{
			nbr_trio = *my_trio;
		}
	}
	// Odd on even step / even on odd step positioned processes, first receiving their neighbor's trio, then
	// sending their trio.
	else
	{
		if (source != MPI_PROC_NULL)
		{
			MPI_Recv(&nbr_trio, 1, trio_type, source, 0, comm, &status);
			MPI_Send(my_trio, 1, trio_type, source, 0, comm);
		}
		else
		{
			nbr_trio = *my_trio; // For process with no neighbor to compare and exchange.
		}
	}

	// compare trio with the neighbor's one.
	res = compare_trio(my_trio, &nbr_trio);

	// For sorting odd rows from right to left
	if (dim == ROWS_DIM && coords[0] % 2 == 1)
	{
		res *= -1;
	}

	// Exchange points if needed
	if (coords[dim] % 2 == doEven)
	{
		if (res > 0)
		{
			set_trio(my_trio, nbr_trio.x, nbr_trio.y, nbr_trio.z);
		}
	}
	else
	{
		if (res < 0)
		{
			set_trio(my_trio, nbr_trio.x, nbr_trio.y, nbr_trio.z);
		}
	}

	//printf("coordinates are %d %d, my trio: (%d, %d, %d), nbr trio: (%d, %d, %d), compare: %d\n",coords[0], coords[1], my_trio->x, my_trio->y, my_trio->z, nbr_trio.x, nbr_trio.y, nbr_trio.z, res);

}

void odd_even_sort(MPI_Comm comm, MPI_Datatype trio_type, Trio* my_trio, int *coords ,int dim, int n)
{
	int i;

	// Do one even then one odd compare and exchange for n times.
	for (i = 0; i < n; ++i) {
		compare_and_exchange(comm, trio_type, my_trio, coords, dim, i % 2);
		MPI_Barrier(comm); // Wait for all processes to end iteration.
	}

}

void shear_sort(MPI_Comm comm, MPI_Datatype trio_type, Trio* my_trio, int *coords, int n)
{
	int i;
	int iters = ceil(log2((double)n)) + 1;

	// Odd even sort rows in alternating directions, and afterward sort columns, for log2(n) + 1 iterations
	for (i = 0; i < iters; ++i) {
		// Sort rows
		odd_even_sort(comm, trio_type, my_trio, coords, ROWS_DIM, n);


		// Sort columns
		odd_even_sort(comm, trio_type, my_trio, coords, COLS_DIM, n);
	}

	// Finally odd even sort rows in alternating directions.
	odd_even_sort(comm, trio_type, my_trio, coords, ROWS_DIM, n);
}

int compare_trio(Trio *t1, Trio *t2)
{
	int res = t1->x - t2->x;
	if (res != 0) return res;
	res = t1->y - t2->y;
	if (res != 0) return res;
	return t1->z - t2->z;
}

MPI_Comm create_cart_comm(int rank, int n, int *coords)
{
	MPI_Comm comm;
	int dim[DIM], period[DIM], reorder;

	dim[0]=n; dim[1]=n; // n * n dimension matrix.
	period[0]=0; period[1]=0; // Non periodic.
	reorder=0; // Don't reorder ranks.
	MPI_Cart_create(MPI_COMM_WORLD, DIM, dim, period, reorder, &comm);
	MPI_Cart_coords(comm, rank, DIM, coords); // Get process coordinates.

	return comm;
}

int read_from_file(char* file_name, int *data)
{
	FILE* f;
	int val_counter=0;

	f = fopen(file_name, "r");
	while(fscanf(f, "%d %d %d %*c", &data[val_counter], &data[val_counter+1], &data[val_counter+2]) == 3)
	{
		val_counter += TRIO_SIZE;
	}
	fclose(f);

	return val_counter/TRIO_SIZE;
}

int get_perfect_square(int num)
{
	float fval = sqrt((double) num);
	int ival = fval;

	if (fval == ival)
	{
		return ival;
	}
	return -1;
}

void create_trio_datatype(MPI_Datatype* trio_type)
{
	int blockLengths[TRIO_SIZE] = { 1, 1, 1 };
	MPI_Aint disp[TRIO_SIZE];
	MPI_Datatype types[TRIO_SIZE] = { MPI_INT, MPI_INT, MPI_INT };

	disp[0] = offsetof(Trio, x);
	disp[1] = offsetof(Trio, y);
	disp[2] = offsetof(Trio, z);

	MPI_Type_create_struct(TRIO_SIZE, blockLengths, disp, types, trio_type);
	MPI_Type_commit(trio_type);
}

void create_trio_mat(Trio *trios, int *data, int num_of_trios)
{
	int i;

	for (i = 0; i < num_of_trios; ++i) {
		trios[i].x = data[i*TRIO_SIZE];
		trios[i].y = data[i*TRIO_SIZE + 1];
		trios[i].z = data[i*TRIO_SIZE + 2];
	}
}

void print_trio_array(Trio *trios, int n)
{
	int i, j;

	for (i = 0; i < n; ++i) {
		for (j = 0; j < n; ++j) {
			if (i % 2 == 0) // Print even rows from left to right.
			{
				printf("(%d, %d, %d)\n", trios[i*n + j].x, trios[i*n + j].y, trios[i*n + j].z);
			}
			else // Print odd rows from right to left.
			{
				printf("(%d, %d, %d)\n", trios[i*n + (n-j-1)].x, trios[i*n + (n-j-1)].y, trios[i*n + (n-j-1)].z);
			}
		}
	}
}
