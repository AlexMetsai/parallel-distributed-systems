/*
 The bitonic sort is also known as Batcher Sort. 
 For a reference of the algorithm, see the article titled 
 Sorting networks and their applications by K. E. Batcher in 1968 


 The following codes take references to the codes avaiable at 

 http://www.cag.lcs.mit.edu/streamit/results/bitonic/code/c/bitonic.c

 http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm

 http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm 
 */

/* 
------- ---------------------- 
   Nikos Pitsianis, Duke CS 
-----------------------------
*/

/* 
------- ---------------------- 
   Metsai Aleksandros, 7723
-----------------------------
*/


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

struct timeval startwtime, endwtime;
double seq_time;


const int ASCENDING  = 1;
const int DESCENDING = 0;


void init(int *a, int size);
void print(int *a, int N);
void test(int *a, int N);
inline void exchange(int *a, int i, int j);
void compare(int *a, int i, int j, int dir);
void myCompare(int *a, int *b, int dir, int size);
void bitonicMerge(int *a, int lo, int cnt, int dir);
void recBitonicSort(int *a, int lo, int cnt, int dir);


/** the main program **/
int main(int argc, char **argv) {


  if (argc != 2) {
    printf("Usage: %s p q\n  where p are the tasks and n=2^q is problem size (power of two)\n", 
	   argv[0]);
    exit(1);
  }
  int rc, numtasks, rank, p, q, size, source, dest;
  int *a, *b, *A;
  p=atoi(argv[0]); ///MIGHT not use
  q=atoi(argv[1]); ///2^q is the vector lenght
  ///N = 1<<atoi(argv[1]);
  ///a = (int *) malloc(N * sizeof(int));
  
  size=1<<q;
  

 // ***START***
 
  MPI_Status Stat;

  
 //----Initialising MPI Enviroment----
 
  rc=MPI_Init(&argc, &argv);
  if(rc!=MPI_SUCCESS){
	  printf("Error starting MPI. Terminating.\n");
	  MPI_Abort(MPI_COMM_WORLD, rc);
  }
  
  int i; ///CAREFUL!!! ***
  
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0){
	   printf("Number of tasks = %d My rank = %d (Master Task)\n", numtasks, rank);
	   ///printf("Size of each vector = %d\n", size);  /this will overflow in big sizes use long int or dont use at all
	   ///printf("Total Size = %d\n",size*numtasks);  /this will overflow in big sizes
   }
  
  if(rank==0){
	  A=(int *)malloc((numtasks*size)*sizeof(int));
	  if(A==NULL){
		printf("Memory Allocation Failed in last task(the one gathering the data). Terminating\n");
		exit(1);
	  }
  }
	  
	
  a=(int *)malloc(size*sizeof(int));
  if(a==NULL){
	printf("First Memory Allocation Failed in task %d. Terminating\n", rank);
	exit(1);
  }
  b=(int *)malloc(size*sizeof(int));
  if(b==NULL){
	printf("Second Memory Allocation Failed in task %d. Terminating\n", rank);
	exit(2);
  }
  ///Fill the vector
  init(a, size);
  MPI_Barrier(MPI_COMM_WORLD); ///Wait for everybody
  
  ///Start the timer  here from the Master
  if(rank==0) gettimeofday (&startwtime, NULL);
  
  ///Sort Each sequence in reverse directions
  if(rank%2==0){
	  recBitonicSort(a, 0, size, ASCENDING);
  }else if(rank%2==1){
	  recBitonicSort(a, 0, size, DESCENDING);
  }
  
  int dir, jump, j;
  MPI_Barrier(MPI_COMM_WORLD);
  ///Sort all the sequences  -shmeiwsh: poly perierga moy bgike, no1 meros gia error
  for(i=1;i<numtasks;(i=i*2)){
	  if((rank%(i*4))<(2*i)){
		  dir=ASCENDING;
	  }else if((rank%(i*4))>=(2*i)){
		  dir=DESCENDING;
	  }
	  for(jump=i;jump>=1;(jump=(jump/2))){
		  if((rank%(jump*2))<jump){
			  source=rank+jump;
			  MPI_Recv(b, size, MPI_INT, source, 1, MPI_COMM_WORLD, &Stat);
			  myCompare(a, b, dir, size);
			  dest=source;
			  MPI_Ssend(b, size, MPI_INT, dest, 1, MPI_COMM_WORLD); ///Send the Vector back
		  }else if((rank%(jump*2))>=jump){
			  dest=rank-jump;
			  MPI_Ssend(a, size, MPI_INT, dest, 1, MPI_COMM_WORLD);
			  source=dest;
			  MPI_Recv(a, size, MPI_INT, source, 1, MPI_COMM_WORLD, &Stat);
		  }
			MPI_Barrier(MPI_COMM_WORLD);
		  	  //if(jump==1) break;
	  }
	  bitonicMerge(a, 0, size, dir);  ///Episis simeio me pithana errors, na thimithw na chekarw
	  MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  ///End the timer here
  if (rank==0){ 
	  gettimeofday (&endwtime, NULL);
	  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
	  printf("MPI clock time = %f\n", seq_time);
  }  
  

  ///Master rank collects the results for testing
  if(rank==0){
	  int j;
	  for(j=0;j<size;j++){
		  A[j]=a[j];
		  //printf("rank 0 vector\n");
		  //print(a, size);
	  }
	  for(i=1;i<numtasks;i++){
		  source=i;
		  MPI_Recv(b, size, MPI_INT, source, 1, MPI_COMM_WORLD, &Stat);
		  for(j=0;j<size;j++){
			  A[(i*size)+j]=b[j];
		  }
		  //printf("rank %d vector\n", source);
		  //print(b, size);
	  }
  }else if(rank>0){
	  dest=0;
	  MPI_Ssend(a, size, MPI_INT, dest, 1, MPI_COMM_WORLD);
  }

  
	MPI_Barrier(MPI_COMM_WORLD);
	free(a); ///free the vectors
	free(b);
  
  
  if(rank==0){
	test(A, (numtasks*size));
	//printf("rank %d\n", rank);
	//print(A, numtasks*size);
	free(A);
  }
    //----Finalising MPI Enviroment---- ***FINISH****
   
  MPI_Finalize();
  

}

/** -------------- SUB-PROCEDURES  ----------------- **/ 

/** procedure test() : verify sort results **/
void test(int *a, int N) {
  int pass = 1;
  int i;
  for (i = 1; i < N; i++) {
    pass &= (a[i-1] <= a[i]);
  }

  printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
}


/** procedure init() : initialize array "a" with data **/
void init(int *a, int size) {
  int i;
  for (i = 0; i < size; i++) {
    a[i] = rand() % size; // (N - i);
  }
}

/** procedure  print() : print array elements **/
void print(int *a, int N) {
  int i;
  for (i = 0; i < N; i++) {
    printf("%d\n", a[i]);
  }
  printf("\n");
}


/** INLINE procedure exchange() : pair swap **/
inline void exchange(int *a, int i, int j) {
  int t;
  t = a[i];
  a[i] = a[j];
  a[j] = t;
}



/** procedure compare() 
   The parameter dir indicates the sorting direction, ASCENDING 
   or DESCENDING; if (a[i] > a[j]) agrees with the direction, 
   then a[i] and a[j] are interchanged.
**/
inline void compare(int *a, int i, int j, int dir) {
  if (dir==(a[i]>a[j])) 
    exchange(a, i,j);
}




/** Procedure bitonicMerge() 
   It recursively sorts a bitonic sequence in ascending order, 
   if dir = ASCENDING, and in descending order otherwise. 
   The sequence to be sorted starts at index position lo,
   the parameter cbt is the number of elements to be sorted. 
 **/
void bitonicMerge(int *a, int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    int i;
    for (i=lo; i<lo+k; i++)
      compare(a, i, i+k, dir);
    bitonicMerge(a, lo, k, dir);
    bitonicMerge(a, lo+k, k, dir);
  }
}



/** function recBitonicSort() 
    first produces a bitonic sequence by recursively sorting 
    its two halves in opposite sorting orders, and then
    calls bitonicMerge to make them in the same order 
 **/
void recBitonicSort(int *a, int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    recBitonicSort(a, lo, k, ASCENDING);
    recBitonicSort(a, lo+k, k, DESCENDING);
    bitonicMerge(a, lo, cnt, dir);
  }
}


/** function sort() 
   Caller of recBitonicSort for sorting the entire array of length N 
   in ASCENDING order
 **/
 /*
void sort() {
  recBitonicSort(0, N, ASCENDING);
}
*/

///My function to compare
void myCompare(int *a, int *b, int dir, int size){
	int i, t;
	for(i=0;i<size;i++){
		  if (dir==(a[i]>b[i]))
		  {
			t = a[i];
			a[i]=b[i];
			b[i]=t;
		}
	}	
}
