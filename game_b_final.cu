/* Metsai Aleksandros 7723
 * metsalex@ece.auth.gr
 * 
 * Multiple cells per thread
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define THRESHOLD 0.4


#define CELLS_PER_THREAD 2
#define THREADS_PER_BLOCK (500/CELLS_PER_THREAD)


struct timeval startwtime, endwtime;
double seq_time;


__global__ void game_b (int *newer, int *old, int N)
{
	
  int index= blockIdx.x*blockDim.x*CELLS_PER_THREAD + threadIdx.x*CELLS_PER_THREAD;

  int count;
  for(count=0;count<CELLS_PER_THREAD;count++){
	int sum=0;
  
	int i = (int)index/N;
	int j=index%N;
  
	if(i==0){
	  
	  //top
	  
	  if(j==0){
		  
		  sum= old[i*N+(j+1)]+ old[N-1]
		  +old[N] + old[N+1] + old[2*N-1]
		  +old[N*(N-1)]+old[N*(N-1) +1] +old[N*N -1];
		  
	  }else if(j==(N-1)){
		  
		   sum= old[0]+ old[j-1]
		  +old[j+N] + old[j+1] +old[j+N-1]
		  +old[N*N-1]+old[N*N-2] + old[N*(N-1)];
		  
	  }else{
		  
		 sum= old[j+1] + old[j-1]
		  +old[j+N] +old[N+(j+1)] +old[N+(j-1)]
		  +old[N*(N-1)+j]+old[N*(N-1)+(j+1)]+old[N*(N-1)+(j-1)];
		  
	  } 
	}
	else if(i==(N-1)){
	 
	  if(j==0){		
		  
		//bottom-left
		
		sum = old[i*N +1] + old[i*N +(N-1)]
		+old[0] + old[1] +old[(N-1)]
		+old[(i-1)*N] +old[(i-1)*N +1] +old[(i-1)*N +(N-1)];
		  
	  }else if(j==(N-1)){		
		  
		  //bottom-right
		  
		  sum = old[i*N] + old[i*N +(j-1)]
		  +old[N-1] + old[N-2] + old[0]
		  +old[(i-1)*N +j] +old[(i-1)*N +(j-1)] +old[(i-1)*N];
		  
	  }else
	  {
		  //bottom
		  
		  sum= old[i*N +(j+1)] +old[i*N +(j-1)] 
		  +old[j] +old[j+1] +old[j-1]
		  +old[(i-1)*N +j] +old[(i-1)*N +(j+1)] +old[(i-1)*N + (j-1)];
			
	  }
	  
	}else if(j==0){			
	  
	  //left
	  
	  sum= old[i*N +1] +old[i*N + (N-1)]
	  +old[(i+1)*N] +old[(i+1)*N + 1] + old[(i+1)*N +(N-1)]
	  +old[(i-1)*N] +old[(i-1)*N +1] +old[(i-1)*N +(N-1)]; 
	  
	}else if(j==(N-1)){
	  
	  /* right */
	  
	  sum= old[i*N] + old[i*N +(j-1)]
	  +old[(i+1)*N +j] +old[(i+1)*N +(j-1)] +old[(i+1)*N]
	  +old[(i-1)*N +j] +old[(i-1)*N +(j-1)] +old[(i-1)*N];
	  	  
	}else{
	 
	  /* General Case */
	  
	  sum= old[i*N+(j+1)] + old[i*N +(j-1)]
	  +old[(i+1)*N + j] +old[(i+1)*N +(j+1)] +old[(i+1)*N +(j-1)]
	  +old[(i-1)*N + j] +old[(i-1)*N +(j+1)] +old[(i-1)*N +(j-1)];
	  
	}
  
	switch(sum){
	  case 3:
	  newer[i*N + j] = 1;
	  break;
	  
	  case 2:
	  newer[i*N + j] = old[i*N + j];
	  break;
	  
	  default:
	  newer[i*N + j]=0;
	}
	index++;
  }
  
}


void read_from_file(int *X, char *filename, int N);
void save_table(int *X, int N);


int main(){
	

	int *table;
	int* newer;
	int* old;
	int *temp;
	
	int blocks, t, N, count;

	
	printf("Set the number of generations\n");
	scanf("%d", &t);
	printf("Set N (table size = NxN)\n");
	scanf("%d", &N);
	int size=N*N*sizeof(int);
	
	/*
	 Insert table here
	 */
	
	
	char filename[20];
	sprintf(filename, "table%dx%d.bin", N, N);
	printf("Reading %dx%d table from file %s\n", N, N, filename);
	table = (int *)malloc(N*N*sizeof(int));
	read_from_file(table, filename, N);  	
	

	printf("This is kernel b\n");
	
	printf("The game will be played for %d generations N=%d\n", t, N);
	
	//!!!Start Timer!!!
	gettimeofday (&startwtime, NULL);
	
	//Allocate space of new and old in device
	cudaMalloc(&newer, size);
	cudaMalloc(&old, size);
	
	cudaMemcpy(old, table, size, cudaMemcpyHostToDevice);
	
	blocks=(N*N)/(THREADS_PER_BLOCK*CELLS_PER_THREAD);

	//Play game for t generations
	for(count=0;count<t;count++){

		game_b<<<blocks, THREADS_PER_BLOCK>>>(newer, old, N);
		cudaThreadSynchronize();
		
		//swap pointers
		temp=old;
		old=newer;
		newer=temp;
	}
	
	//copy back table
	cudaMemcpy(table, old, size, cudaMemcpyDeviceToHost);
	
	//!!!End Timer!!!
	gettimeofday (&endwtime, NULL);
	  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
	  printf("Cuda clock time = %f\n", seq_time);
	

	save_table(table, N);
	
	cudaFree(newer);
	cudaFree(old);
	
	free(table);
	
	return(0);
}


void read_from_file(int *X, char *filename, int N){

  FILE *fp = fopen(filename, "r+");

  int size = fread(X, sizeof(int), N*N, fp);

  printf("elements: %d\n", size);

  fclose(fp);

}

void save_table(int *X, int N){

  FILE *fp;

  char filename[20];

  sprintf(filename, "cuda_b_table%dx%d.bin", N, N);

  printf("Saving table in file %s\n", filename);

  fp = fopen(filename, "w+");

  fwrite(X, sizeof(int), N*N, fp);

  fclose(fp);

}





