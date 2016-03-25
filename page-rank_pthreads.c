/* Metsai Aleksandros 7723
 * metsalex@ece.auth.gr
 * 
 * Page Rank pthreads
 */
 
 #include <stdlib.h>
 #include <stdio.h>
 #include <time.h>
 #include <math.h>
 #include <sys/time.h>
 #include <pthread.h>
 
 #define ERROR 0.0001

 
 struct timeval startwtime, endwtime;
 double seq_time;
 
 //all the arrays global so there won't be need for structs
 double *x;
 double *z;
 double error;
 double p;
 double delta;
 double temp_add;
 
 //number of repeats required for convergence
 int repeats;
 
 //total number of nodes
 int nodes;
 
 //number of threads
 int threads;
 
 pthread_mutex_t lock= PTHREAD_MUTEX_INITIALIZER;
 
 pthread_t *threads_array;
 
 //used to define which part of the data a thread will process
 typedef struct{
	int begin;
	int end;
	int id;
 }Thread_job;
 
 Thread_job *thread_jobs;
 
 //The struct which implements the connections
 typedef struct{
 int size;
 int* index;
 } Connection;
 
 Connection *connections;
 
 void page_rank();
 
int main(){

  int i, j;
  
  printf("Set the number of nodes\n");
  scanf("%d", &nodes);
  printf("Set the number of threads\n");
  scanf("%d", &threads);
  
  repeats=0;
  
  char* filename="data.txt";
  
  connections= (Connection *)malloc(nodes*sizeof(Connection));
  
  for(i=0;i<nodes;i++){
	connections[i].size=0;
	connections[i].index=(int*)calloc(0, sizeof(int));	//not sure if malloc will work here
  }
  
  //Read from file
  FILE *f=fopen(filename, "r");
  if(f==NULL){
	printf("Error in opening file\n");
	return 1;
  }
  
  
  char* type= "%d\t%d\n";
  
  
  while(!feof(f)){
	
	if(fscanf(f, type, &j, &i)){		//no1 location for error
	
		
		if(j>nodes || i>nodes){
			//ignore higher nodes
			//this will probably never happen
			//keep it in case of wrong data or nodes
			continue;
		}
		
		//correct matlab's indexing
		//j--;
		//i--;
		
		//No2 location for error
		//Add the connection to the Array
		connections[j].size++;
		connections[j].index=(int*)realloc(connections[j].index, connections[j].size*sizeof(int));
		connections[j].index[connections[j].size-1]= i;
	}
  }

  
  
  //close file
  fclose(f);
  
  //allocate thread arrays
  threads_array= (pthread_t *)malloc(threads*sizeof(pthread_t));
  thread_jobs= (Thread_job*)malloc(threads*sizeof(Thread_job));
  
  int nodes_per_thread= ceil((double) nodes/threads); //ceil(x)= smallest int > x
  int data_jump=0;
  
  for(i=0;i<threads;i++){
  
	thread_jobs[i].id=i;
	thread_jobs[i].begin=data_jump;
	
	data_jump+= nodes_per_thread;
	
	if(data_jump>nodes){
		thread_jobs[i].end= nodes-1;
	}else{
		thread_jobs[i].end= data_jump;
	}
  }
  
  //Start timer here
  gettimeofday (&startwtime, NULL);
  
  page_rank();
  
  //End timer here
  gettimeofday (&endwtime, NULL);
	  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
	  printf("Parallel clock time = %f\n", seq_time);
  
  printf("Number of repeats for convergence %d\n", repeats);
  
  //Save probabilities to file
  f=fopen("x_out.txt", "w");
  if (f==NULL){
	printf("Error saving data\n");
	return 2;
  }
  for(i=0;i<nodes;i++)
	fprintf(f,"%f\n",x[i]);
  fclose(f);
  
  double temp_val=0;
  for(i=0;i<nodes;i++) temp_val+=x[i];
  printf("Total sum of probabilities %f\n", temp_val);
  
  //free
  for(i=0;i<nodes;i++){
	free(connections[i].index);
  }
  free(connections);
  free(threads_array);
  free(thread_jobs);
  free(x);
  free(z);
  
  return 0;
  
}

 
void* copy_data(void* temp_thread_job){
  
  Thread_job *thread_job= (Thread_job*)temp_thread_job;
  int i;
  for(i=thread_job->begin;i<thread_job->end;i++){
	z[i]=x[i];
	x[i]=0;
  }
  pthread_exit(NULL);
}

void* first_loop(void* temp_thread_job){

  Thread_job *thread_job= (Thread_job*)temp_thread_job;
  int i, j;
  double temp_val;
  double temp_add_local=0;
  
  for(j=thread_job->begin; j<thread_job->end; j++){ //no1 location for error
  
	Connection temp= connections[j];
	if(temp.size==0){
		//if c(j) == 0
		temp_add_local+=(double)(z[j]/nodes);
	}else{
	
		temp_val=(double) z[j]/temp.size;	
		
		for(i=0;i<temp.size;i++){
			x[temp.index[i]]+=temp_val;
		}
	}
  }
  //add the possibility to global variable
  //careful for data race
  pthread_mutex_lock(&lock);
  
  temp_add+=temp_add_local;
  pthread_mutex_unlock(&lock);
  
  pthread_exit(NULL);
}

void* second_loop(void* temp_thread_job){

  Thread_job *thread_job= (Thread_job*)temp_thread_job;
  
  int i;
  double temp_val;
  double max=0;
  
  for(i=thread_job->begin; i<thread_job->end; i++){
	
	x[i]=  p*(x[i]+temp_add) +delta;  
	temp_val=fabs(x[i]-z[i]); //absolute value
	if(temp_val>max) max= temp_val;
  }
  
  //find global max error
  pthread_mutex_lock(&lock);
  
  if(error<max) error=max;
  pthread_mutex_unlock(&lock);
  
  pthread_exit(NULL);
}
	


  
void page_rank(){
  
  int i, j;
  
  p= (double)0.85;
  delta= (double)(1-p)/nodes;
  
  x=(double*)malloc(nodes*sizeof(double));
  z=(double*)malloc(nodes*sizeof(double));
  
  for(i=0; i<nodes;i++){
	x[i]= (double)1/nodes;		//possible error, might need to cast in double 
  }
  
  do{
	
	temp_add=0;
	
	//copy x to z
	for(i=0;i<threads;i++)
		pthread_create(&threads_array[i], NULL, &copy_data, (void*)&thread_jobs[i]);
		
	for(i=0;i<threads;i++)
		pthread_join(threads_array[i], NULL);
	
	//first loop
	for(i=0;i<threads;i++)
		pthread_create(&threads_array[i], NULL, &first_loop, (void*)&thread_jobs[i]);
	
	for(i=0;i<threads;i++)
		pthread_join(threads_array[i], NULL);
		
	error=0;
	//second loop
	for(i=0;i<threads;i++)
		pthread_create(&threads_array[i], NULL, &second_loop, (void*)&thread_jobs[i]);
	
	for(i=0;i<threads;i++)
		pthread_join(threads_array[i], NULL);
	
	repeats++;
	}while(error>ERROR);
}
