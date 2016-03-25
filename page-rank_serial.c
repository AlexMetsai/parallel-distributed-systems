/* Metsai Aleksandros 7723
 * metsalex@ece.auth.gr
 * 
 * Page Rank Serial
 */
 
 #include <stdlib.h>
 #include <stdio.h>
 #include <time.h>
 #include <math.h>
 #include <sys/time.h>
 
 #define ERROR 0.0001

 
 struct timeval startwtime, endwtime;
 double seq_time;
 
 //x is global to save data in main
 double *x;
 
 //number of repeats required for convergence
 int repeats;
 
 //total number of nodes
 int nodes;
 
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
  
  //Start timer here
  gettimeofday (&startwtime, NULL);
  
  page_rank();
  
  //End timer here
  gettimeofday (&endwtime, NULL);
	  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
	  printf("Serial clock time = %f\n", seq_time);
  
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
  return 0;
  
}
  
  
void page_rank(){

  int i, j;
  
  double p=(double)0.85;
  double delta=(double)(1-p)/nodes;
  
  x=(double*)malloc(nodes*sizeof(double));
  double *z=(double*)malloc(nodes*sizeof(double));  //calloc probably not needed right now, they swap in loop
  
  for(i=0; i<nodes;i++){
	x[i]= (double)1/nodes;		//possible error, might need to cast in double
  }
  
  double error;
  double temp_val;
  
  do{
		z=x;
		x=(double*)calloc(nodes, sizeof(double));	//zeroes
		double add_temp=0;
		
		for(j=0;j<nodes;j++){
			
			Connection temp=connections[j];
			
			if(temp.size==0){

				//if c(j) == 0
				add_temp=add_temp+(double)(z[j]/nodes); // to be used later for: x = x + z(j)/n;
				
			}else{
				
				temp_val=(double) z[j]/temp.size;	//CHECK HERE
				for(i=0;i<temp.size;i++){
					//x(L{j}) = x(L{j}) + z(j)/c(j);
					x[temp.index[i]]+=temp_val;
				}
			}
		}
		
		//compute error
		error=0;
		for(i=0;i<nodes;i++){
		
			x[i]=p*(x[i]+add_temp)+delta;
			temp_val=fabs(x[i]-z[i]); //absolute value
			if(temp_val>error) error=temp_val;
		}
		repeats++;
	}while(error>ERROR);
	
	
	

	
}
