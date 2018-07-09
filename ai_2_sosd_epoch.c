/*
AI Assignment 2
Parkhi Mohan - 201601061
V S Pragna - 201601106
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>

typedef struct neuralNet{
	double *wts, val, error;
} NEURALNET;

double randomGeneration(){
	// random number generation between 1 and 2
	return 0.1 + rand()/(9999999999.0);
}

NEURALNET** createNetwork(int numOfLayers, int networkLayers[]){
	NEURALNET** neuralNetwork = (NEURALNET**)malloc(numOfLayers*sizeof(NEURALNET *));
	int i, j, k;
	// adding +1 for bias in each layer
	for(i=0; i<numOfLayers; i++) neuralNetwork[i] = (NEURALNET*)malloc((networkLayers[i]+1)*sizeof(NEURALNET));
	for(i=1; i<numOfLayers; i++){
		// for when j=0
		neuralNetwork[i][0].val = 1.0;
		neuralNetwork[i][0].wts = NULL;
		neuralNetwork[i][0].error = 0.0;
		for(j=1; j<networkLayers[i]+1; j++){
			neuralNetwork[i][j].val = 0.0;
			// allocate memory space for the array
			neuralNetwork[i][j].wts = (double*)malloc((networkLayers[i-1]+1)*sizeof(double));
			for(k=0; k<networkLayers[i-1]+1; k++) neuralNetwork[i][j].wts[k] = randomGeneration();
			neuralNetwork[i][j].error = 0.0;
		}
	}
	return neuralNetwork;
}

void readFromFile(char* fileName, double* data[]){
	FILE* file = fopen(fileName, "r");
	char ch, values[100];
	int i=0, j=0, r=0, c=0;
	while((ch=getc(file))!=EOF){
		if(i==0){ 
			if(ch=='\n') i=1;
		}
		else{
			if(ch==',' || ch=='\n'){
				values[j] = '\0';
				data[r][c] = atoi(values);
				c++;
				if(c==17){
					c=0;
					r++;
				}
				j=0;	
			}
			else values[j++] = ch;
		}
	}
	fclose(file);
}

void networkInputLayer(NEURALNET** neuralNetwork, int numOfFeatures, double rowWise[]){
	int i;
	neuralNetwork[0][0].val = 1.0;
	neuralNetwork[0][0].wts = NULL;
	neuralNetwork[0][0].error = 0.0;

	for(i=1; i<numOfFeatures+1; i++){
	neuralNetwork[0][i].val = rowWise[i-1];
	neuralNetwork[0][i].wts = NULL;
	neuralNetwork[0][i].error = 0.0;
	}
}


double sigmoid(double x){
    return 1.0/(1.0 + exp(0.0-x));
}

void networkActivation(NEURALNET** neuralNetwork, int numOfLayers, int networkLayers[]){
	int i, j, k;
	for(i=1; i<numOfLayers; i++){
		for(j=1; j<networkLayers[i]+1; j++){
			double temp = 0.0;
			for(k=0; k<networkLayers[i-1]+1; k++){
				if(i==1 || k==0) temp += neuralNetwork[i][j].wts[k]*neuralNetwork[i-1][k].val;
				else temp += neuralNetwork[i][j].wts[k]*sigmoid(neuralNetwork[i-1][k].val);
			}
			neuralNetwork[i][j].val = temp;
		}
	}
}

void errorBackPropagation(NEURALNET** neuralNetwork, int numOfLayers, int networkLayers[], int answer[]){
	int i, j, k;
	for(i=numOfLayers-1; i>0; i--){
		for(j=1; j<=networkLayers[i]; j++){
			if(i==(numOfLayers-1)) neuralNetwork[i][j].error = ((answer[j-1])-sigmoid(neuralNetwork[i][j].val))*(1.0-sigmoid(neuralNetwork[i][j].val))*sigmoid(neuralNetwork[i][j].val);
			else{
				double temp = 0.0;
				for(k=1; k<=networkLayers[i+1]; k++) temp += neuralNetwork[i+1][k].error * neuralNetwork[i+1][k].wts[j]*(sigmoid(neuralNetwork[i][j].val)*(1.0- sigmoid(neuralNetwork[i][j].val)));
				neuralNetwork[i][j].error = temp;
			}
		}
	}
}

void updateWeights(NEURALNET** neuralNetwork, int numOfLayers, int networkLayers[], int eta){
	int i, j, k;
	for(i=1; i<numOfLayers; i++){
		for(j=1; j<=networkLayers[i]; j++){
			for(k=0; k<=networkLayers[i-1]; k++){
				if(i==1 || k==0) neuralNetwork[i][j].wts[k] += eta*neuralNetwork[i-1][k].val*neuralNetwork[i][j].error;
				else neuralNetwork[i][j].wts[k] += eta*sigmoid(neuralNetwork[i-1][k].val)*neuralNetwork[i][j].error;
			}
		}
	}
}

void printNeuralNetwork(NEURALNET** neuralNetwork, int numOfLayers, int networkLayers[]){
    int i,j,k;
    for(i=0;i<numOfLayers;i++){
        printf("LAYER %d:\n", i+1);
        for(j=0;j<=networkLayers[i];j++){
            if(i!=0 && j!=0) printf("%lf-> ",sigmoid(neuralNetwork[i][j].val));
            else printf("%lf-> ",neuralNetwork[i][j].val);
            if(neuralNetwork[i][j].wts!=NULL){
                for(k=0;k<=networkLayers[i-1];k++) printf("%lf, ",neuralNetwork[i][j].wts[k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
void normalize(double* trainfile[], int rows, int cols){
	int i,j;
    double* maxValues=(double *)malloc((cols-1)*sizeof(double));
    double* minValues=(double *)malloc((cols-1)*sizeof(double));
    for(i=0;i<cols-1;i++){
        maxValues[i]=-1.0;
        minValues[i]=trainfile[0][i+1];
    }
    for(i=0;i<rows-1;i++){
        for(j=1;j<cols;j++){
            if(trainfile[i][j] > maxValues[j-1]) maxValues[j-1]=trainfile[i][j];
            if(trainfile[i][j] < minValues[j-1]) minValues[j-1]=trainfile[i][j];
        }
    }
    for(i=0;i<rows-1;i++){
        for(j=1;j<cols;j++){
            trainfile[i][j]=((trainfile[i][j]-minValues[j-1])/(maxValues[j-1]-minValues[j-1]));
        }
    }
}

void plotGraph(int hidden[], double accuracyArray[]){
	char * commandsForGnuplot[] = {"set title \"SUM OF SQUARED DEVIATION LOSS WITH EPOCH\"", "set xlabel \"NODES IN HIDDEN LAYER\"" , "set ylabel \"ACCURACY\"", "plot 'data.temp' with linespoints title 'ACCURACY'"};
    FILE * temp = fopen("data.temp", "w");
    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
    int i;
    for (i=0; i < 6 ; i++) fprintf(temp, "%d %lf \n", hidden[i], accuracyArray[i]); //Write the data to a temporary file

    for (i=0; i < 4 ; i++) fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.
}

int main(){

	int hidden[6] = {5, 6, 7, 8, 9, 10}, check=1;
	double accuracyArray[6];
	while(check<7){	
		srand((unsigned int)time(NULL));
		int numOfLayers = 3, learningRate = 0.001, epoch = 0, networkLayers[3] = {16, hidden[check-1], 10}, i, j;
		int trainFileR = 2217, trainFileC = 17, testFileR = 999, testFileC = 17;
		double **trainData = (double**)malloc((trainFileR-1)*sizeof(double*)), **testData = (double**)malloc((testFileR-1)*sizeof(double*));
		for(i=0; i<trainFileR; i++) trainData[i] = (double*)malloc(trainFileC*sizeof(double));
		for(i=0; i<testFileR; i++) testData[i] = (double*)malloc(testFileC*sizeof(double));

		char trainFileData[] = "/home/parkhi/Desktop/trainData.csv";
		readFromFile(trainFileData, trainData);
		normalize(trainData,trainFileR,trainFileC);

		char testFileData[] = "/home/parkhi/Desktop/testData.csv";
		readFromFile(testFileData, testData);
		normalize(testData,testFileR,testFileC);

		NEURALNET** neuralNetwork = createNetwork(numOfLayers, networkLayers);
		
		double *rowWise = (double*)malloc(16*sizeof(double));
		int *answer = (int*)malloc(10*sizeof(int));

		while(epoch<100){
			for(i=0; i<trainFileR-1; i++){
					for(j=0; j<10; j++) answer[j] = 0;
					answer[(int)trainData[i][0]-1] = 1;
					for(j=1; j<trainFileC; j++) rowWise[j-1] = trainData[i][j];
					// using rowWise and answer to create input layer
					networkInputLayer(neuralNetwork, networkLayers[0], rowWise);
					/*int p;
					for(p=0; p<=16; p++) printf("%lf ", neuralNetwork[0][p].val);
					printf("\n");*/

					// updating the neural network
					networkActivation(neuralNetwork, numOfLayers, networkLayers);
					errorBackPropagation(neuralNetwork, numOfLayers, networkLayers, answer);
					updateWeights(neuralNetwork, numOfLayers, networkLayers, learningRate);
			}
			epoch++;
		}

		//printNeuralNetwork(neuralNetwork, numOfLayers, networkLayers);

		// testing the neural network
		double accuracy;
	    int correct=0, incorrect=0;
	    int maxClass=0;
	    for(i=0;i<testFileR-1;i++){
	        for(j=1;j<testFileC;j++) rowWise[j-1]=testData[i][j];
	        networkInputLayer(neuralNetwork, networkLayers[0], rowWise);
	        networkActivation(neuralNetwork, numOfLayers, networkLayers);
	        double maxAct=-1.0;
	        for(j=1;j<=networkLayers[numOfLayers-1];j++){
	            if(sigmoid(neuralNetwork[numOfLayers-1][j].val) > maxAct){
	                maxAct=sigmoid(neuralNetwork[numOfLayers-1][j].val);
	                maxClass=j;
	            }
	        }
	        if(maxClass==((int)testData[i][0])) correct += 1;
	        else incorrect += 1;
	    }
	    accuracy=(correct*1.0/(correct+incorrect))*100.0; 
	    accuracyArray[check-1] = accuracy;
	    /*printf("HIDDEN: %d\n", hidden[check-1]);
	    printf("ACCURACY: %lf\n",accuracy);*/
	    check++;
	}
    int ite;
    for(ite=0; ite<6; ite++){
    	printf("HIDDEN: %d\n", hidden[ite]);
    	printf("ACCURACY: %lf\n",accuracyArray[ite]);
    }
    plotGraph(hidden, accuracyArray);
	return 0;
}
