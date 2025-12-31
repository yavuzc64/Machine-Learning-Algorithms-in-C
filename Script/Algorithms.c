#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
//#include <random.h>

#define N 50
#define VECTOR_SIZE (N * N + 1)
#define VECTOR_CHAR_SIZE 50000
#define TRAINING_SET_SIZE 202
#define MAX_ITERATION 500
#define EXPECTED_ERROR 0.001
#define LEARNING_RATE 0.0001

#define DEV 0

typedef struct {
    double data[VECTOR_SIZE];
    double type; 
} Vector;

double matrisCarpim (double*, Vector*);
double tanhTurev (double);
void initializeWeight(double*,double);
double gradientDescent(Vector*,double*,int, FILE*);
double stochasticGradientDescent(Vector* , double* , int ,FILE* );
double adamOptimization(Vector* , double* , int ,FILE* );

void shuffleVectors(Vector* , int );
double testModel(Vector* , int , double* );
void writeResultToCSV(FILE* file, int epoch, double loss, double time) {
    fprintf(file, "%d,%.6f,%.6f\n", epoch, loss, time);
}


int main() {
	double firstWeight[5]={-0.003,-0.002, 0.00, 0.002, 0.003};
    FILE *file = fopen("imagedata.csv", "r");
    if (!file) {
        printf("Dosya acilamadi");
        return 1;
    }
    else{
    	printf("dosya acildi");
	}
	int vector_count = 0;
	/*
		sabit arrayler memory de stackte tutuluyor
		fakat belli bir limitte stack kismi doluyor
		bu nedenle memorynin heap kisimini kullanmamiz gerek
		bunu da malloc yani dynamic memory allocation ile yapariz
	*/
	Vector *vectors = malloc(TRAINING_SET_SIZE * sizeof(Vector));
	if (!vectors) {
		printf("Bellek tahsisi basarisiz.\n");
		fclose(file);
		return 1;
	}       
	char *line = malloc(VECTOR_CHAR_SIZE * sizeof(char));
	if (!line) {
		printf("Bellek tahsisi basarisiz.\n");
		fclose(file);
		free(vectors);
		return 1;
	}
	double* weights = malloc(VECTOR_SIZE * sizeof(double));
    if (!weights) {
        printf("\nagirlik malloc hata.\n");
        free(vectors);
        return 1;
    }
	/*			datayi dosyadan okuma			*/
    while (fgets(line, VECTOR_CHAR_SIZE, file) && vector_count < TRAINING_SET_SIZE && !feof(file)) {
        char *token = strtok(line, ",");  // Ýlk giris
        int i = 0;

        while (token != NULL) {
            if (i < VECTOR_SIZE-1) {//sonunda bias var
                vectors[vector_count].data[i] = atof(token);  //index
            }
			else {//bias ve type durumu
            	vectors[vector_count].data[i] = atof(token);//bias
            	vectors[vector_count].type = atof(token);	//type
            }
            token = strtok(NULL, ",");// yeni index icin
            i++;
        }

        vector_count++;
    }
    free(line);
	fclose(file);  // Dosyayý kapat
	printf("\ndosya kapandi\n");
    
    
    shuffleVectors(vectors, TRAINING_SET_SIZE);
    
    int testSize = TRAINING_SET_SIZE * 0.2;  // Test kümesi boyutu (%20)
    int trainSize = TRAINING_SET_SIZE - testSize;  // Eðitim kümesi boyutu
	
    Vector* testSet = malloc(testSize * sizeof(Vector));  // Test kümesi
    Vector* trainSet = malloc(trainSize * sizeof(Vector));  // Eðitim kümesi

    if (!testSet || !trainSet) {
        printf("Bellek tahsisi basarisiz.\n");
        free(vectors);
        return 1;
    }
	int i ;
    for (i= 0; i < testSize; i++) {
        testSet[i] = vectors[i];
    }

    for ( i = 0; i < trainSize; i++) {
        trainSet[i] = vectors[testSize + i];
    }
    
    
    
	char filename[30];
	
	printf("\n\nGD	:\n");
    for(i = 0; i<5; i++){
    	sprintf(filename, "Gradient_D_W%d.csv", i + 1);
    	FILE* gdFile = fopen(filename, "w");
    	
		initializeWeight(weights, firstWeight[i]);
    	gradientDescent(trainSet, weights, trainSize, gdFile);
    	fclose(gdFile);
	}
    
    testModel(testSet,testSize,weights);
    
    printf("\n\nSGD	:\n");
    for(i = 0; i<5; i++){
    	sprintf(filename, "S_Gradient_D_W%d.csv", i + 1);
    	FILE* sgdFile = fopen(filename, "w");
    	
		initializeWeight(weights, firstWeight[i]);
    	stochasticGradientDescent(trainSet, weights, trainSize, sgdFile);
    	fclose(sgdFile);
	}
    
    testModel(testSet,testSize,weights);
    
    printf("\n\nADAM:\n");
    for(i = 0; i<5; i++){
    	sprintf(filename, "ADAM_W%d.csv", i + 1);
    	FILE* adamFile = fopen(filename, "w");
    	
		initializeWeight(weights, firstWeight[i]);
    	stochasticGradientDescent(trainSet, weights, trainSize, adamFile);
    	fclose(adamFile);
	}
    
    testModel(testSet,testSize,weights);
    
    free(testSet);
    free(trainSet);
    free(weights);
    free(vectors);
	
    return 0;
}



void shuffleVectors(Vector* vectors, int size) {//hazýr
    srand(time(NULL));
    int i,j;
    for (i = size - 1; i > 0; i--) {
        j = rand() % (i + 1);
        Vector temp = vectors[i];
        vectors[i] = vectors[j];
        vectors[j] = temp;
    }
}
double matrisCarpim (double* w, Vector* v){
	double result = 0.0;
	int i;
	for(i = 0; i < VECTOR_SIZE; i++){
		result += w[i] * v->data[i];
	}
	return result;
}

double tanhTurev (double a){
	return 1.0 - tanh(a) * tanh(a); 
}

void initializeWeight(double* weight, double value){
	int i;
	for(i = 0 ; i<VECTOR_SIZE ; i++){
		weight[i] = value;
	}
}

double gradientDescent(Vector* vec, double* weights, int setSize, FILE* file){
	clock_t start = clock();
	double loss=1, error, totalError=0;
	double temp, predictedY, gradient;
	int i, j, k, step=0;
	double zaman=0;
		
	
	//for(k = 0 ; k<1000; k++)
	while(loss > EXPECTED_ERROR && step <MAX_ITERATION)
	{
		totalError = 0;
		for(i=0;i<setSize;i++){
			temp = matrisCarpim(weights, &vec[i]);
			predictedY = tanh(temp);
			error = predictedY - vec[i].type; //		yeliz(1)
			
			totalError += (error*error)/2;
			
			gradient = error * tanhTurev(temp) ;
			
			for(j = 0; j<VECTOR_SIZE ; j++){
				weights[j] -= LEARNING_RATE * gradient * vec[i].data[j];
			}		
		}
		loss = totalError / setSize;  // Ortalama loss
		zaman = (double)(clock() - start) / CLOCKS_PER_SEC;
		writeResultToCSV(file, step, loss, zaman);
	    step++;
	    if(step%100 == 0&& DEV == 1)
	    	printf("Step: %d, Loss: %lf, Zaman: %lf\n", step, loss, zaman);
	    
	}
	
	
	
	//printf("\nGradien d iterasyon sayisi :  %d\nLOSS :  %lf", step, loss);
	/*
		sonda weight i cikti almayi unutma
	*/
	return loss;
}

double stochasticGradientDescent(Vector* vec, double* weights, int setSize, FILE* file){
	clock_t start = clock();
	double loss=1, error, totalError=0;
	double temp, predictedY, gradient;
	int miniBatch = 10;
	int i, j, k, step=0;
	float zaman=0 , eps = LEARNING_RATE;
	
	//initializeWeight(weights, 0.002);
	
	while(loss > EXPECTED_ERROR && step <MAX_ITERATION)
	{
		totalError = 0;
		for(i=0;i<miniBatch;i++){ //mini-batch ile gradyanti dengeliyoruz 10 tane kullandik
			k = rand() % TRAINING_SET_SIZE*0.8f;
			temp = matrisCarpim(weights, &vec[k]);
			predictedY = tanh(temp);
			error = predictedY - vec[k].type;
			
			totalError += (error*error)/2;
			
			gradient = error * tanhTurev(temp) ;
			
			for(j = 0; j<VECTOR_SIZE ; j++){
				weights[j] -= eps * gradient * vec[k].data[j];
			}		
			
		}
		loss = totalError / miniBatch;
	    step++;
	    zaman = (double)(clock() - start) / CLOCKS_PER_SEC;
	    writeResultToCSV(file, step, loss, zaman);
	    	//SGD de sona dogru titresimi azaltmak icin LEARNING_RATE azaltiyoruz
	    if (fmod(zaman, 0.005f) == 0.0f) {
		    eps *= 0.9f;
		}
	    
		if(step%100 == 0 && DEV == 1)
	    	printf("Step: %d, Loss: %lf, Zaman: %lf\n", step, loss, zaman);
		
	}
	return loss;
}

double adamOptimization(Vector* vec, double* weights, int setSize, FILE* file) {
    clock_t start = clock();
    
    //parametrelerin tanimlanmasi
    double beta1 = 0.9;     //first moment icin //daha küçük alýrsak daha gürültülü sonuclar alýrýz
    double beta2 = 0.999;   //second moment icin //ne kadar büyük olursa önceki gradientlerin etkisi o kadar büyük olur hem beta1 hem beta2 için
    double epsilon = 1e-8; 
    double loss = 1, error, totalError = 0;
    double temp, predictedY, gradient;
    int i, j, step = 0;
    double zaman = 0;
    double m_hat,v_hat;
    
    // first moment ve second moment icin hafizada dynamic yer acilmasi
    double* m = malloc(VECTOR_SIZE * sizeof(double));
    double* v = malloc(VECTOR_SIZE * sizeof(double));
    
    // baslangic degerlerini sifirladik
    for(i = 0; i < VECTOR_SIZE; i++) {
        m[i] = 0.0;
        v[i] = 0.0;
    }
    
    while(loss > EXPECTED_ERROR && step < MAX_ITERATION) {
        totalError = 0;
        
        // Her vector icin gradientlerin guncellenmesi
        for(i = 0; i < setSize; i++) {
            temp = matrisCarpim(weights, &vec[i]);
            predictedY = tanh(temp);
            error = predictedY - vec[i].type;
            
            totalError += (error * error) / 2;
            
            gradient = error * tanhTurev(temp);
            
            // ADAM algoritmasý icin gerekli guncellemeler
            for(j = 0; j < VECTOR_SIZE; j++) {
                // first moment guncellemesi
                m[j] = beta1 * m[j] + (1 - beta1) * gradient * vec[i].data[j];//gradientleri topluyoruz
                
                // second moment guncellemesi
                v[j] = beta2 * v[j] + (1 - beta2) * pow(gradient * vec[i].data[j], 2);//gradientlerin karelerini topluyoruz
                
                // bias düzeltmesi
                m_hat = m[j] / (1 - pow(beta1, step + 1)); //m_0ýn etkisini azaltmak icin .gradyanlarýn hareketli ortalamasý
                v_hat = v[j] / (1 - pow(beta2, step + 1)); //v_0ýn etkisini azaltmak icin .gradyan karelerinin hareketli ortalamasý
                
                // weights guncellemesi
                weights[j] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + epsilon);
            }
        }
        
        // Loss hesaplama
        loss = totalError / setSize;
        step++;
        zaman = (double)(clock() - start) / CLOCKS_PER_SEC;
        writeResultToCSV(file, step, loss, zaman);
        
        // developer modu icin output
        if(step % 100 == 0 && DEV == 1)
            printf("Step: %d, Loss: %lf, Zaman: %lf\n", step, loss, zaman);
    }
    if(step!=MAX_ITERATION && DEV == 1)
    	printf("Step: %d, Loss: %lf, Zaman: %lf\n", step, loss, zaman);
    
    // bellekte tutulan yerlerin free edilmesi
    free(m);
    free(v);
    
    return loss;
}

double testModel(Vector* testSet, int testSize, double* weights) {
    double totalError = 0.0;
    int correct = 0, i;
    for (i = 0; i < testSize; i++) {
        double temp = matrisCarpim(weights, testSet);
        double predictedY = tanh(temp);
        double actualY = testSet[i].type;


        if(predictedY>=0&&testSet[i].type==1){
        	correct++;
		}
		else if(predictedY<0&&testSet[i].type==-1){
        	correct++;
		}
    }
    //printf("Test Error: %lf\n", totalError / testSize);
    printf("Accuracy: %lf%%\n", (double)correct / testSize * 100.0);
    printf("dogru sayisi :%d\n", correct);
    return totalError / testSize;
}
