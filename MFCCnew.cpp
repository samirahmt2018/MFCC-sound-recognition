#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#include<iostream>
#include<fstream>
#include<cmath>
#include<vector>
#include <complex> 
#include <bitset> 

using namespace std;
struct  MFCC_DATA{
    long length;
    double *data;
};
void Discrete_Cosine_Transform(int direction, int length, double X[]){
	double pi = 3.14159265358979323846;

	double *x = new double[length];

	for (int i = 0; i < length; i++){
		x[i] = X[i];
	}
	for (int k = 0; k < length; k++){
		double sum = 0;

		if (direction == 1){
			for (int n = 0; n < length; n++){
				sum += ((k == 0) ? (sqrt(0.5)) : (1)) * x[n] * cos(pi * (n + 0.5) * k / length);
			}
		}
		else
		if (direction == -1){
			for (int n = 0; n < length; n++){
				sum += ((n == 0) ? (sqrt(0.5)) : (1)) * x[n] * cos(pi * n * (k + 0.5) / length);
			}
		}
		X[k] = sum * sqrt(2.0 / length);
	}
	delete[] x;
}
void DCT(int direction, int length, double X[]){
	if (direction == 1 || direction == -1){
		Discrete_Cosine_Transform(direction, length, X);
		return;
	}
	fprintf(stderr, "[DCT], direction = {-1 (inversed transform), 1 (forward transform)}\n");
}
void Fast_Fourier_Transform(int direction, int length, double Xr[], double Xi[]){
	int log_length = (int)log2((double)length);

	double pi = 3.14159265358979323846;

	for (int i = 0, j = 0; i < length; i++, j = 0){
		for (int k = 0; k < log_length; k++){
			j = (j << 1) | (1 & (i >> k));
		}
		if (j < i){
			double t;

			t = Xr[i];
			Xr[i] = Xr[j];
			Xr[j] = t;

			t = Xi[i];
			Xi[i] = Xi[j];
			Xi[j] = t;
		}
	}
	for (int i = 0; i < log_length; i++){
		int L = (int)pow(2.0, i);

		for (int j = 0; j < length - 1; j += 2 * L){
			for (int k = 0; k < L; k++){
				double argument = direction * -pi * k / L;

				double xr = Xr[j + k + L] * cos(argument) - Xi[j + k + L] * sin(argument);
				double xi = Xr[j + k + L] * sin(argument) + Xi[j + k + L] * cos(argument);

				Xr[j + k + L] = Xr[j + k] - xr;
				Xi[j + k + L] = Xi[j + k] - xi;
				Xr[j + k] = Xr[j + k] + xr;
				Xi[j + k] = Xi[j + k] + xi;
			}
		}
	}
	if (direction == -1){
		for (int k = 0; k < length; k++){
			Xr[k] /= length;
			Xi[k] /= length;
		}
	}
}
void FFT(int direction, int length, double Xr[], double Xi[]){
	int log_length = (int)log2((double)length);

	if (direction != 1 && direction != -1){
		fprintf(stderr, "[FFT], direction = {-1 (inversed transform), 1 (forward transform)}\n");
		return;
	}
	if (1 << log_length != length){
		fprintf(stderr, "[FFT], length must be a power of 2\n");
		return;
	}
	Fast_Fourier_Transform(direction, length, Xr, Xi);
}
double Mel_Scale(int direction, double x){
	switch (direction){
	case -1:
		return 700 * (exp(x / 1125.0) - 1);
	case 1:
		return 1125 * log(1 + x / 700.0);
	}
	fprintf(stderr, "[Mel_Scale], direction = {-1 (inversed transform), 1 (forward transform)}\n");
	return 0;
}
void MFCC(int length_frame, int length_DFT, int number_coefficients, int number_filterbanks, int sample_rate, double frame[], double feature_vector[]){
	double max_Mels_frequency = Mel_Scale(1, sample_rate / 2);
	double min_Mels_frequency = Mel_Scale(1, 300);
	double interval = (max_Mels_frequency - min_Mels_frequency) / (number_filterbanks + 1);

	double *filterbank = new double[number_filterbanks];
	double *Xr = new double[length_DFT];
	double *Xi = new double[length_DFT];

	for (int i = 0; i < number_filterbanks; i++){
		filterbank[i] = 0;
	}
	for (int i = 0; i < length_DFT; i++){
		Xr[i] = (i < length_frame) ? (frame[i]) : (0);
		Xi[i] = 0;
	}
	FFT(1, length_DFT, Xr, Xi);

	for (int i = 0; i < length_DFT / 2 + 1; i++){
		double frequency = (sample_rate / 2) * i / (length_DFT / 2);
		double Mel_frequency = Mel_Scale(1, frequency);
		double power = (Xr[i] * Xr[i] + Xi[i] * Xi[i]) / length_frame;

		for (int j = 0; j < number_filterbanks; j++){
			double frequency_boundary[] = { min_Mels_frequency + interval * (j + 0), min_Mels_frequency + interval * (j + 1), min_Mels_frequency + interval * (j + 2) };

			if (frequency_boundary[0] <= Mel_frequency && Mel_frequency <= frequency_boundary[1]){
				double lower_frequency = Mel_Scale(-1, frequency_boundary[0]);
				double upper_frequency = Mel_Scale(-1, frequency_boundary[1]);

				filterbank[j] += power * (frequency - lower_frequency) / (upper_frequency - lower_frequency);
			}
			else
			if (frequency_boundary[1] <= Mel_frequency && Mel_frequency <= frequency_boundary[2]){
				double lower_frequency = Mel_Scale(-1, frequency_boundary[1]);
				double upper_frequency = Mel_Scale(-1, frequency_boundary[2]);

				filterbank[j] += power * (upper_frequency - frequency) / (upper_frequency - lower_frequency);
			}
		}
	}

	for (int i = 0; i < number_filterbanks; i++){
		filterbank[i] = log(filterbank[i]);
	}
	DCT(1, number_filterbanks, filterbank);

	for (int i = 0; i < number_coefficients; i++){
		feature_vector[i] = filterbank[i];
	}

	delete[] filterbank;
	delete[] Xr;
	delete[] Xi;
}

vector<vector<double>> findMfcc(float audioData[],int l){
	int stride = 160;
	int length_frame = 400;
	int length_DFT = 512;
	int number_coefficients = 13;
	int number_filterbanks = 26;

	int number_feature_vectors=(l - length_frame) / stride + 1;
	int nSamplesPerSec=8000;

	double pi = 3.14159265358979323846;

	double** feature_vector;
        vector< vector<double>> mfcc_Data;
    //     data_holder.length=number_feature_vectors*number_coefficients*3;
      //  data_holder.data=new double[data_holder.length];
	//printf("size of mfcc %d",sizeof(MFCCData));
	feature_vector = new double*[number_feature_vectors = (l - length_frame) / stride + 1];

	for (int i = 0; i < number_feature_vectors; i++){
		feature_vector[i] = new double[3 * number_coefficients];
	}

	// MFCC
	for (int i = 0; i <= l - length_frame; i += stride){
		double *frame = new double[length_frame];

		// pre-emphasis
		for (int j = 0; j < length_frame; j++){
			if (i + j < l){
				frame[j] = audioData[i + j] - 0.95 * audioData[i + j - 1];
			}
			else{
				frame[j] = 0;
			}
		}

		// windowing
		for (int j = 0; j < length_frame; j++){
			frame[j] *= 0.54 - 0.46 * cos(2 * pi * j / (length_frame - 1));
		}

		MFCC(length_frame, length_DFT, number_coefficients, number_filterbanks, nSamplesPerSec, frame, feature_vector[i / stride]);

		delete[] frame;
	}

	// deltas
	for (int i = 0; i < number_feature_vectors; i++){
		int prev = (i == 0) ? (0) : (i - 1);
		int next = (i == number_feature_vectors - 1) ? (number_feature_vectors - 1) : (i + 1);

		for (int j = 0; j < number_coefficients; j++){
			feature_vector[i][number_coefficients + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
		}
	}

	// delta-deltas
	for (int i = 0; i < number_feature_vectors; i++){
		int prev = (i == 0) ? (0) : (i - 1);
		int next = (i == number_feature_vectors - 1) ? (number_feature_vectors - 1) : (i + 1);

		for (int j = number_coefficients; j < 2 * number_coefficients; j++){
			feature_vector[i][number_coefficients + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
		}
	}

	//FILE *file = fopen("Dr.txt", "w");
       // printf("\n Mfcc Data size is %d * %d \n",sizeof(feature_vector)[1],sizeof(feature_vector)[0]);
        int count=0;
	for (int i = 0; i < number_feature_vectors; i++){
	    // double power=0;
             vector<double> rowdata;
            for (int j = 0; j < 3 * number_coefficients; j++){
		//   fprintf(file, "%lf ", feature_vector[i][j]);
                       // printf("%lf ",feature_vector[i][j]);
                     rowdata.push_back(feature_vector[i][j]);
                 //  power+=pow(feature_vector[i][j],2);
                                               // printf("count=%d, feature_vector %f, MFCCData %f \n",count,feature_vector[i][j],MFCCData[count]);
                        
            }
		
           //fprintf(file, "%lf ",);
	    mfcc_Data.push_back(rowdata);
            rowdata.clear();
       //  fprintf(file, "\n ");
           //     printf("\n ");
	}
	 //printf("MFCC extracted success size=%d" ,count-1);
	//fclose(file);

	/*for (int i = 0; i < number_feature_vectors; i++){
		delete[] feature_vector[i];
	}*/
	delete[] feature_vector;
       

	return mfcc_Data;
        
}
double minimum(double x, double y,double z){
    if(x<y){
        if(x<z){
            return x;
        }
        else{
            return z;
        }
    }
    else{
        if (y<z){
            return y;
        }
        else{
            return z;
        }
    }
}
double DTW(vector<vector<double>> &x,vector<vector<double>> &y){
    long n=x.size();
    long m=y.size();
    //printf("\nDTW sizes of n=%d and m=%d and y(m-2,)=%f",n,m,y[m-2][1]);
    
    vector <vector <double>> dtw(n, vector<double>(m));
     vector <vector <double>> dtw2(n, vector<double>(m));
   
   /* for(int i=1;i<n;i++){
        dtw[i][0]=100000.00;
    }
    for(int j=1;j<m;j++){
        dtw[0][j]=100000.00;
    }
    dtw[0][0]=0.00;
    double cost=0;
     for(int i=1;i<n;i++){
        for(int j=1;j<m;j++){
            cost = sqrt(pow((x[i]-y[j]),2)+pow((i/n-j/m),2));
            dtw[i][j] = cost + minimum(dtw[i-1][j], dtw[i][j-1],dtw[i-1][j-1]);    // match
            //printf("\nlast i=%d and last j=%d",i,j);
    }
    }*/
     for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
              dtw[i][j] = 0.0; 
              dtw2[i][j]=100000000.0;// match
            //printf("\nlast i=%d and last j=%d",i,j);
    }
    }
    
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            double sum=0;
            long o=x[i].size();
            for(int k=0;k<o;k++){
                sum+=pow((x[i][k]-y[j][k]),2);
            }
            dtw[i][j] = sum; 
            // match
            //printf("\nlast i=%d and last j=%d",i,j);
    }
    }
    dtw2[0][0]=dtw[0][0];
     for(int i=1;i<n;i++){
        for(int j=0;j<m;j++){
            double D1,D2,D3;
            D1=dtw2[i-1][j];
            if(j>0){
                D2=dtw2[i-1][j-1];
            }
            else{
                D2=100000000.0;
            }
            if(j>1){
                D3=dtw2[i-1][j-2];
            }
            else{
                D3=100000000.0;
            }
            dtw2[i][j]=dtw[i][j]+minimum(D1,D2,D3);
            
    }
    }
   // 
 /*  printf("\ndistance @(%d,%d) is %lf",0,0,dtw2[0][0]);
 printf("\ndistance bn (%lf) and (%lf) @(%d,%d) is %lf",x[0],y[0],0,0,dtw[0][0]);
   printf("\ndistance @(%d,%d) is %lf",1,0,dtw2[1][0]);
     */      
    return dtw2[n-1][m-1];
    
  //return 0.0;
}
