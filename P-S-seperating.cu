#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include<iostream>
using namespace std;
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<complex>
#include<cufft.h>
#include "cuComplex.h"//cuda核函数复数计算库文件

#define Tn 3000
#define pi 3.141592653	
#define fm 20
#define dt 0.0005
#define dx 5.0
#define dz 5.0
#define N 6
#define pml 100

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        printf("%s\n",cudaGetErrorString( err )); \
        }


__device__ float a[6] = { 1.2213365, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };

void write_bin(float* dst, char* filename, int xnum, int tn, int L) {
	FILE* binfile;
	binfile = fopen(filename, "wb");
	for (int i = L; i < xnum - L; i++)
		for (int k = L; k < tn - L; k++)
			fwrite(&dst[i * tn + k], sizeof(float), 1, binfile);
	fclose(binfile);
}

void write_bin1(float* dst, char* filename, int xnum, int tn, int L) {
	FILE* binfile;
	binfile = fopen(filename, "wb");
	for (int i = L; i < xnum - L; i++)
		for (int k = 0; k < tn; k++)
			fwrite(&dst[i * tn + k], sizeof(float), 1, binfile);
	fclose(binfile);
}
void write_bin2(float* dst, char* filename, int xnum, int tn) {
	FILE* binfile;
	binfile = fopen(filename, "wb");
	for (int i = 0; i < xnum; i++)
		for (int k = 0; k < tn; k++)
			fwrite(&dst[i * tn + k], sizeof(float), 1, binfile);
	fclose(binfile);
}
void read_bin(float* dst, char* filename, int xnum, int tn, int L) {
	FILE* binfile;
	binfile = fopen(filename, "wb");
	for (int i = L; i < xnum - L; i++)
		for (int k = L; k < tn - L; k++)
			fread(&dst[i * tn + k], sizeof(float), 1, binfile);
	fclose(binfile);
}

void addpml(float* vp, float* vs, float* rou, float* vp_pml, float* vs_pml, float* rou_pml, int Xn, int Zn) {
	int i, j;

	for (i = 0; i < Xn; i++) {
		for (j = 0; j < Zn; j++) {
			vp_pml[i * Zn + j] = vp[i * Zn + j];
			vs_pml[i * Zn + j] = vs[i * Zn + j];
			rou_pml[i * Zn + j] = rou[i * Zn + j];
		}
	}
	//hengxiang
	for (i = 0; i < Xn; i++) {
		for (j = 0; j < pml; j++) {
			vp_pml[i * Zn + j] = vp_pml[i * Zn + pml];
			vs_pml[i * Zn + j] = vs_pml[i * Zn + pml];
			rou_pml[i * Zn + j] = rou_pml[i * Zn + pml];
		}
		for (j = Zn - pml; j < Zn; j++) {
			vp_pml[i * Zn + j] = vp_pml[i * Zn + (Zn - pml - 1)];
			vs_pml[i * Zn + j] = vs_pml[i * Zn + (Zn - pml - 1)];
			rou_pml[i * Zn + j] = rou_pml[i * Zn + (Zn - pml - 1)];
		}
	}
	//zongxiang
	for (j = 0; j < Zn; j++) {
		for (i = 0; i < pml; i++) {
			vp_pml[i * Zn + j] = vp_pml[pml * Zn + j];
			vs_pml[i * Zn + j] = vs_pml[pml * Zn + j];
			rou_pml[i * Zn + j] = rou_pml[pml * Zn + j];
		}
		for (i = Xn - pml; i < Xn; i++) {
			vp_pml[i * Zn + j] = vp_pml[(Xn - pml - 1) * Zn + j];
			vs_pml[i * Zn + j] = vs_pml[(Xn - pml - 1) * Zn + j];
			rou_pml[i * Zn + j] = rou_pml[(Xn - pml - 1) * Zn + j];
		}
	}



}
void dumpingfactor(float* vp_pml, float* ddx, float* ddz, int Xn, int Zn, int xn, int zn) {
	float R = pow((float)10, -6);//·ŽÉäÏµÊý

	int i, j, x, z, l;

	//float af = 10e-6, aa = 0.25, b = 0.75;

	float rr = 0.000001;
	for (i = 0; i < Xn; i++)
		for (j = 0; j < Zn; j++)
		{
			if (i < pml)
			{
				l = pml - i;
				ddx[i * Zn + j] = log10(1 / rr) * (5.0 * vp_pml[i * Zn + j] / (2.0 * pml)) * pow(1.0 * l / pml, 4.0);
			}

			if (i > xn + pml)
			{
				l = i - xn - pml;
				ddx[i * Zn + j] = log10(1 / rr) * (5.0 * vp_pml[i * Zn + j] / (2.0 * pml)) * pow(1.0 * l / pml, 4.0);
			}

			if (j < pml)
			{
				l = pml - j;
				ddz[i * Zn + j] = log10(1 / rr) * (5.0 * vp_pml[i * Zn + j] / (2.0 * pml)) * pow(1.0 * l / pml, 4.0);

			}

			if (j > zn + pml)
			{
				l = j - zn - pml;
				ddz[i * Zn + j] = log10(1 / rr) * (5.0 * vp_pml[i * Zn + j] / (2.0 * pml)) * pow(1.0 * l / pml, 4.0);
			}
		}


}

__global__ void forward_u(int Xn, int Zn, float* ux, float* uz, float* ddx, float* ddz, 
	float* upx_next, float* upx_now, float* upx_past, float* upz_next, float* upz_now, float* upz_past, float* usx_next, float* usx_now, float* usx_past, float* usz_next, float* usz_now, float* usz_past,	
	float* recordupx, float* recordupz, float* recordusx, float* recordusz, float* recordux, float* recorduz, 
	float* theta, float* omega, int receiver_depth, int t, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* O_duz_xdz, float* O_duz_zdx, float* O_dux_zdx, float* O_dux_xdz, float* O_thetax, float* O_thetaz, float* O_omegaz, float* O_omegax) {

	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	
	float dux_xdz = 0.0f;
	float duz_zdx = 0.0f;
	float duz_xdz = 0.0f;
	float dux_zdx = 0.0f;
	float dthetadx = 0.0f;
	float dthetadz = 0.0f;
	float domegadx = 0.0f;
	float domegadz = 0.0f;

	//printf("a[m] = %f\n", a[3]);
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{
		dthetadx = (a[0] * (theta[(i + 1) * Zn + j] - theta[(i - 0) * Zn + j])
			+ a[1] * (theta[(i + 2) * Zn + j] - theta[(i - 1) * Zn + j])
			+ a[2] * (theta[(i + 3) * Zn + j] - theta[(i - 2) * Zn + j])
			+ a[3] * (theta[(i + 4) * Zn + j] - theta[(i - 3) * Zn + j])
			+ a[4] * (theta[(i + 5) * Zn + j] - theta[(i - 4) * Zn + j])
			+ a[5] * (theta[(i + 6) * Zn + j] - theta[(i - 5) * Zn + j])) / dx;

		dthetadz = (a[0] * (theta[(i)*Zn + j + 1] - theta[(i)*Zn + j - 0])
			+ a[1] * (theta[(i)*Zn + j + 2] - theta[(i)*Zn + j - 1])
			+ a[2] * (theta[(i)*Zn + j + 3] - theta[(i)*Zn + j - 2])
			+ a[3] * (theta[(i)*Zn + j + 4] - theta[(i)*Zn + j - 3])
			+ a[4] * (theta[(i)*Zn + j + 5] - theta[(i)*Zn + j - 4])
			+ a[5] * (theta[(i)*Zn + j + 6] - theta[(i)*Zn + j - 5])) / dz;

		domegadx = (a[0] * (omega[(i + 0) * Zn + j] - omega[(i - 1) * Zn + j])
			+ a[1] * (omega[(i + 1) * Zn + j] - omega[(i - 2) * Zn + j])
			+ a[2] * (omega[(i + 2) * Zn + j] - omega[(i - 3) * Zn + j])
			+ a[3] * (omega[(i + 3) * Zn + j] - omega[(i - 4) * Zn + j])
			+ a[4] * (omega[(i + 4) * Zn + j] - omega[(i - 5) * Zn + j])
			+ a[5] * (omega[(i + 5) * Zn + j] - omega[(i - 6) * Zn + j])) / dx;

		domegadz = (a[0] * (omega[(i)*Zn + j + 0] - omega[(i)*Zn + j - 1])
			+ a[1] * (omega[(i)*Zn + j + 1] - omega[(i)*Zn + j - 2])
			+ a[2] * (omega[(i)*Zn + j + 2] - omega[(i)*Zn + j - 3])
			+ a[3] * (omega[(i)*Zn + j + 3] - omega[(i)*Zn + j - 4])
			+ a[4] * (omega[(i)*Zn + j + 4] - omega[(i)*Zn + j - 5])
			+ a[5] * (omega[(i)*Zn + j + 5] - omega[(i)*Zn + j - 6])) / dz;


		dux_xdz = (a[0] * (duxdx[(i)*Zn + j + 1] - duxdx[(i)*Zn + j - 0])
			+ a[1] * (duxdx[(i)*Zn + j + 2] - duxdx[(i)*Zn + j - 1])
			+ a[2] * (duxdx[(i)*Zn + j + 3] - duxdx[(i)*Zn + j - 2])
			+ a[3] * (duxdx[(i)*Zn + j + 4] - duxdx[(i)*Zn + j - 3])
			+ a[4] * (duxdx[(i)*Zn + j + 5] - duxdx[(i)*Zn + j - 4])
			+ a[5] * (duxdx[(i)*Zn + j + 6] - duxdx[(i)*Zn + j - 5])) / dz;

		dux_zdx = (a[0] * (duxdz[(i + 0) * Zn + j] - duxdz[(i - 1) * Zn + j])
			+ a[1] * (duxdz[(i + 1) * Zn + j] - duxdz[(i - 2) * Zn + j])
			+ a[2] * (duxdz[(i + 2) * Zn + j] - duxdz[(i - 3) * Zn + j])
			+ a[3] * (duxdz[(i + 3) * Zn + j] - duxdz[(i - 4) * Zn + j])
			+ a[4] * (duxdz[(i + 4) * Zn + j] - duxdz[(i - 5) * Zn + j])
			+ a[5] * (duxdz[(i + 5) * Zn + j] - duxdz[(i - 6) * Zn + j])) / dx;

		duz_xdz = (a[0] * (duzdx[(i)*Zn + j + 0] - duzdx[(i)*Zn + j - 1])
			+ a[1] * (duzdx[(i)*Zn + j + 1] - duzdx[(i)*Zn + j - 2])
			+ a[2] * (duzdx[(i)*Zn + j + 2] - duzdx[(i)*Zn + j - 3])
			+ a[3] * (duzdx[(i)*Zn + j + 3] - duzdx[(i)*Zn + j - 4])
			+ a[4] * (duzdx[(i)*Zn + j + 4] - duzdx[(i)*Zn + j - 5])
			+ a[5] * (duzdx[(i)*Zn + j + 5] - duzdx[(i)*Zn + j - 6])) / dz;

		duz_zdx = (a[0] * (duzdz[(i + 1) * Zn + j] - duzdz[(i - 0) * Zn + j])
			+ a[1] * (duzdz[(i + 2) * Zn + j] - duzdz[(i - 1) * Zn + j])
			+ a[2] * (duzdz[(i + 3) * Zn + j] - duzdz[(i - 2) * Zn + j])
			+ a[3] * (duzdz[(i + 4) * Zn + j] - duzdz[(i - 3) * Zn + j])
			+ a[4] * (duzdz[(i + 5) * Zn + j] - duzdz[(i - 4) * Zn + j])
			+ a[5] * (duzdz[(i + 6) * Zn + j] - duzdz[(i - 5) * Zn + j])) / dx;

		O_duz_xdz[i * Zn + j] = O_duz_xdz[i * Zn + j] + (ddz[i * Zn + j] * duz_xdz - ddz[i * Zn + j] * O_duz_xdz[i * Zn + j]) * dt;
		O_duz_zdx[i * Zn + j] = O_duz_zdx[i * Zn + j] + (ddx[i * Zn + j] * duz_zdx - ddx[i * Zn + j] * O_duz_zdx[i * Zn + j]) * dt;
		O_dux_zdx[i * Zn + j] = O_dux_zdx[i * Zn + j] + (ddx[i * Zn + j] * dux_zdx - ddx[i * Zn + j] * O_dux_zdx[i * Zn + j]) * dt;
		O_dux_xdz[i * Zn + j] = O_dux_xdz[i * Zn + j] + (ddz[i * Zn + j] * dux_xdz - ddz[i * Zn + j] * O_dux_xdz[i * Zn + j]) * dt;

		O_thetax[i * Zn + j] = O_thetax[i * Zn + j] + (ddx[i * Zn + j] * dthetadx - ddx[i * Zn + j] * O_thetax[i * Zn + j]) * dt;
		O_thetaz[i * Zn + j] = O_thetaz[i * Zn + j] + (ddz[i * Zn + j] * dthetadz - ddz[i * Zn + j] * O_thetaz[i * Zn + j]) * dt;
		O_omegaz[i * Zn + j] = O_omegaz[i * Zn + j] + (ddz[i * Zn + j] * domegadz - ddz[i * Zn + j] * O_omegaz[i * Zn + j]) * dt;
		O_omegax[i * Zn + j] = O_omegax[i * Zn + j] + (ddx[i * Zn + j] * domegadx - ddx[i * Zn + j] * O_omegax[i * Zn + j]) * dt;

		upx_next[i * Zn + j] = 2 * upx_now[i * Zn + j] - upx_past[i * Zn + j] + (dt * dt) * (dthetadx - O_thetax[i * Zn + j] + duz_xdz - duz_zdx - O_duz_xdz[i * Zn + j] + O_duz_zdx[i * Zn + j]);
		upz_next[i * Zn + j] = 2 * upz_now[i * Zn + j] - upz_past[i * Zn + j] + (dt * dt) * (dthetadz - O_thetaz[i * Zn + j] + dux_zdx - dux_xdz - O_dux_zdx[i * Zn + j] + O_dux_xdz[i * Zn + j]);
		usx_next[i * Zn + j] = 2 * usx_now[i * Zn + j] - usx_past[i * Zn + j] + (dt * dt) * (domegadz - O_omegaz[i * Zn + j] + duz_xdz - duz_zdx - O_duz_xdz[i * Zn + j] + O_duz_zdx[i * Zn + j]);
		usz_next[i * Zn + j] = 2 * usz_now[i * Zn + j] - usz_past[i * Zn + j] + (dt * dt) * (-domegadx + O_omegax[i * Zn + j] + dux_zdx - dux_xdz - O_dux_zdx[i * Zn + j] + O_dux_xdz[i * Zn + j]);

		if (t < Tn - 1)
		{
			upx_past[i * Zn + j] = upx_now[i * Zn + j];	upx_now[i * Zn + j] = upx_next[i * Zn + j];
			upz_past[i * Zn + j] = upz_now[i * Zn + j];	upz_now[i * Zn + j] = upz_next[i * Zn + j];
			usx_past[i * Zn + j] = usx_now[i * Zn + j];	usx_now[i * Zn + j] = usx_next[i * Zn + j];
			usz_past[i * Zn + j] = usz_now[i * Zn + j];	usz_now[i * Zn + j] = usz_next[i * Zn + j];
		}
		


		ux[i * Zn + j] = upx_next[i * Zn + j] + usx_next[i * Zn + j];
		uz[i * Zn + j] = upz_next[i * Zn + j] + usz_next[i * Zn + j];
	}
	if (j = receiver_depth)
	{
		recordupx[i * Tn + t] = upx_next[i * Zn + j];
		recordupz[i * Tn + t] = upz_next[i * Zn + j];
		recordusx[i * Tn + t] = usx_next[i * Zn + j];
		recordusz[i * Tn + t] = usz_next[i * Zn + j];
		recordux[i * Tn + t] = ux[i * Zn + j];
		recorduz[i * Tn + t] = uz[i * Zn + j];
		
	}

	
}

__global__ void forward_s(int Xn, int Zn, float* vp, float* vs, float* ux, float* uz, float* ddx, float* ddz, float* theta, float* omega,
	int t, int shotx, int shotz, float* source,float* F_xx, float* F_zz, float* F_xz, float* F_zx,
	float* duxdz, float* duzdz, float* duxdx, float* duzdx) {
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	
	float dux_dx = 0.0f;
	float duz_dz = 0.0f;
	float dux_dz = 0.0f;
	float duz_dx = 0.0f;
	
	float s;
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{
		if (i == shotx && j == shotz)
		{
			s = source[t];
		}
		else
		{
			s = 0.0;
		}

		dux_dx = (a[0] * (ux[(i + 0) * Zn + j] - ux[(i - 1) * Zn + j])
			+ a[1] * (ux[(i + 1) * Zn + j] - ux[(i - 2) * Zn + j])
			+ a[2] * (ux[(i + 2) * Zn + j] - ux[(i - 3) * Zn + j])
			+ a[3] * (ux[(i + 3) * Zn + j] - ux[(i - 4) * Zn + j])
			+ a[4] * (ux[(i + 4) * Zn + j] - ux[(i - 5) * Zn + j])
			+ a[5] * (ux[(i + 5) * Zn + j] - ux[(i - 6) * Zn + j])) / dx;

		dux_dz = (a[0] * (ux[(i)*Zn + j + 1] - ux[(i)*Zn + j - 0])
			+ a[1] * (ux[(i)*Zn + j + 2] - ux[(i)*Zn + j - 1])
			+ a[2] * (ux[(i)*Zn + j + 3] - ux[(i)*Zn + j - 2])
			+ a[3] * (ux[(i)*Zn + j + 4] - ux[(i)*Zn + j - 3])
			+ a[4] * (ux[(i)*Zn + j + 5] - ux[(i)*Zn + j - 4])
			+ a[5] * (ux[(i)*Zn + j + 6] - ux[(i)*Zn + j - 5])) / dz;

		duz_dz = (a[0] * (uz[(i)*Zn + j + 0] - uz[(i)*Zn + j - 1])
			+ a[1] * (uz[(i)*Zn + j + 1] - uz[(i)*Zn + j - 2])
			+ a[2] * (uz[(i)*Zn + j + 2] - uz[(i)*Zn + j - 3])
			+ a[3] * (uz[(i)*Zn + j + 3] - uz[(i)*Zn + j - 4])
			+ a[4] * (uz[(i)*Zn + j + 4] - uz[(i)*Zn + j - 5])
			+ a[5] * (uz[(i)*Zn + j + 5] - uz[(i)*Zn + j - 6])) / dz;

		duz_dx = (a[0] * (uz[(i + 1) * Zn + j] - uz[(i - 0) * Zn + j])
			+ a[1] * (uz[(i + 2) * Zn + j] - uz[(i - 1) * Zn + j])
			+ a[2] * (uz[(i + 3) * Zn + j] - uz[(i - 2) * Zn + j])
			+ a[3] * (uz[(i + 4) * Zn + j] - uz[(i - 3) * Zn + j])
			+ a[4] * (uz[(i + 5) * Zn + j] - uz[(i - 4) * Zn + j])
			+ a[5] * (uz[(i + 6) * Zn + j] - uz[(i - 5) * Zn + j])) / dx;

		F_xx[i * Zn + j] = F_xx[i * Zn + j] + (ddx[i * Zn + j] * dux_dx - ddx[i * Zn + j] * F_xx[i * Zn + j]) * dt;
		F_zz[i * Zn + j] = F_zz[i * Zn + j] + (ddz[i * Zn + j] * duz_dz - ddz[i * Zn + j] * F_zz[i * Zn + j]) * dt;
		F_xz[i * Zn + j] = F_xz[i * Zn + j] + (ddz[i * Zn + j] * dux_dz - ddz[i * Zn + j] * F_xz[i * Zn + j]) * dt;
		F_zx[i * Zn + j] = F_zx[i * Zn + j] + (ddx[i * Zn + j] * duz_dx - ddx[i * Zn + j] * F_zx[i * Zn + j]) * dt;

		theta[i * Zn + j] = vp[i * Zn + j] * vp[i * Zn + j] * (dux_dx + duz_dz - F_xx[i * Zn + j] - F_zz[i * Zn + j]);
		omega[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dz - duz_dx - F_xz[i * Zn + j] + F_zx[i * Zn + j]);
		duzdx[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (duz_dx - F_zx[i * Zn + j]);
		duzdz[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (duz_dz - F_zz[i * Zn + j]);
		duxdz[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dz - F_xz[i * Zn + j]);
		duxdx[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dx - F_xx[i * Zn + j]);

		theta[i * Zn + j] = theta[i * Zn + j] + s;

	}

	
}
__global__ void save_wavefiled(int Xn,int Zn,int L,float* d_ux_up, float* d_uz_up, float* d_theta_up, float* d_omega_up, float* d_duzdx_up, float* d_duzdz_up, float* d_duxdx_up, float* d_duxdz_up,
	float* d_ux_dn, float* d_uz_dn, float* d_theta_dn, float* d_omega_dn, float* d_duzdx_dn, float* d_duzdz_dn, float* d_duxdx_dn, float* d_duxdz_dn,
	float* d_ux_lf, float* d_uz_lf, float* d_theta_lf, float* d_omega_lf, float* d_duzdx_lf, float* d_duzdz_lf, float* d_duxdx_lf, float* d_duxdz_lf,
	float* d_ux_rt, float* d_uz_rt, float* d_theta_rt, float* d_omega_rt, float* d_duzdx_rt, float* d_duzdz_rt, float* d_duxdx_rt, float* d_duxdz_rt,
	float* d_ux, float* d_uz, float* d_theta, float* d_omega, float* d_duzdx, float* d_duzdz, float* d_duxdx, float* d_duxdz, int t,
	float* d_upx_next_up, float* d_upz_next_up, float* d_usx_next_up, float* d_usz_next_up, float* d_upx_now_up, float* d_upz_now_up, float* d_usx_now_up, float* d_usz_now_up,
	float* d_upx_next_dn, float* d_upz_next_dn, float* d_usx_next_dn, float* d_usz_next_dn, float* d_upx_now_dn, float* d_upz_now_dn, float* d_usx_now_dn, float* d_usz_now_dn,
	float* d_upx_next_lf, float* d_upz_next_lf, float* d_usx_next_lf, float* d_usz_next_lf, float* d_upx_now_lf, float* d_upz_now_lf, float* d_usx_now_lf, float* d_usz_now_lf,
	float* d_upx_next_rt, float* d_upz_next_rt, float* d_usx_next_rt, float* d_usz_next_rt, float* d_upx_now_rt, float* d_upz_now_rt, float* d_usx_now_rt, float* d_usz_now_rt,
	float* d_upx_next, float* d_upz_next, float* d_usx_next, float* d_usz_next, float* d_upx_now, float* d_upz_now, float* d_usx_now, float* d_usz_now)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= 0 && i < Xn && j >= L - N && j < L + N)
	{
		d_ux_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_ux[i * Zn + j];
		d_uz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_uz[i * Zn + j];
		d_theta_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_theta[i * Zn + j];
		d_omega_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_omega[i * Zn + j];
		d_duzdx_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_duzdx[i * Zn + j];
		d_duzdz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_duzdz[i * Zn + j];
		d_duxdx_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_duxdx[i * Zn + j];
		d_duxdz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_duxdz[i * Zn + j];
		d_upx_next_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_upx_next[i * Zn + j];
		d_upz_next_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_upz_next[i * Zn + j];
		d_usx_next_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_usx_next[i * Zn + j];
		d_usz_next_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_usz_next[i * Zn + j];
		d_upx_now_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_upx_now[i * Zn + j];
		d_upz_now_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_upz_now[i * Zn + j];
		d_usx_now_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_usx_now[i * Zn + j];
		d_usz_now_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_usz_now[i * Zn + j];
	}
	if (i >= 0 && i < Xn && j >= Zn - L - N && j < Zn - L + N)
	{
		d_ux_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_ux[i * Zn + j];
		d_uz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_uz[i * Zn + j];
		d_theta_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_theta[i * Zn + j];
		d_omega_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_omega[i * Zn + j];
		d_duzdx_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_duzdx[i * Zn + j];
		d_duzdz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_duzdz[i * Zn + j];
		d_duxdx_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_duxdx[i * Zn + j];
		d_duxdz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_duxdz[i * Zn + j];
		d_upx_next_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_upx_next[i * Zn + j];
		d_upz_next_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_upz_next[i * Zn + j];
		d_usx_next_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_usx_next[i * Zn + j];
		d_usz_next_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_usz_next[i * Zn + j];
		d_upx_now_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_upx_now[i * Zn + j];
		d_upz_now_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_upz_now[i * Zn + j];
		d_usx_now_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_usx_now[i * Zn + j];
		d_usz_now_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_usz_now[i * Zn + j];
	}
	if (i >= L - N && i < L + N && j >= 0 && j < Zn)
	{
		d_ux_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_ux[i * Zn + j];
		d_uz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_uz[i * Zn + j];
		d_theta_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_theta[i * Zn + j];
		d_omega_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_omega[i * Zn + j];
		d_duzdx_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_duzdx[i * Zn + j];
		d_duzdz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_duzdz[i * Zn + j];
		d_duxdx_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_duxdx[i * Zn + j];
		d_duxdz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_duxdz[i * Zn + j];
		d_upx_next_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_upx_next[i * Zn + j];
		d_upz_next_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_upz_next[i * Zn + j];
		d_usx_next_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_usx_next[i * Zn + j];
		d_usz_next_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_usz_next[i * Zn + j];
		d_upx_now_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_upx_now[i * Zn + j];
		d_upz_now_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_upz_now[i * Zn + j];
		d_usx_now_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_usx_now[i * Zn + j];
		d_usz_now_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_usz_now[i * Zn + j];

	}
	if (i >= Xn - L - N && i < Xn - L + N && j >= 0 && j < Zn)
	{
		d_ux_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_ux[i * Zn + j];
		d_uz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_uz[i * Zn + j];
		d_theta_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_theta[i * Zn + j];
		d_omega_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_omega[i * Zn + j];
		d_duzdx_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_duzdx[i * Zn + j];
		d_duzdz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_duzdz[i * Zn + j];
		d_duxdx_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_duxdx[i * Zn + j];
		d_duxdz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_duxdz[i * Zn + j];
		d_upx_next_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_upx_next[i * Zn + j];
		d_upz_next_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_upz_next[i * Zn + j];
		d_usx_next_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_usx_next[i * Zn + j];
		d_usz_next_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_usz_next[i * Zn + j];
		d_upx_now_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_upx_now[i * Zn + j];
		d_upz_now_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_upz_now[i * Zn + j];
		d_usx_now_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_usx_now[i * Zn + j];
		d_usz_now_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_usz_now[i * Zn + j];

	}

}
__global__ void read_last_wavefiled(int Xn, int Zn, int L,float* d_theta_s, float* d_omega_s, float* d_ux_s, float* d_uz_s, float* d_duzdx_s, float* d_duzdz_s, float* d_duxdz_s, float* d_duxdx_s,
	float* d_upx_next_s, float* d_upz_next_s, float* d_usx_next_s, float* d_usz_next_s,
	float* d_theta, float* d_omega, float* d_ux, float* d_uz, float* d_duzdx, float* d_duzdz, float* d_duxdz, float* d_duxdx,
	float* d_upx_next, float* d_upz_next, float* d_usx_next, float* d_usz_next, 
	float* d_upx_now, float* d_upz_now, float* d_usx_now, float* d_usz_now,
	float* d_upx_now_s, float* d_upz_now_s, float* d_usx_now_s, float* d_usz_now_s)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L - N && i < Xn - L + N && j >= L - N && j < Zn - L + N)
	{
		d_theta_s[i * Zn + j] = d_theta[i * Zn + j];
		d_omega_s[i * Zn + j] = d_omega[i * Zn + j];
		d_ux_s[i * Zn + j] = d_ux[i * Zn + j];
		d_uz_s[i * Zn + j] = d_uz[i * Zn + j];

		d_upx_next_s[i * Zn + j] = d_upx_next[i * Zn + j];
		d_upz_next_s[i * Zn + j] = d_upz_next[i * Zn + j];
		d_usx_next_s[i * Zn + j] = d_usx_next[i * Zn + j];
		d_usz_next_s[i * Zn + j] = d_usz_next[i * Zn + j];
		d_duzdx_s[i * Zn + j] = d_duzdx[i * Zn + j];
		d_duzdz_s[i * Zn + j] = d_duzdz[i * Zn + j];
		d_duxdx_s[i * Zn + j] = d_duxdx[i * Zn + j];
		d_duxdz_s[i * Zn + j] = d_duxdz[i * Zn + j];
		d_upx_now_s[i * Zn + j] = d_upx_now[i * Zn + j];
		d_upz_now_s[i * Zn + j] = d_upz_now[i * Zn + j];
		d_usx_now_s[i * Zn + j] = d_usx_now[i * Zn + j];
		d_usz_now_s[i * Zn + j] = d_usz_now[i * Zn + j];
	}
}
__global__ void read_wavefiled_NT1(int Xn, int Zn, int L, 	float* d_upx_past_s, float* d_upz_past_s, float* d_usx_past_s, float* d_usz_past_s,
	float* d_upx_next, float* d_upz_next, float* d_usx_next, float* d_usz_next)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L - N && i < Xn - L + N && j >= L - N && j < Zn - L + N)
	{
		
		d_upx_past_s[i * Zn + j] = d_upx_next[i * Zn + j];
		d_upz_past_s[i * Zn + j] = d_upz_next[i * Zn + j];
		d_usx_past_s[i * Zn + j] = d_usx_next[i * Zn + j];
		d_usz_past_s[i * Zn + j] = d_usz_next[i * Zn + j];
		
	}
}
__global__ void read_wavefiled_NT2(int Xn, int Zn, int L, 
	float* d_upx_past_s, float* d_upz_past_s, float* d_usx_past_s, float* d_usz_past_s,
	float* d_upx_now, float* d_upz_now, float* d_usx_now, float* d_usz_now)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L - N && i < Xn - L + N && j >= L - N && j < Zn - L + N)
	{		
		d_upx_past_s[i * Zn + j] = d_upx_now[i * Zn + j];
		d_upz_past_s[i * Zn + j] = d_upz_now[i * Zn + j];
		d_usx_past_s[i * Zn + j] = d_usx_now[i * Zn + j];
		d_usz_past_s[i * Zn + j] = d_usz_now[i * Zn + j];
	}
}
__global__ void read_wavefiled_NT3(int Xn, int Zn, int L, 
	float* d_upx_next_s, float* d_upz_next_s, float* d_usx_next_s, float* d_usz_next_s,	
	float* d_upx_next, float* d_upz_next, float* d_usx_next, float* d_usz_next,
	float* d_upx_now, float* d_upz_now, float* d_usx_now, float* d_usz_now,
	float* d_upx_now_s, float* d_upz_now_s, float* d_usx_now_s, float* d_usz_now_s, 
	float* d_upx_past, float* d_upz_past, float* d_usx_past, float* d_usz_past,
	float* d_upx_past_s, float* d_upz_past_s, float* d_usx_past_s, float* d_usz_past_s)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L - N && i < Xn - L + N && j >= L - N && j < Zn - L + N)
	{
		
		d_upx_next_s[i * Zn + j] = d_upx_next[i * Zn + j];
		d_upz_next_s[i * Zn + j] = d_upz_next[i * Zn + j];
		d_usx_next_s[i * Zn + j] = d_usx_next[i * Zn + j];
		d_usz_next_s[i * Zn + j] = d_usz_next[i * Zn + j];
	
		d_upx_now_s[i * Zn + j] = d_upx_now[i * Zn + j];
		d_upz_now_s[i * Zn + j] = d_upz_now[i * Zn + j];
		d_usx_now_s[i * Zn + j] = d_usx_now[i * Zn + j];
		d_usz_now_s[i * Zn + j] = d_usz_now[i * Zn + j];

		d_upx_past_s[i * Zn + j] = d_upx_past[i * Zn + j];
		d_upz_past_s[i * Zn + j] = d_upz_past[i * Zn + j];
		d_usx_past_s[i * Zn + j] = d_usx_past[i * Zn + j];
		d_usz_past_s[i * Zn + j] = d_usz_past[i * Zn + j];
	}
}

__global__ void remove(int Xn, int Zn, int Sx, int Sz, int Z_receive, int t0, float dh, float* v, float* record_vx, float* record_vz)
{
	int  t;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float distance;
	if (i >= 0 && i < Xn && j == Z_receive)
	{
		for (t = 0; t < Tn; t++)
		{
			distance = sqrtf(float(abs(Sx - i) * abs(Sx - i) + abs(Z_receive - Sz) * abs(Z_receive - Sz)));
			if (t < (2 * t0 + distance * dh * 1.0 / (dt * v[Sx * Zn + j])))
			{
				record_vx[i * Tn + t] = 0;
				record_vz[i * Tn + t] = 0;

			}


		}

	}
}
__global__ void load_record(int Xn, int Zn,int L, int reciver, float* vx, float* vz, float* record_vx, float* record_vz, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L && i < Xn - L && j == reciver)
	{
		vx[i * Zn + j] = record_vx[i * Tn + t];
		vz[i * Zn + j] = record_vz[i * Tn + t];

	}
}
__global__ void reshot_u(int Xn, int Zn, int L, float* ux, float* uz, float* upx_next, float* upz_next, float* usx_next, float* usz_next, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* upx_now, float* upz_now, float* usx_now, float* usz_now,
	float* upx_past, float* upz_past, float* usx_past, float* usz_past) {
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dux_xdz = 0.0f;
	float duz_zdx = 0.0f;
	float duz_xdz = 0.0f;
	float dux_zdx = 0.0f;
	float dthetadx = 0.0f;
	float dthetadz = 0.0f;
	float domegadx = 0.0f;
	float domegadz = 0.0f;
	
	if (i >= L+N && i < Xn - L-N && j >= L+N && j < Zn - L-N)
	{
		dthetadx = (a[0] * (theta[(i + 1) * Zn + j] - theta[(i - 0) * Zn + j])
			+ a[1] * (theta[(i + 2) * Zn + j] - theta[(i - 1) * Zn + j])
			+ a[2] * (theta[(i + 3) * Zn + j] - theta[(i - 2) * Zn + j])
			+ a[3] * (theta[(i + 4) * Zn + j] - theta[(i - 3) * Zn + j])
			+ a[4] * (theta[(i + 5) * Zn + j] - theta[(i - 4) * Zn + j])
			+ a[5] * (theta[(i + 6) * Zn + j] - theta[(i - 5) * Zn + j])) / dx;

		dthetadz = (a[0] * (theta[(i)*Zn + j + 1] - theta[(i)*Zn + j - 0])
			+ a[1] * (theta[(i)*Zn + j + 2] - theta[(i)*Zn + j - 1])
			+ a[2] * (theta[(i)*Zn + j + 3] - theta[(i)*Zn + j - 2])
			+ a[3] * (theta[(i)*Zn + j + 4] - theta[(i)*Zn + j - 3])
			+ a[4] * (theta[(i)*Zn + j + 5] - theta[(i)*Zn + j - 4])
			+ a[5] * (theta[(i)*Zn + j + 6] - theta[(i)*Zn + j - 5])) / dz;

		domegadx = (a[0] * (omega[(i + 0) * Zn + j] - omega[(i - 1) * Zn + j])
			+ a[1] * (omega[(i + 1) * Zn + j] - omega[(i - 2) * Zn + j])
			+ a[2] * (omega[(i + 2) * Zn + j] - omega[(i - 3) * Zn + j])
			+ a[3] * (omega[(i + 3) * Zn + j] - omega[(i - 4) * Zn + j])
			+ a[4] * (omega[(i + 4) * Zn + j] - omega[(i - 5) * Zn + j])
			+ a[5] * (omega[(i + 5) * Zn + j] - omega[(i - 6) * Zn + j])) / dx;

		domegadz = (a[0] * (omega[(i)*Zn + j + 0] - omega[(i)*Zn + j - 1])
			+ a[1] * (omega[(i)*Zn + j + 1] - omega[(i)*Zn + j - 2])
			+ a[2] * (omega[(i)*Zn + j + 2] - omega[(i)*Zn + j - 3])
			+ a[3] * (omega[(i)*Zn + j + 3] - omega[(i)*Zn + j - 4])
			+ a[4] * (omega[(i)*Zn + j + 4] - omega[(i)*Zn + j - 5])
			+ a[5] * (omega[(i)*Zn + j + 5] - omega[(i)*Zn + j - 6])) / dz;


		dux_xdz = (a[0] * (duxdx[(i)*Zn + j + 1] - duxdx[(i)*Zn + j - 0])
			+ a[1] * (duxdx[(i)*Zn + j + 2] - duxdx[(i)*Zn + j - 1])
			+ a[2] * (duxdx[(i)*Zn + j + 3] - duxdx[(i)*Zn + j - 2])
			+ a[3] * (duxdx[(i)*Zn + j + 4] - duxdx[(i)*Zn + j - 3])
			+ a[4] * (duxdx[(i)*Zn + j + 5] - duxdx[(i)*Zn + j - 4])
			+ a[5] * (duxdx[(i)*Zn + j + 6] - duxdx[(i)*Zn + j - 5])) / dz;

		dux_zdx = (a[0] * (duxdz[(i + 0) * Zn + j] - duxdz[(i - 1) * Zn + j])
			+ a[1] * (duxdz[(i + 1) * Zn + j] - duxdz[(i - 2) * Zn + j])
			+ a[2] * (duxdz[(i + 2) * Zn + j] - duxdz[(i - 3) * Zn + j])
			+ a[3] * (duxdz[(i + 3) * Zn + j] - duxdz[(i - 4) * Zn + j])
			+ a[4] * (duxdz[(i + 4) * Zn + j] - duxdz[(i - 5) * Zn + j])
			+ a[5] * (duxdz[(i + 5) * Zn + j] - duxdz[(i - 6) * Zn + j])) / dx;

		duz_xdz = (a[0] * (duzdx[(i)*Zn + j + 0] - duzdx[(i)*Zn + j - 1])
			+ a[1] * (duzdx[(i)*Zn + j + 1] - duzdx[(i)*Zn + j - 2])
			+ a[2] * (duzdx[(i)*Zn + j + 2] - duzdx[(i)*Zn + j - 3])
			+ a[3] * (duzdx[(i)*Zn + j + 3] - duzdx[(i)*Zn + j - 4])
			+ a[4] * (duzdx[(i)*Zn + j + 4] - duzdx[(i)*Zn + j - 5])
			+ a[5] * (duzdx[(i)*Zn + j + 5] - duzdx[(i)*Zn + j - 6])) / dz;

		duz_zdx = (a[0] * (duzdz[(i + 1) * Zn + j] - duzdz[(i - 0) * Zn + j])
			+ a[1] * (duzdz[(i + 2) * Zn + j] - duzdz[(i - 1) * Zn + j])
			+ a[2] * (duzdz[(i + 3) * Zn + j] - duzdz[(i - 2) * Zn + j])
			+ a[3] * (duzdz[(i + 4) * Zn + j] - duzdz[(i - 3) * Zn + j])
			+ a[4] * (duzdz[(i + 5) * Zn + j] - duzdz[(i - 4) * Zn + j])
			+ a[5] * (duzdz[(i + 6) * Zn + j] - duzdz[(i - 5) * Zn + j])) / dx;

		upx_next[i * Zn + j] = upx_now[i * Zn + j]; upx_now[i * Zn + j] = upx_past[i * Zn + j];
		upz_next[i * Zn + j] = upz_now[i * Zn + j]; upz_now[i * Zn + j] = upz_past[i * Zn + j];
		usx_next[i * Zn + j] = usx_now[i * Zn + j]; usx_now[i * Zn + j] = usx_past[i * Zn + j];
		usz_next[i * Zn + j] = usz_now[i * Zn + j]; usz_now[i * Zn + j] = usz_past[i * Zn + j];



		upx_past[i * Zn + j] = 2 * upx_now[i * Zn + j] - upx_next[i * Zn + j] + (dt * dt) * (dthetadx + duz_xdz - duz_zdx);
		upz_past[i * Zn + j] = 2 * upz_now[i * Zn + j] - upz_next[i * Zn + j] + (dt * dt) * (dthetadz + dux_zdx - dux_xdz);
		usx_past[i * Zn + j] = 2 * usx_now[i * Zn + j] - usx_next[i * Zn + j] + (dt * dt) * (domegadz + duz_xdz - duz_zdx);
		usz_past[i * Zn + j] = 2 * usz_now[i * Zn + j] - usz_next[i * Zn + j] + (dt * dt) * (-domegadx + dux_zdx - dux_xdz);



		ux[i * Zn + j] = upx_past[i * Zn + j] + usx_past[i * Zn + j];
		uz[i * Zn + j] = upz_past[i * Zn + j] + usz_past[i * Zn + j];

	}

}


__global__ void reshot_s(int Xn, int Zn, int L, float* ux, float* uz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* vp, float* vs) {

	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dux_dx = 0.0f;
	float duz_dz = 0.0f;
	float dux_dz = 0.0f;
	float duz_dx = 0.0f;
	
	float s;
	if (i >= L + N && i < Xn - L - N && j >= L + N && j < Zn - L - N)
	{


		dux_dx = (a[0] * (ux[(i + 0) * Zn + j] - ux[(i - 1) * Zn + j])
			+ a[1] * (ux[(i + 1) * Zn + j] - ux[(i - 2) * Zn + j])
			+ a[2] * (ux[(i + 2) * Zn + j] - ux[(i - 3) * Zn + j])
			+ a[3] * (ux[(i + 3) * Zn + j] - ux[(i - 4) * Zn + j])
			+ a[4] * (ux[(i + 4) * Zn + j] - ux[(i - 5) * Zn + j])
			+ a[5] * (ux[(i + 5) * Zn + j] - ux[(i - 6) * Zn + j])) / dx;

		dux_dz = (a[0] * (ux[(i)*Zn + j + 1] - ux[(i)*Zn + j - 0])
			+ a[1] * (ux[(i)*Zn + j + 2] - ux[(i)*Zn + j - 1])
			+ a[2] * (ux[(i)*Zn + j + 3] - ux[(i)*Zn + j - 2])
			+ a[3] * (ux[(i)*Zn + j + 4] - ux[(i)*Zn + j - 3])
			+ a[4] * (ux[(i)*Zn + j + 5] - ux[(i)*Zn + j - 4])
			+ a[5] * (ux[(i)*Zn + j + 6] - ux[(i)*Zn + j - 5])) / dz;

		duz_dz = (a[0] * (uz[(i)*Zn + j + 0] - uz[(i)*Zn + j - 1])
			+ a[1] * (uz[(i)*Zn + j + 1] - uz[(i)*Zn + j - 2])
			+ a[2] * (uz[(i)*Zn + j + 2] - uz[(i)*Zn + j - 3])
			+ a[3] * (uz[(i)*Zn + j + 3] - uz[(i)*Zn + j - 4])
			+ a[4] * (uz[(i)*Zn + j + 4] - uz[(i)*Zn + j - 5])
			+ a[5] * (uz[(i)*Zn + j + 5] - uz[(i)*Zn + j - 6])) / dz;

		duz_dx = (a[0] * (uz[(i + 1) * Zn + j] - uz[(i - 0) * Zn + j])
			+ a[1] * (uz[(i + 2) * Zn + j] - uz[(i - 1) * Zn + j])
			+ a[2] * (uz[(i + 3) * Zn + j] - uz[(i - 2) * Zn + j])
			+ a[3] * (uz[(i + 4) * Zn + j] - uz[(i - 3) * Zn + j])
			+ a[4] * (uz[(i + 5) * Zn + j] - uz[(i - 4) * Zn + j])
			+ a[5] * (uz[(i + 6) * Zn + j] - uz[(i - 5) * Zn + j])) / dx;



		theta[i * Zn + j] = vp[i * Zn + j] * vp[i * Zn + j] * (dux_dx + duz_dz);
		omega[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dz - duz_dx);
		duzdx[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (duz_dx);
		duzdz[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (duz_dz);
		duxdz[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dz);
		duxdx[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dx);



	}

}
__global__ void read_wavefiled1(int t,int L, int Xn, int Zn,
	float* d_ux_up, float* d_uz_up,float* d_ux_dn, float* d_uz_dn,float* d_ux_lf, float* d_uz_lf, float* d_ux_rt, float* d_uz_rt, float* d_ux, float* d_uz)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= 0 && i < Xn && j >= L - N && j < L + N)
	{		
		d_ux[i * Zn + j] = d_ux_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_uz[i * Zn + j] = d_uz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
	
	}
	if (i >= 0 && i < Xn && j >= Zn - L - N && j < Zn - L + N)
	{
		d_ux[i * Zn + j] = d_ux_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_uz[i * Zn + j] = d_uz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		
	}
	if (i >= L - N && i < L + N && j >= 0 && j < Zn)
	{		
		d_ux[i * Zn + j] = d_ux_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_uz[i * Zn + j] = d_uz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		

	}
	if (i >= Xn - L - N && i < Xn - L + N && j >= 0 && j < Zn)
	{		
		d_ux[i * Zn + j] = d_ux_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_uz[i * Zn + j] = d_uz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		

	}
	

}
__global__ void read_wavefiled2(float* d_theta_up, float* d_omega_up, float* d_duzdx_up, float* d_duzdz_up, float* d_duxdx_up, float* d_duxdz_up,
	float* d_theta_dn, float* d_omega_dn, float* d_duzdx_dn, float* d_duzdz_dn, float* d_duxdx_dn, float* d_duxdz_dn,
	float* d_theta_lf, float* d_omega_lf, float* d_duzdx_lf, float* d_duzdz_lf, float* d_duxdx_lf, float* d_duxdz_lf,
	float* d_theta_rt, float* d_omega_rt, float* d_duzdx_rt, float* d_duzdz_rt, float* d_duxdx_rt, float* d_duxdz_rt,
	float* d_theta, float* d_omega, float* d_duzdx, float* d_duzdz, float* d_duxdx, float* d_duxdz, int t, int L, int Xn, int Zn)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= 0 && i < Xn && j >= L - N && j < L + N)
	{
		d_theta[i * Zn + j] = d_theta_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_omega[i * Zn + j] = d_omega_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_duzdx[i * Zn + j] = d_duzdx_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_duzdz[i * Zn + j] = d_duzdz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_duxdx[i * Zn + j] = d_duxdx_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_duxdz[i * Zn + j] = d_duxdz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		
	}
	if (i >= 0 && i < Xn && j >= Zn - L - N && j < Zn - L + N)
	{
		d_theta[i * Zn + j] = d_theta_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_omega[i * Zn + j] = d_omega_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_duzdx[i * Zn + j] = d_duzdx_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_duzdz[i * Zn + j] = d_duzdz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_duxdx[i * Zn + j] = d_duxdx_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_duxdz[i * Zn + j] = d_duxdz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		
	}
	if (i >= L - N && i < L + N && j >= 0 && j < Zn)
	{
		d_theta[i * Zn + j] = d_theta_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_omega[i * Zn + j] = d_omega_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_duzdx[i * Zn + j] = d_duzdx_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_duzdz[i * Zn + j] = d_duzdz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_duxdx[i * Zn + j] = d_duxdx_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_duxdz[i * Zn + j] = d_duxdz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		

	}
	if (i >= Xn - L - N && i < Xn - L + N && j >= 0 && j < Zn)
	{
		d_theta[i * Zn + j] = d_theta_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_omega[i * Zn + j] = d_omega_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_duzdx[i * Zn + j] = d_duzdx_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_duzdz[i * Zn + j]= d_duzdz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_duxdx[i * Zn + j] = d_duxdx_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_duxdz[i * Zn + j] = d_duxdz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		

	}


}
__global__ void rt_s_res(int Xn, int Zn, float* ux, float* uz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* ddx, float* ddz, float* vp, float* vs,
	float* F_xx, float* F_zz, float* F_xz, float* F_zx) {

	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dux_dx = 0.0f;
	float duz_dz = 0.0f;
	float dux_dz = 0.0f;
	float duz_dx = 0.0f;
	
	float s;
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{

		dux_dx = (a[0] * (ux[(i + 0) * Zn + j] - ux[(i - 1) * Zn + j])
			+ a[1] * (ux[(i + 1) * Zn + j] - ux[(i - 2) * Zn + j])
			+ a[2] * (ux[(i + 2) * Zn + j] - ux[(i - 3) * Zn + j])
			+ a[3] * (ux[(i + 3) * Zn + j] - ux[(i - 4) * Zn + j])
			+ a[4] * (ux[(i + 4) * Zn + j] - ux[(i - 5) * Zn + j])
			+ a[5] * (ux[(i + 5) * Zn + j] - ux[(i - 6) * Zn + j])) / dx;

		dux_dz = (a[0] * (ux[(i)*Zn + j + 1] - ux[(i)*Zn + j - 0])
			+ a[1] * (ux[(i)*Zn + j + 2] - ux[(i)*Zn + j - 1])
			+ a[2] * (ux[(i)*Zn + j + 3] - ux[(i)*Zn + j - 2])
			+ a[3] * (ux[(i)*Zn + j + 4] - ux[(i)*Zn + j - 3])
			+ a[4] * (ux[(i)*Zn + j + 5] - ux[(i)*Zn + j - 4])
			+ a[5] * (ux[(i)*Zn + j + 6] - ux[(i)*Zn + j - 5])) / dz;

		duz_dz = (a[0] * (uz[(i)*Zn + j + 0] - uz[(i)*Zn + j - 1])
			+ a[1] * (uz[(i)*Zn + j + 1] - uz[(i)*Zn + j - 2])
			+ a[2] * (uz[(i)*Zn + j + 2] - uz[(i)*Zn + j - 3])
			+ a[3] * (uz[(i)*Zn + j + 3] - uz[(i)*Zn + j - 4])
			+ a[4] * (uz[(i)*Zn + j + 4] - uz[(i)*Zn + j - 5])
			+ a[5] * (uz[(i)*Zn + j + 5] - uz[(i)*Zn + j - 6])) / dz;

		duz_dx = (a[0] * (uz[(i + 1) * Zn + j] - uz[(i - 0) * Zn + j])
			+ a[1] * (uz[(i + 2) * Zn + j] - uz[(i - 1) * Zn + j])
			+ a[2] * (uz[(i + 3) * Zn + j] - uz[(i - 2) * Zn + j])
			+ a[3] * (uz[(i + 4) * Zn + j] - uz[(i - 3) * Zn + j])
			+ a[4] * (uz[(i + 5) * Zn + j] - uz[(i - 4) * Zn + j])
			+ a[5] * (uz[(i + 6) * Zn + j] - uz[(i - 5) * Zn + j])) / dx;

		F_xx[i * Zn + j] = F_xx[i * Zn + j] + (ddx[i * Zn + j] * dux_dx - ddx[i * Zn + j] * F_xx[i * Zn + j]) * dt;
		F_zz[i * Zn + j] = F_zz[i * Zn + j] + (ddz[i * Zn + j] * duz_dz - ddz[i * Zn + j] * F_zz[i * Zn + j]) * dt;
		F_xz[i * Zn + j] = F_xz[i * Zn + j] + (ddz[i * Zn + j] * dux_dz - ddz[i * Zn + j] * F_xz[i * Zn + j]) * dt;
		F_zx[i * Zn + j] = F_zx[i * Zn + j] + (ddx[i * Zn + j] * duz_dx - ddx[i * Zn + j] * F_zx[i * Zn + j]) * dt;

		theta[i * Zn + j] = vp[i * Zn + j] * vp[i * Zn + j] * (dux_dx + duz_dz - F_xx[i * Zn + j] - F_zz[i * Zn + j]);
		omega[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dz - duz_dx - F_xz[i * Zn + j] + F_zx[i * Zn + j]);
		duzdx[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (duz_dx - F_zx[i * Zn + j]);
		duzdz[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (duz_dz - F_zz[i * Zn + j]);
		duxdz[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dz - F_xz[i * Zn + j]);
		duxdx[i * Zn + j] = vs[i * Zn + j] * vs[i * Zn + j] * (dux_dx - F_xx[i * Zn + j]);


	}

}
__global__ void rt_u_res(int Xn, int Zn, float* ux, float* uz, float* upx_next, float* upz_next, float* usx_next, float* usz_next, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* ddz, float* ddx, float* upx_now, float* upz_now, float* usx_now, float* usz_now,
	float* upx_past, float* upz_past, float* usx_past, float* usz_past, float* O_duz_xdz, float* O_duz_zdx, float* O_dux_zdx, float* O_dux_xdz, float* O_thetax, float* O_thetaz, float* O_omegaz, float* O_omegax) {


	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dux_xdz = 0.0f;
	float duz_zdx = 0.0f;
	float duz_xdz = 0.0f;
	float dux_zdx = 0.0f;
	float dthetadx = 0.0f;
	float dthetadz = 0.0f;
	float domegadx = 0.0f;
	float domegadz = 0.0f;
	
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{
		dthetadx = (a[0] * (theta[(i + 1) * Zn + j] - theta[(i - 0) * Zn + j])
			+ a[1] * (theta[(i + 2) * Zn + j] - theta[(i - 1) * Zn + j])
			+ a[2] * (theta[(i + 3) * Zn + j] - theta[(i - 2) * Zn + j])
			+ a[3] * (theta[(i + 4) * Zn + j] - theta[(i - 3) * Zn + j])
			+ a[4] * (theta[(i + 5) * Zn + j] - theta[(i - 4) * Zn + j])
			+ a[5] * (theta[(i + 6) * Zn + j] - theta[(i - 5) * Zn + j])) / dx;

		dthetadz = (a[0] * (theta[(i)*Zn + j + 1] - theta[(i)*Zn + j - 0])
			+ a[1] * (theta[(i)*Zn + j + 2] - theta[(i)*Zn + j - 1])
			+ a[2] * (theta[(i)*Zn + j + 3] - theta[(i)*Zn + j - 2])
			+ a[3] * (theta[(i)*Zn + j + 4] - theta[(i)*Zn + j - 3])
			+ a[4] * (theta[(i)*Zn + j + 5] - theta[(i)*Zn + j - 4])
			+ a[5] * (theta[(i)*Zn + j + 6] - theta[(i)*Zn + j - 5])) / dz;

		domegadx = (a[0] * (omega[(i + 0) * Zn + j] - omega[(i - 1) * Zn + j])
			+ a[1] * (omega[(i + 1) * Zn + j] - omega[(i - 2) * Zn + j])
			+ a[2] * (omega[(i + 2) * Zn + j] - omega[(i - 3) * Zn + j])
			+ a[3] * (omega[(i + 3) * Zn + j] - omega[(i - 4) * Zn + j])
			+ a[4] * (omega[(i + 4) * Zn + j] - omega[(i - 5) * Zn + j])
			+ a[5] * (omega[(i + 5) * Zn + j] - omega[(i - 6) * Zn + j])) / dx;

		domegadz = (a[0] * (omega[(i)*Zn + j + 0] - omega[(i)*Zn + j - 1])
			+ a[1] * (omega[(i)*Zn + j + 1] - omega[(i)*Zn + j - 2])
			+ a[2] * (omega[(i)*Zn + j + 2] - omega[(i)*Zn + j - 3])
			+ a[3] * (omega[(i)*Zn + j + 3] - omega[(i)*Zn + j - 4])
			+ a[4] * (omega[(i)*Zn + j + 4] - omega[(i)*Zn + j - 5])
			+ a[5] * (omega[(i)*Zn + j + 5] - omega[(i)*Zn + j - 6])) / dz;


		dux_xdz = (a[0] * (duxdx[(i)*Zn + j + 1] - duxdx[(i)*Zn + j - 0])
			+ a[1] * (duxdx[(i)*Zn + j + 2] - duxdx[(i)*Zn + j - 1])
			+ a[2] * (duxdx[(i)*Zn + j + 3] - duxdx[(i)*Zn + j - 2])
			+ a[3] * (duxdx[(i)*Zn + j + 4] - duxdx[(i)*Zn + j - 3])
			+ a[4] * (duxdx[(i)*Zn + j + 5] - duxdx[(i)*Zn + j - 4])
			+ a[5] * (duxdx[(i)*Zn + j + 6] - duxdx[(i)*Zn + j - 5])) / dz;

		dux_zdx = (a[0] * (duxdz[(i + 0) * Zn + j] - duxdz[(i - 1) * Zn + j])
			+ a[1] * (duxdz[(i + 1) * Zn + j] - duxdz[(i - 2) * Zn + j])
			+ a[2] * (duxdz[(i + 2) * Zn + j] - duxdz[(i - 3) * Zn + j])
			+ a[3] * (duxdz[(i + 3) * Zn + j] - duxdz[(i - 4) * Zn + j])
			+ a[4] * (duxdz[(i + 4) * Zn + j] - duxdz[(i - 5) * Zn + j])
			+ a[5] * (duxdz[(i + 5) * Zn + j] - duxdz[(i - 6) * Zn + j])) / dx;

		duz_xdz = (a[0] * (duzdx[(i)*Zn + j + 0] - duzdx[(i)*Zn + j - 1])
			+ a[1] * (duzdx[(i)*Zn + j + 1] - duzdx[(i)*Zn + j - 2])
			+ a[2] * (duzdx[(i)*Zn + j + 2] - duzdx[(i)*Zn + j - 3])
			+ a[3] * (duzdx[(i)*Zn + j + 3] - duzdx[(i)*Zn + j - 4])
			+ a[4] * (duzdx[(i)*Zn + j + 4] - duzdx[(i)*Zn + j - 5])
			+ a[5] * (duzdx[(i)*Zn + j + 5] - duzdx[(i)*Zn + j - 6])) / dz;

		duz_zdx = (a[0] * (duzdz[(i + 1) * Zn + j] - duzdz[(i - 0) * Zn + j])
			+ a[1] * (duzdz[(i + 2) * Zn + j] - duzdz[(i - 1) * Zn + j])
			+ a[2] * (duzdz[(i + 3) * Zn + j] - duzdz[(i - 2) * Zn + j])
			+ a[3] * (duzdz[(i + 4) * Zn + j] - duzdz[(i - 3) * Zn + j])
			+ a[4] * (duzdz[(i + 5) * Zn + j] - duzdz[(i - 4) * Zn + j])
			+ a[5] * (duzdz[(i + 6) * Zn + j] - duzdz[(i - 5) * Zn + j])) / dx;

		O_duz_xdz[i * Zn + j] = O_duz_xdz[i * Zn + j] + (ddz[i * Zn + j] * duz_xdz - ddz[i * Zn + j] * O_duz_xdz[i * Zn + j]) * dt;
		O_duz_zdx[i * Zn + j] = O_duz_zdx[i * Zn + j] + (ddx[i * Zn + j] * duz_zdx - ddx[i * Zn + j] * O_duz_zdx[i * Zn + j]) * dt;
		O_dux_zdx[i * Zn + j] = O_dux_zdx[i * Zn + j] + (ddx[i * Zn + j] * dux_zdx - ddx[i * Zn + j] * O_dux_zdx[i * Zn + j]) * dt;
		O_dux_xdz[i * Zn + j] = O_dux_xdz[i * Zn + j] + (ddz[i * Zn + j] * dux_xdz - ddz[i * Zn + j] * O_dux_xdz[i * Zn + j]) * dt;

		O_thetax[i * Zn + j] = O_thetax[i * Zn + j] + (ddx[i * Zn + j] * dthetadx - ddx[i * Zn + j] * O_thetax[i * Zn + j]) * dt;
		O_thetaz[i * Zn + j] = O_thetaz[i * Zn + j] + (ddz[i * Zn + j] * dthetadz - ddz[i * Zn + j] * O_thetaz[i * Zn + j]) * dt;
		O_omegaz[i * Zn + j] = O_omegaz[i * Zn + j] + (ddz[i * Zn + j] * domegadz - ddz[i * Zn + j] * O_omegaz[i * Zn + j]) * dt;
		O_omegax[i * Zn + j] = O_omegax[i * Zn + j] + (ddx[i * Zn + j] * domegadx - ddx[i * Zn + j] * O_omegax[i * Zn + j]) * dt;

		upx_next[i * Zn + j] = 2 * upx_now[i * Zn + j] - upx_past[i * Zn + j] + (dt * dt) * (dthetadx - O_thetax[i * Zn + j] + duz_xdz - duz_zdx - O_duz_xdz[i * Zn + j] + O_duz_zdx[i * Zn + j]);
		upz_next[i * Zn + j] = 2 * upz_now[i * Zn + j] - upz_past[i * Zn + j] + (dt * dt) * (dthetadz - O_thetaz[i * Zn + j] + dux_zdx - dux_xdz - O_dux_zdx[i * Zn + j] + O_dux_xdz[i * Zn + j]);
		usx_next[i * Zn + j] = 2 * usx_now[i * Zn + j] - usx_past[i * Zn + j] + (dt * dt) * (domegadz - O_omegaz[i * Zn + j] + duz_xdz - duz_zdx - O_duz_xdz[i * Zn + j] + O_duz_zdx[i * Zn + j]);
		usz_next[i * Zn + j] = 2 * usz_now[i * Zn + j] - usz_past[i * Zn + j] + (dt * dt) * (-domegadx + O_omegax[i * Zn + j] + dux_zdx - dux_xdz - O_dux_zdx[i * Zn + j] + O_dux_xdz[i * Zn + j]);

		upx_past[i * Zn + j] = upx_now[i * Zn + j];	upx_now[i * Zn + j] = upx_next[i * Zn + j];
		upz_past[i * Zn + j] = upz_now[i * Zn + j];	upz_now[i * Zn + j] = upz_next[i * Zn + j];
		usx_past[i * Zn + j] = usx_now[i * Zn + j];	usx_now[i * Zn + j] = usx_next[i * Zn + j];
		usz_past[i * Zn + j] = usz_now[i * Zn + j];	usz_now[i * Zn + j] = usz_next[i * Zn + j];


		ux[i * Zn + j] = upx_next[i * Zn + j] + usx_next[i * Zn + j];
		uz[i * Zn + j] = upz_next[i * Zn + j] + usz_next[i * Zn + j];

	}


}
__global__ void poynting(int Xn, int Zn, int L,float* upx_r, float* upz_r, float* usx_r, float* usz_r, float* theta_r, float* omega_r, float* upx_s, float* upz_s, float* theta_s, float* omega_s,
	float* Epx_S, float* Epz_S, float* Epx_R, float* Epz_R, float* Esx_R, float* Esz_R,
	float* fenzi_PP, float* fenzi_PS, float* fenmu_P,
	float* RR_upx_u, float* RR_upx_d, float* RR_upx_l, float* RR_upx_r, float* RR_upz_u, float* RR_upz_d, float* RR_upz_l, float* RR_upz_r,
	float* RR_usx_u, float* RR_usx_d, float* RR_usx_l, float* RR_usx_r, float* RR_usz_u, float* RR_usz_d, float* RR_usz_l, float* RR_usz_r,
	float* SS_upx_u, float* SS_upx_d, float* SS_upx_l, float* SS_upx_r, float* SS_upz_u, float* SS_upz_d, float* SS_upz_l, float* SS_upz_r)
{
	int i, j;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;


	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{
		
		Epx_S[i * Zn + j] = -theta_s[i * Zn + j] * upx_s[i * Zn + j];
		Epz_S[i * Zn + j] = -theta_s[i * Zn + j] * upz_s[i * Zn + j];  


	
		Epx_R[i * Zn + j] = -theta_r[i * Zn + j] * upx_r[i * Zn + j];
		Epz_R[i * Zn + j] = -theta_r[i * Zn + j] * upz_r[i * Zn + j];

		Esx_R[i * Zn + j] = omega_r[i * Zn + j] * usz_r[i * Zn + j];
		Esz_R[i * Zn + j] = -omega_r[i * Zn + j] * usx_r[i * Zn + j];





	
		if (Epz_R[i * Zn + j] >= 0)
		{
			RR_upx_u[i * Zn + j] = 0.0;
			RR_upx_d[i * Zn + j] = upx_r[i * Zn + j];
		}
		else
		{
			RR_upx_u[i * Zn + j] = upx_r[i * Zn + j];
			RR_upx_d[i * Zn + j] = 0.0;
		}

		if (Epx_R[i * Zn + j] >= 0)
		{
			RR_upx_l[i * Zn + j] = 0.0;
			RR_upx_r[i * Zn + j] = upx_r[i * Zn + j];
		}
		else
		{
			RR_upx_l[i * Zn + j] = upx_r[i * Zn + j];
			RR_upx_r[i * Zn + j] = 0.0;
		}

	
		if (Epz_R[i * Zn + j] >= 0)
		{
			RR_upz_u[i * Zn + j] = 0.0;
			RR_upz_d[i * Zn + j] = upz_r[i * Zn + j];
		}
		else
		{
			RR_upz_u[i * Zn + j] = upz_r[i * Zn + j];
			RR_upz_d[i * Zn + j] = 0.0;
		}

		if (Epx_R[i * Zn + j] >= 0)
		{
			RR_upz_l[i * Zn + j] = 0.0;
			RR_upz_r[i * Zn + j] = upz_r[i * Zn + j];
		}
		else
		{
			RR_upz_l[i * Zn + j] = upz_r[i * Zn + j];
			RR_upz_r[i * Zn + j] = 0.0;
		}


	
		if (Esz_R[i * Zn + j] >= 0)
		{
			RR_usx_u[i * Zn + j] = 0.0;
			RR_usx_d[i * Zn + j] = usx_r[i * Zn + j];
		}
		else
		{
			RR_usx_u[i * Zn + j] = usx_r[i * Zn + j];
			RR_usx_d[i * Zn + j] = 0.0;
		}

		if (Esx_R[i * Zn + j] >= 0)
		{
			RR_usx_l[i * Zn + j] = 0.0;
			RR_usx_r[i * Zn + j] = usx_r[i * Zn + j];
		}
		else
		{
			RR_usx_l[i * Zn + j] = usx_r[i * Zn + j];
			RR_usx_r[i * Zn + j] = 0.0;
		}

	
		if (Esz_R[i * Zn + j] >= 0)
		{
			RR_usz_u[i * Zn + j] = 0.0;
			RR_usz_d[i * Zn + j] = usz_r[i * Zn + j];
		}
		else
		{
			RR_usz_u[i * Zn + j] = usz_r[i * Zn + j];
			RR_usz_d[i * Zn + j] = 0.0;
		}

		if (Esx_R[i * Zn + j] >= 0)
		{
			RR_usz_l[i * Zn + j] = 0.0;
			RR_usz_r[i * Zn + j] = usz_r[i * Zn + j];
		}
		else
		{
			RR_usz_l[i * Zn + j] = usz_r[i * Zn + j];
			RR_usz_r[i * Zn + j] = 0.0;
		}

		
		if (Epz_S[i * Zn + j] >= 0)
		{
			SS_upx_u[i * Zn + j] = 0.0;
			SS_upx_d[i * Zn + j] = upx_s[i * Zn + j];
		}
		else
		{
			SS_upx_u[i * Zn + j] = upx_s[i * Zn + j];
			SS_upx_d[i * Zn + j] = 0.0;
		}

		if (Epx_S[i * Zn + j] >= 0)
		{
			SS_upx_l[i * Zn + j] = 0.0;
			SS_upx_r[i * Zn + j] = upx_s[i * Zn + j];
		}
		else
		{
			SS_upx_l[i * Zn + j] = upx_s[i * Zn + j];
			SS_upx_r[i * Zn + j] = 0.0;
		}

		
		if (Epz_S[i * Zn + j] >= 0)
		{
			SS_upz_u[i * Zn + j] = 0.0;
			SS_upz_d[i * Zn + j] = upz_s[i * Zn + j];
		}
		else
		{
			SS_upz_u[i * Zn + j] = upz_s[i * Zn + j];
			SS_upz_d[i * Zn + j] = 0.0;
		}

		if (Epx_S[i * Zn + j] >= 0)
		{
			SS_upz_l[i * Zn + j] = 0.0;
			SS_upz_r[i * Zn + j] = upz_s[i * Zn + j];
		}
		else
		{
			SS_upz_l[i * Zn + j] = upz_s[i * Zn + j];
			SS_upz_r[i * Zn + j] = 0.0;
		}


		fenzi_PP[i * Zn + j] += SS_upx_u[i * Zn + j] * RR_upx_d[i * Zn + j] + SS_upz_u[i * Zn + j] * RR_upz_d[i * Zn + j] +
			SS_upx_d[i * Zn + j] * RR_upx_u[i * Zn + j] + SS_upz_d[i * Zn + j] * RR_upz_u[i * Zn + j] +
			SS_upx_r[i * Zn + j] * RR_upx_l[i * Zn + j] + SS_upz_r[i * Zn + j] * RR_upz_l[i * Zn + j] +
			SS_upx_l[i * Zn + j] * RR_upx_r[i * Zn + j] + SS_upz_l[i * Zn + j] * RR_upz_r[i * Zn + j];

		fenzi_PS[i * Zn + j] += SS_upx_u[i * Zn + j] * RR_usx_d[i * Zn + j] + SS_upz_u[i * Zn + j] * RR_usz_d[i * Zn + j] +
			SS_upx_d[i * Zn + j] * RR_usx_u[i * Zn + j] + SS_upz_d[i * Zn + j] * RR_usz_u[i * Zn + j] +
			SS_upx_r[i * Zn + j] * RR_usx_l[i * Zn + j] + SS_upz_r[i * Zn + j] * RR_usz_l[i * Zn + j] +
			SS_upx_l[i * Zn + j] * RR_usx_r[i * Zn + j] + SS_upz_l[i * Zn + j] * RR_usz_r[i * Zn + j];

		fenmu_P[i * Zn + j] += upx_s[i * Zn + j] * upx_s[i * Zn + j] + upz_s[i * Zn + j] * upz_s[i * Zn + j];
	}

}
__global__ void corr_v(int Xn, int Zn, int L, float* fenzi_PP, float* fenzi_PS, float* fenmu_P, float* SS_Px, float* SS_Pz, float* PP_Px, float* PP_Pz, float* PP_Sx, float* PP_Sz)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{
		fenzi_PP[i * Zn + j] += SS_Px[i * Zn + j] * PP_Px[i * Zn + j] + SS_Pz[i * Zn + j] * PP_Pz[i * Zn + j];
		fenmu_P[i * Zn + j] += SS_Px[i * Zn + j] * SS_Px[i * Zn + j] + SS_Pz[i * Zn + j] * SS_Pz[i * Zn + j];
		fenzi_PS[i * Zn + j] += SS_Px[i * Zn + j] * PP_Sx[i * Zn + j] + SS_Pz[i * Zn + j] * PP_Sz[i * Zn + j];

	}
}
__global__ void image_fun(int Xn, int Zn, int L, float* fenzi, float* fenmu, float* image)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{
		
		image[i * Zn + j] = fenzi[i * Zn + j] / fenmu[i * Zn + j];
	
	}
}
__global__ void Laplace(int Xn, int Zn, int L, float dh, float* image, float* image_lap)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= L + 2 && i < Xn - L - 2 && j >= L + 2 && j < Zn - L - 2)
	{
		image_lap[i * Zn + j] = (1 / dh / dh) * (image[(i + 1) * Zn + j] + image[(i - 1) * Zn + j] - 2 * image[i * Zn + j] +
			image[i * Zn + j + 1] + image[i * Zn + j - 1] - 2 * image[i * Zn + j]);
	}

}
int main() {
	cudaSetDevice(1);
	int i, j, t,t0;
	int xn = 400;
	int zn = 200;
	int shot_num = 80;
	int shotx;
	int shotz;
	
	int receiver_interp = 1;
	int receiver_depth = pml + 4;
	int receiver_num = 400;	
	int shotno;
	
	int Xn;
	int Zn;

	Xn = xn + 2 * pml;
	Zn = zn + 2 * pml;
	
	FILE* fp;	
	char filename[2000];
	int dh;
	dh = 5.0;

	
	float* source = (float*)calloc(Tn, sizeof(float));
	float Nk = pi * pi * fm * fm * dt * dt;
	t0 = ceil(1.0 / (fm * dt));
	for (t = 0; t < Tn; t++)
	{
		source[t] = (1.0 - 2.0 * Nk * (t - t0) * (t - t0)) * exp(-Nk * (t - t0) * (t - t0));
		
	}
	char vpfile[1000], vsfile[1000], denfile[1000];
	float* vp = (float*)calloc(Xn * Zn, sizeof(float));
	float* vs = (float*)calloc(Xn * Zn, sizeof(float));
	float* rou = (float*)calloc(Xn * Zn, sizeof(float));
	float* vp_pml = (float*)calloc(Xn * Zn, sizeof(float));
	float* vs_pml = (float*)calloc(Xn * Zn, sizeof(float));
	float* rou_pml = (float*)calloc(Xn * Zn, sizeof(float));
	float* ddx = (float*)calloc(Xn * Zn, sizeof(float));
	float* ddz = (float*)calloc(Xn * Zn, sizeof(float));

	dim3 dimGrid(ceil(Xn / 8.0), ceil(Zn / 8.0), 1);
	dim3 dimBlock(8, 8, 1);

	
	for (i = pml; i < Xn - pml; i++)
	{
		for (j = pml; j < zn / 2 + pml; j++)
		{
			vp[i * Zn + j] = 3200;
			vs[i * Zn + j] = 1700;
			rou[i * Zn + j] = 1.0;
		}
	}

	for (i = pml; i < Xn - pml; i++)
	{
		for (j = zn / 2 + pml; j < Zn - pml; j++)
		{
			vp[i * Zn + j] = 3900;
			vs[i * Zn + j] = 2300;
			rou[i * Zn + j] = 1.0;
		}
	}

	addpml(vp, vs, rou, vp_pml, vs_pml, rou_pml, Xn, Zn);


	dumpingfactor(vp_pml, ddx, ddz, Xn, Zn, xn, zn);
	sprintf(filename, "./snapshot/vp_%d_%d.dat", xn, zn);
	write_bin(vp_pml, filename, Xn, Zn, pml);
	sprintf(filename, "./snapshot/vs_%d_%d.dat", xn, zn);
	write_bin(vs_pml, filename, Xn, Zn, pml);

	float* frontwaveux = (float*)calloc(Xn * Zn, sizeof(float));
	float* frontwaveuz = (float*)calloc(Xn * Zn, sizeof(float));
	float* ux = (float*)calloc(Xn * Zn, sizeof(float));
	float* uz = (float*)calloc(Xn * Zn, sizeof(float));
	float* upx = (float*)calloc(Xn * Zn, sizeof(float));
	float* upz = (float*)calloc(Xn * Zn, sizeof(float));
	float* usx = (float*)calloc(Xn * Zn, sizeof(float));
	float* usz = (float*)calloc(Xn * Zn, sizeof(float));
	float* record_upx = (float*)calloc(Xn * Tn, sizeof(float));
	float* record_upz = (float*)calloc(Xn * Tn, sizeof(float));
	float* record_usx = (float*)calloc(Xn * Tn, sizeof(float));
	float* record_usz = (float*)calloc(Xn * Tn, sizeof(float));
	float* record_ux = (float*)calloc(Xn * Tn, sizeof(float));
	float* record_uz = (float*)calloc(Xn * Tn, sizeof(float));
	float* ux_s = (float*)calloc(Xn * Zn, sizeof(float));
	float* uz_s = (float*)calloc(Xn * Zn, sizeof(float));
	float* ux_r = (float*)calloc(Xn * Zn, sizeof(float));
	float* uz_r = (float*)calloc(Xn * Zn, sizeof(float));
	
	float* image_PP = (float*)calloc(Xn * Zn, sizeof(float));
	float* image_PS = (float*)calloc(Xn * Zn, sizeof(float));
	float* image_PP_lap = (float*)calloc(Xn * Zn, sizeof(float));
	float* image_PS_lap = (float*)calloc(Xn * Zn, sizeof(float));
	float* image_PP_pyt = (float*)calloc(Xn * Zn, sizeof(float));
	float* image_PS_pyt = (float*)calloc(Xn * Zn, sizeof(float));
	float* All_image_PP = (float*)calloc(Xn * Zn, sizeof(float));
	float* All_image_PS = (float*)calloc(Xn * Zn, sizeof(float));
	float* All_image_PP_lap = (float*)calloc(Xn * Zn, sizeof(float));
	float* All_image_PS_lap = (float*)calloc(Xn * Zn, sizeof(float));
	float* All_image_PP_pyt = (float*)calloc(Xn * Zn, sizeof(float));
	float* All_image_PS_pyt = (float*)calloc(Xn * Zn, sizeof(float));
	

	float* d_ddx, * d_ddz, * d_vp, * d_vs;
	float* d_upx_next, * d_upx_now, * d_upx_past, * d_upz_next, * d_upz_now, * d_upz_past, * d_usx_next, * d_usx_now, * d_usx_past, * d_usz_next, * d_usz_now, * d_usz_past;	
	float* d_recordupx, * d_recordupz, * d_recordusx, * d_recordusz, * d_recordux, * d_recorduz;
	float* O_duz_xdz, * O_duz_zdx, * O_dux_zdx, * O_dux_xdz, * O_thetax, * O_thetaz, * O_omegaz, * O_omegax;
	float* F_xx, * F_zz, * F_xz, * F_zx;
	float* d_ux, * d_uz, * d_theta, * d_omega, * d_duzdx, * d_duzdz, * d_duxdx, * d_duxdz;
	float* d_source;

	float* d_ux_up, * d_uz_up, * d_theta_up, * d_omega_up, * d_duzdx_up, * d_duzdz_up, * d_duxdx_up, * d_duxdz_up;
	float* d_ux_dn, * d_uz_dn, * d_theta_dn, * d_omega_dn, * d_duzdx_dn, * d_duzdz_dn, * d_duxdx_dn, * d_duxdz_dn;
	float* d_ux_lf, * d_uz_lf, * d_theta_lf, * d_omega_lf, * d_duzdx_lf, * d_duzdz_lf, * d_duxdx_lf, * d_duxdz_lf;
	float* d_ux_rt, * d_uz_rt, * d_theta_rt, * d_omega_rt, * d_duzdx_rt, * d_duzdz_rt, * d_duxdx_rt, * d_duxdz_rt;
	float* d_upx_next_up, * d_upz_next_up, * d_usx_next_up, * d_usz_next_up, * d_upx_now_up, * d_upz_now_up, * d_usx_now_up, * d_usz_now_up;
	float* d_upx_next_dn, * d_upz_next_dn, * d_usx_next_dn, * d_usz_next_dn, * d_upx_now_dn, * d_upz_now_dn, * d_usx_now_dn, * d_usz_now_dn;
	float* d_upx_next_lf, * d_upz_next_lf, * d_usx_next_lf, * d_usz_next_lf, * d_upx_now_lf, * d_upz_now_lf, * d_usx_now_lf, * d_usz_now_lf;
	float* d_upx_next_rt, * d_upz_next_rt, * d_usx_next_rt, * d_usz_next_rt, * d_upx_now_rt, * d_upz_now_rt, * d_usx_now_rt, * d_usz_now_rt;

	float* d_ux_s, * d_uz_s, * d_theta_s, * d_omega_s, * d_duzdx_s, * d_duzdz_s, * d_duxdz_s, * d_duxdx_s;
	float* d_upx_next_s, * d_upz_next_s, * d_usx_next_s, * d_usz_next_s;
	float* d_upx_now_s, * d_upz_now_s, * d_usx_now_s, * d_usz_now_s;
	float* d_upx_past_s, * d_upz_past_s, * d_usx_past_s, * d_usz_past_s;

	float* d_ux_r, * d_uz_r, * d_theta_r, * d_omega_r, * d_duxdz_r, * d_duzdz_r, * d_duxdx_r, * d_duzdx_r;
	float* d_upx_next_r, * d_upz_next_r, * d_usx_next_r, * d_usz_next_r;
	float* d_upx_now_r, * d_upz_now_r, * d_usx_now_r, * d_usz_now_r;
	float* d_upx_past_r, * d_upz_past_r, * d_usx_past_r, * d_usz_past_r;
	float* O_duz_xdz_r, * O_duz_zdx_r, * O_dux_zdx_r, * O_dux_xdz_r;
	float* O_thetax_r, * O_thetaz_r, * O_omegaz_r, * O_omegax_r;
	float* F_xx_r, * F_zz_r, * F_xz_r, * F_zx_r;
	
	float* d_Epx_S, * d_Epz_S, * d_Epx_R, * d_Epz_R, * d_Esx_R, * d_Esz_R;	
	float* d_fenzi_PP_pyt, * d_fenzi_PS_pyt, * d_fenmu_P_pyt;
	float* d_fenzi_PP, * d_fenzi_PS, * d_fenmu_P;
	float* d_RR_upx_up, * d_RR_upx_dn, * d_RR_upx_lf, * d_RR_upx_rt;
	float* d_RR_upz_up, * d_RR_upz_dn, * d_RR_upz_lf, * d_RR_upz_rt;
	float* d_RR_usx_up, * d_RR_usx_dn, * d_RR_usx_lf, * d_RR_usx_rt;
	float* d_RR_usz_up, * d_RR_usz_dn, * d_RR_usz_lf, * d_RR_usz_rt;
	float* d_SS_upx_up, * d_SS_upx_dn, * d_SS_upx_lf, * d_SS_upx_rt;
	float* d_SS_upz_up, * d_SS_upz_dn, * d_SS_upz_lf, * d_SS_upz_rt;
	
	float* d_image_PP, * d_image_PS;
	float* d_image_PP_lap, * d_image_PS_lap;
	float* d_image_PP_pyt, * d_image_PS_pyt;

	cudaMalloc((void**)&d_ddx, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_ddz, Xn * Zn * sizeof(float));	
	cudaMalloc((void**)&d_vp, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_vs, Xn * Zn * sizeof(float));

	cudaMalloc((void**)&d_upx_next, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_upx_now, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_upx_past, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_next, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_now, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_past, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_next, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_now, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_past, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_next, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_now, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_past, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_ux, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_uz, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_theta, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_omega, Xn * Zn * sizeof(float));
	
	cudaMalloc((void**)&O_duz_xdz, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&O_duz_zdx, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&O_dux_zdx, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&O_dux_xdz, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&O_thetax, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&O_thetaz, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&O_omegaz, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&O_omegax, Xn * Zn * sizeof(float));	
	cudaMalloc((void**)&F_xx, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&F_zz, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&F_xz, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&F_zx, Xn * Zn * sizeof(float));	
	cudaMalloc((void**)&d_duzdx, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_duzdz, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_duxdx, Xn * Zn * sizeof(float));
	cudaMalloc((void**)&d_duxdz, Xn * Zn * sizeof(float));

	cudaMalloc((void**)&d_recordupx, Xn* Tn * sizeof(float));
	cudaMalloc((void**)&d_recordupz, Xn* Tn * sizeof(float));
	cudaMalloc((void**)&d_recordusx, Xn* Tn * sizeof(float));
	cudaMalloc((void**)&d_recordusz, Xn* Tn * sizeof(float));
	cudaMalloc((void**)&d_recordux, Xn* Tn * sizeof(float));
	cudaMalloc((void**)&d_recorduz, Xn* Tn * sizeof(float));
	cudaMalloc((void**)&d_source, Tn * sizeof(float));
	

	cudaMalloc((void**)&d_ux_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_uz_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_theta_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_omega_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdx_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdz_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdx_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdz_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upx_next_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upz_next_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usx_next_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usz_next_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upx_now_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upz_now_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usx_now_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usz_now_up, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_ux_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_uz_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_theta_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_omega_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdx_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdz_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdx_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdz_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upx_next_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upz_next_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usx_next_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usz_next_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upx_now_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upz_now_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usx_now_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usz_now_dn, 2 * N * Xn * (Tn - 1) * sizeof(float));
	
	cudaMalloc((void**)&d_ux_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_uz_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_theta_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_omega_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdx_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdz_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdx_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdz_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upx_next_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upz_next_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usx_next_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usz_next_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upx_now_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upz_now_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usx_now_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usz_now_lf, 2 * N * Zn * (Tn - 1) * sizeof(float));	
	cudaMalloc((void**)&d_ux_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_uz_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_theta_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_omega_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdx_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdz_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdx_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdz_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upx_next_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upz_next_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usx_next_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usz_next_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upx_now_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_upz_now_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usx_now_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMalloc((void**)&d_usz_now_rt, 2 * N * Zn * (Tn - 1) * sizeof(float));
		
	cudaMalloc((void**)&d_ux_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_uz_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_theta_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_omega_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_duzdx_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_duzdz_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_duxdz_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_duxdx_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upx_next_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_next_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_next_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_next_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upx_now_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_now_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_now_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_now_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upx_past_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_past_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_past_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_past_s, Xn* Zn * sizeof(float));
	
	cudaMalloc((void**)&d_ux_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_uz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_theta_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_omega_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_duxdz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_duzdz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_duxdx_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_duzdx_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upx_next_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_next_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_next_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_next_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upx_now_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_now_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_now_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_now_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upx_past_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_upz_past_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usx_past_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_usz_past_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_duz_xdz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_duz_zdx_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_dux_zdx_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_dux_xdz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_thetax_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_thetaz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_omegaz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_omegax_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&F_xx_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&F_zz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&F_xz_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&F_zx_r, Xn* Zn * sizeof(float));

	cudaMalloc((void**)&d_Epx_S, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_Epz_S, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_Epx_R, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_Epz_R, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_Esx_R, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_Esz_R, Xn* Zn * sizeof(float));

	cudaMalloc((void**)&d_fenzi_PP_pyt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_fenzi_PS_pyt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_fenmu_P_pyt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_fenzi_PP, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_fenzi_PS, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_fenmu_P, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PP, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PS, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PP_lap, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PS_lap, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PP_pyt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PS_pyt, Xn* Zn * sizeof(float));

	cudaMalloc((void**)&d_RR_upx_up, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_upz_up, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_usx_up, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_usz_up, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_SS_upx_up, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_SS_upz_up, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_upx_dn, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_upz_dn, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_usx_dn, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_usz_dn, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_SS_upx_dn, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_SS_upz_dn, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_upx_lf, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_upz_lf, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_usx_lf, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_usz_lf, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_SS_upx_lf, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_SS_upz_lf, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_upx_rt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_upz_rt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_usx_rt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_RR_usz_rt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_SS_upx_rt, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_SS_upz_rt, Xn* Zn * sizeof(float));


	cudaMemset(d_vp, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_vs, 0, Xn * Zn * sizeof(float));	
	cudaMemset(d_ddx, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_ddz, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_upx_next, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_upx_now, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_upx_past, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_upz_next, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_upz_now, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_upz_past, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_usx_next, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_usx_now, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_usx_past, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_usz_next, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_usz_now, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_usz_past, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_ux, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_uz, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_theta, 0, Xn * Zn * sizeof(float));
	cudaMemset(d_omega, 0, Xn * Zn * sizeof(float));
	
	cudaMemset(O_duz_xdz, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_duz_zdx, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_dux_zdx, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_dux_xdz, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_thetax, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_thetaz, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_omegaz, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_omegax, 0, Xn* Zn * sizeof(float));	
	cudaMemset(F_xx, 0, Xn* Zn * sizeof(float));
	cudaMemset(F_zz, 0, Xn* Zn * sizeof(float));
	cudaMemset(F_xz, 0, Xn* Zn * sizeof(float));
	cudaMemset(F_zx, 0, Xn* Zn * sizeof(float));	
	cudaMemset(d_duzdx, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duzdz, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duxdx, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duxdz, 0, Xn* Zn * sizeof(float));

	cudaMemset(d_recordupx, 0, Xn * Tn * sizeof(float));
	cudaMemset(d_recordupz, 0, Xn * Tn * sizeof(float));
	cudaMemset(d_recordusx, 0, Xn * Tn * sizeof(float));
	cudaMemset(d_recordusz, 0, Xn * Tn * sizeof(float));
	cudaMemset(d_recordux, 0, Xn * Tn * sizeof(float));
	cudaMemset(d_recorduz, 0, Xn * Tn * sizeof(float));
	

	cudaMemset(d_ux_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_uz_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_theta_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_omega_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duzdx_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duzdz_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duxdx_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duxdz_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upx_next_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upz_next_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usx_next_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usz_next_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upx_now_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upz_now_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usx_now_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usz_now_up, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_ux_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_uz_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_theta_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_omega_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duzdx_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duzdz_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duxdx_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duxdz_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upx_next_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upz_next_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usx_next_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usz_next_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upx_now_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upz_now_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usx_now_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usz_now_dn, 0, 2 * N * Xn * (Tn - 1) * sizeof(float));	
	cudaMemset(d_ux_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_uz_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_theta_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_omega_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duzdx_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duzdz_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duxdx_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duxdz_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upx_next_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upz_next_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usx_next_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usz_next_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upx_now_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upz_now_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usx_now_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usz_now_lf, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_ux_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_uz_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_theta_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_omega_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duzdx_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duzdz_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duxdx_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_duxdz_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upx_next_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upz_next_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usx_next_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usz_next_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upx_now_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_upz_now_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usx_now_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	cudaMemset(d_usz_now_rt, 0, 2 * N * Zn * (Tn - 1) * sizeof(float));
	
	cudaMemset(d_ux_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_uz_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_theta_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_omega_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duzdx_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duzdz_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duxdz_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duxdx_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upx_next_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upz_next_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usx_next_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usz_next_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upx_now_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upz_now_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usx_now_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usz_now_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upx_past_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upz_past_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usx_past_s, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usz_past_s, 0, Xn* Zn * sizeof(float));
	
	cudaMemset(d_ux_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_uz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_theta_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_omega_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duxdz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duzdz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duxdx_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_duzdx_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upx_next_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upz_next_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usx_next_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usz_next_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upx_now_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upz_now_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usx_now_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usz_now_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upx_past_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_upz_past_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usx_past_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_usz_past_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_duz_xdz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_duz_zdx_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_dux_zdx_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_dux_xdz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_thetax_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_thetaz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_omegaz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(O_omegax_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(F_xx_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(F_zz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(F_xz_r, 0, Xn* Zn * sizeof(float));
	cudaMemset(F_zx_r, 0, Xn* Zn * sizeof(float));
	
	cudaMemset(d_Epx_S, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_Epz_S, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_Epx_R, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_Epz_R, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_Esx_R, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_Esz_R, 0, Xn* Zn * sizeof(float));	
	cudaMemset(d_RR_upx_up, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_upx_dn, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_upx_lf, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_upx_rt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_upz_up, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_upz_dn, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_upz_lf, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_upz_rt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_usx_up, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_usx_dn, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_usx_lf, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_usx_rt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_usz_up, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_usz_dn, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_usz_lf, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_RR_usz_rt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_SS_upx_up, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_SS_upx_dn, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_SS_upx_lf, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_SS_upx_rt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_SS_upz_up, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_SS_upz_dn, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_SS_upz_lf, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_SS_upz_rt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_fenzi_PP_pyt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_fenzi_PS_pyt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_fenmu_P_pyt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_fenzi_PP, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_fenzi_PS, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_fenmu_P, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_image_PP, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_image_PS, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_image_PP_lap, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_image_PS_lap, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_image_PP_pyt, 0, Xn* Zn * sizeof(float));
	cudaMemset(d_image_PS_pyt, 0, Xn* Zn * sizeof(float));

	cudaMemcpy(d_ddx, ddx, Xn* Zn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ddz, ddz, Xn* Zn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vp, vp_pml, Xn* Zn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vs, vs_pml, Xn* Zn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_source, source, Tn * sizeof(float), cudaMemcpyHostToDevice);


	int k;
	for (k = 0; k < 1; k++)
	{
		cout << k + 1 << " th iteration:" << endl;

		for (shotno = 0; shotno < shot_num; shotno++) {

			shotx = pml + shotno * 5;
			shotz = pml + 1;
			

			cout << k + 1 << "  shotnumber:" << shotno + 1 << endl;
			cudaMemset(d_upx_next, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upx_now, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upx_past, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_next, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_now, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_past, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_next, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_now, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_past, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_next, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_now, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_past, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_ux, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_uz, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_theta, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_omega, 0, Xn * Zn * sizeof(float));

			cudaMemset(O_duz_xdz, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_duz_zdx, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_dux_zdx, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_dux_xdz, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_thetax, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_thetaz, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_omegaz, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_omegax, 0, Xn * Zn * sizeof(float));
			cudaMemset(F_xx, 0, Xn * Zn * sizeof(float));
			cudaMemset(F_zz, 0, Xn * Zn * sizeof(float));
			cudaMemset(F_xz, 0, Xn * Zn * sizeof(float));
			cudaMemset(F_zx, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duzdx, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duzdz, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duxdx, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duxdz, 0, Xn * Zn * sizeof(float));

			cudaMemset(d_recordupx, 0, Xn * Tn * sizeof(float));
			cudaMemset(d_recordupz, 0, Xn * Tn * sizeof(float));
			cudaMemset(d_recordusx, 0, Xn * Tn * sizeof(float));
			cudaMemset(d_recordusz, 0, Xn * Tn * sizeof(float));
			cudaMemset(d_recordux, 0, Xn * Tn * sizeof(float));
			cudaMemset(d_recorduz, 0, Xn * Tn * sizeof(float));

			cudaMemset(d_ux_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_uz_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_theta_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_omega_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duzdx_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duzdz_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duxdz_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duxdx_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upx_next_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_next_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_next_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_next_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upx_now_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_now_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_now_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_now_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upx_past_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_past_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_past_s, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_past_s, 0, Xn * Zn * sizeof(float));

			cudaMemset(d_ux_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_uz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_theta_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_omega_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duxdz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duzdz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duxdx_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_duzdx_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upx_next_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_next_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_next_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_next_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upx_now_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_now_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_now_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_now_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upx_past_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_upz_past_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usx_past_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(d_usz_past_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_duz_xdz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_duz_zdx_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_dux_zdx_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_dux_xdz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_thetax_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_thetaz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_omegaz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(O_omegax_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(F_xx_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(F_zz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(F_xz_r, 0, Xn * Zn * sizeof(float));
			cudaMemset(F_zx_r, 0, Xn * Zn * sizeof(float));
			for (t = 0; t < Tn; t++) {

				forward_s << <dimGrid, dimBlock >> > (Xn, Zn, d_vp, d_vs, d_ux, d_uz, d_ddx, d_ddz, d_theta, d_omega, t, shotx, shotz, d_source,
					F_xx, F_zz, F_xz, F_zx, d_duxdz, d_duzdz, d_duxdx, d_duzdx);

				forward_u << <dimGrid, dimBlock >> > (Xn, Zn, d_ux, d_uz, d_ddx, d_ddz,
					d_upx_next, d_upx_now, d_upx_past, d_upz_next, d_upz_now, d_upz_past, d_usx_next, d_usx_now, d_usx_past, d_usz_next, d_usz_now, d_usz_past,
					d_recordupx, d_recordupz, d_recordusx, d_recordusz, d_recordux, d_recorduz,
					d_theta, d_omega, receiver_depth, t, d_duxdz, d_duzdz, d_duxdx, d_duzdx,
					O_duz_xdz, O_duz_zdx, O_dux_zdx, O_dux_xdz, O_thetax, O_thetaz, O_omegaz, O_omegax);

				if (t != Tn - 1)
				{

					save_wavefiled << <dimGrid, dimBlock >> > (Xn, Zn, pml, d_ux_up, d_uz_up, d_theta_up, d_omega_up, d_duzdx_up, d_duzdz_up, d_duxdx_up, d_duxdz_up,
						d_ux_dn, d_uz_dn, d_theta_dn, d_omega_dn, d_duzdx_dn, d_duzdz_dn, d_duxdx_dn, d_duxdz_dn,
						d_ux_lf, d_uz_lf, d_theta_lf, d_omega_lf, d_duzdx_lf, d_duzdz_lf, d_duxdx_lf, d_duxdz_lf,
						d_ux_rt, d_uz_rt, d_theta_rt, d_omega_rt, d_duzdx_rt, d_duzdz_rt, d_duxdx_rt, d_duxdz_rt,
						d_ux, d_uz, d_theta, d_omega, d_duzdx, d_duzdz, d_duxdx, d_duxdz, t,
						d_upx_next_up, d_upz_next_up, d_usx_next_up, d_usz_next_up, d_upx_now_up, d_upz_now_up, d_usx_now_up, d_usz_now_up,
						d_upx_next_dn, d_upz_next_dn, d_usx_next_dn, d_usz_next_dn, d_upx_now_dn, d_upz_now_dn, d_usx_now_dn, d_usz_now_dn,
						d_upx_next_lf, d_upz_next_lf, d_usx_next_lf, d_usz_next_lf, d_upx_now_lf, d_upz_now_lf, d_usx_now_lf, d_usz_now_lf,
						d_upx_next_rt, d_upz_next_rt, d_usx_next_rt, d_usz_next_rt, d_upx_now_rt, d_upz_now_rt, d_usx_now_rt, d_usz_now_rt,
						d_upx_next, d_upz_next, d_usx_next, d_usz_next, d_upx_now, d_upz_now, d_usx_now, d_usz_now);
				}
				
				
			}
			remove << <dimGrid, dimBlock >> > (Xn, Zn,shotx, shotz, receiver_depth, t0, dh, d_vp, d_recordux, d_recorduz);
			cudaMemcpy(record_ux, d_recordux, Xn * Tn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(record_uz, d_recorduz, Xn * Tn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(record_upx, d_recordupx, Xn * Tn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(record_upz, d_recordupz, Xn * Tn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(record_usx, d_recordusx, Xn * Tn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(record_usz, d_recordusz, Xn * Tn * sizeof(float), cudaMemcpyDeviceToHost);

			sprintf(filename, "./record/record_ux_%d_%d_%d.dat", xn, t, shotno + 1);
			write_bin1(record_ux, filename, Xn, Tn, pml);
			sprintf(filename, "./record/record_uz_%d_%d_%d.dat", xn, t, shotno + 1);
			write_bin1(record_uz, filename, Xn, Tn, pml);
		
			for (t = Tn - 1; t >= 0; t--)
			{
				
				
				load_record << <dimGrid, dimBlock >> > (Xn, Zn, pml, receiver_depth, d_ux_r, d_uz_r, d_recordux, d_recorduz, t);

				rt_s_res << <dimGrid, dimBlock >> > (Xn, Zn, d_ux_r, d_uz_r, d_theta_r, d_omega_r, d_duxdz_r, d_duzdz_r, d_duxdx_r, d_duzdx_r,
					d_ddx, d_ddz, d_vp, d_vs, F_xx_r, F_zz_r, F_xz_r, F_zx_r);

				rt_u_res << <dimGrid, dimBlock >> > (Xn, Zn, d_ux_r, d_uz_r, d_upx_next_r, d_upz_next_r, d_usx_next_r, d_usz_next_r, d_theta_r, d_omega_r, d_duxdz_r, d_duzdz_r, d_duxdx_r, d_duzdx_r,
					d_ddz, d_ddx, d_upx_now_r, d_upz_now_r, d_usx_now_r, d_usz_now_r,
					d_upx_past_r, d_upz_past_r, d_usx_past_r, d_usz_past_r, O_duz_xdz_r, O_duz_zdx_r, O_dux_zdx_r, O_dux_xdz_r,
					O_thetax_r, O_thetaz_r, O_omegaz_r, O_omegax_r);

				if (t == Tn - 1)
				{
					read_wavefiled_NT1 << <dimGrid, dimBlock >> > (Xn, Zn, pml, d_upx_past_s, d_upz_past_s, d_usx_past_s, d_usz_past_s,
						d_upx_next, d_upz_next, d_usx_next, d_usz_next);

				}
				if (t == Tn - 2)
				{
					read_wavefiled_NT2 << <dimGrid, dimBlock >> > (Xn, Zn, pml, d_upx_past_s, d_upz_past_s, d_usx_past_s, d_usz_past_s,
						d_upx_now, d_upz_now, d_usx_now, d_usz_now);
					

				}
				if (t == Tn - 3)
				{
					


					read_wavefiled_NT3 << <dimGrid, dimBlock >> > (Xn, Zn, pml, d_upx_next_s, d_upz_next_s, d_usx_next_s, d_usz_next_s,
						d_upx_next, d_upz_next, d_usx_next, d_usz_next, d_upx_now, d_upz_now, d_usx_now, d_usz_now,
						d_upx_now_s, d_upz_now_s, d_usx_now_s, d_usz_now_s,
						d_upx_past, d_upz_past, d_usx_past, d_usz_past,
						d_upx_past_s, d_upz_past_s, d_usx_past_s, d_usz_past_s);

					
				}


				if (t < Tn - 3) 
				{
					reshot_u << <dimGrid, dimBlock >> > (Xn, Zn, pml, d_ux_s, d_uz_s, d_upx_next_s, d_upz_next_s, d_usx_next_s, d_usz_next_s, d_theta_s, d_omega_s,
						d_duxdz_s, d_duzdz_s, d_duxdx_s, d_duzdx_s,
						d_upx_now_s, d_upz_now_s, d_usx_now_s, d_usz_now_s,
						d_upx_past_s, d_upz_past_s, d_usx_past_s, d_usz_past_s);

					read_wavefiled1 << <dimGrid, dimBlock >> > (t, pml, Xn, Zn, d_ux_up, d_uz_up, d_ux_dn, d_uz_dn, d_ux_lf, d_uz_lf, d_ux_rt, d_uz_rt, d_ux_s, d_uz_s);
					

					reshot_s << <dimGrid, dimBlock >> > (Xn, Zn, pml, d_ux_s, d_uz_s, d_theta_s, d_omega_s, d_duxdz_s, d_duzdz_s, d_duxdx_s, d_duzdx_s,
						d_vp, d_vs);

					read_wavefiled2 << <dimGrid, dimBlock >> > (d_theta_up, d_omega_up, d_duzdx_up, d_duzdz_up, d_duxdx_up, d_duxdz_up,
						d_theta_dn, d_omega_dn, d_duzdx_dn, d_duzdz_dn, d_duxdx_dn, d_duxdz_dn,
						d_theta_lf, d_omega_lf, d_duzdx_lf, d_duzdz_lf, d_duxdx_lf, d_duxdz_lf,
						d_theta_rt, d_omega_rt, d_duzdx_rt, d_duzdz_rt, d_duxdx_rt, d_duxdz_rt,
						d_theta_s, d_omega_s, d_duzdx_s, d_duzdz_s, d_duxdx_s, d_duxdz_s, t, pml, Xn, Zn);

					

				}
				

				poynting << <dimGrid, dimBlock >> > (Xn, Zn, pml,d_upx_next_r, d_upz_next_r, d_usx_next_r, d_usz_next_r, d_theta_r, d_omega_r, d_upx_past_s, d_upz_past_s, d_theta_s, d_omega_s,
					d_Epx_S, d_Epz_S, d_Epx_R, d_Epz_R, d_Esx_R, d_Esz_R, d_fenzi_PP_pyt, d_fenzi_PS_pyt, d_fenmu_P_pyt,					
					d_RR_upx_up, d_RR_upx_dn, d_RR_upx_lf, d_RR_upx_rt, d_RR_upz_up, d_RR_upz_dn, d_RR_upz_lf, d_RR_upz_rt,
					d_RR_usx_up, d_RR_usx_dn, d_RR_usx_lf, d_RR_usx_rt, d_RR_usz_up, d_RR_usz_dn, d_RR_usz_lf, d_RR_usz_rt,
					d_SS_upx_up, d_SS_upx_dn, d_SS_upx_lf, d_SS_upx_rt, d_SS_upz_up, d_SS_upz_dn, d_SS_upz_lf, d_SS_upz_rt);

				corr_v << <dimGrid, dimBlock >> > (Xn, Zn, pml,d_fenzi_PP, d_fenzi_PS, d_fenmu_P, d_upx_past_s, d_upz_past_s, d_upx_next_r, d_upz_next_r, d_usx_next_r, d_usz_next_r);
				
				if (t % 500 == 0) {
					printf("t=%d\n", t);
				}
			}
			image_fun << <dimGrid, dimBlock >> > (Xn, Zn, pml, d_fenzi_PP, d_fenmu_P, d_image_PP);
			image_fun << <dimGrid, dimBlock >> > (Xn, Zn, pml, d_fenzi_PS, d_fenmu_P, d_image_PS);

			cudaMemcpy(image_PP, d_image_PP, Xn * Zn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(image_PS, d_image_PS, Xn * Zn * sizeof(float), cudaMemcpyDeviceToHost);

			image_fun << <dimGrid, dimBlock >> > (Xn, Zn, pml,d_fenzi_PP_pyt, d_fenmu_P_pyt, d_image_PP_pyt);
			image_fun << <dimGrid, dimBlock >> > (Xn, Zn, pml,d_fenzi_PS_pyt, d_fenmu_P_pyt, d_image_PS_pyt);
			cudaMemcpy(image_PP_pyt, d_image_PP_pyt, Xn * Zn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(image_PS_pyt, d_image_PS_pyt, Xn * Zn * sizeof(float), cudaMemcpyDeviceToHost);

			Laplace << <dimGrid, dimBlock >> > (Xn, Zn, pml,dx, d_image_PP_pyt, d_image_PP_lap);
			Laplace << <dimGrid, dimBlock >> > (Xn, Zn, pml,dx, d_image_PS_pyt, d_image_PS_lap);
	
			cudaMemcpy(image_PP_lap, d_image_PP_lap, Xn * Zn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(image_PS_lap, d_image_PS_lap, Xn * Zn * sizeof(float), cudaMemcpyDeviceToHost);

	
			sprintf(filename, "./image2/image_PP_lap_%d_%d_%d.dat", xn, zn, shotno + 1);
			write_bin(image_PP_lap, filename, Xn, Zn, pml);
			
			sprintf(filename, "./image2/image_PS_lap_%d_%d_%d.dat", xn, zn, shotno + 1);
			write_bin(image_PS_lap, filename, Xn, Zn, pml);
			
		}
	
		
		for (shotno = 0; shotno < shot_num; shotno++)
		{
			sprintf(filename, "./image2/image_PP_lap_%d_%d_%d.dat", xn, zn, shotno + 1);
			if ((fp = fopen(filename, "rb")) != NULL)
			{
				float a = 0;
				for (i = pml; i < Xn - pml; i++)
				{
					for (j = pml; j < Zn - pml; j++)
					{
						fread(&image_PP_lap[i * Zn + j], sizeof(float), 1, fp);
					}
				}
			}fclose(fp);

			sprintf(filename, "./image2/image_PS_lap_%d_%d_%d.dat", xn, zn, shotno + 1);
			if ((fp = fopen(filename, "rb")) != NULL)
			{
				float a = 0;
				for (i = pml; i < Xn - pml; i++)
				{
					for (j = pml; j < Zn - pml; j++)
					{
						fread(&image_PS_lap[i * Zn + j], sizeof(float), 1, fp);
					}
				}
			}fclose(fp);


			for (i = pml; i < Xn - pml; i++)
			{
				for (j = pml; j < Zn - pml; j++)
				{
					All_image_PP_lap[i * Zn + j] += image_PP_lap[i * Zn + j];
					All_image_PS_lap[i * Zn + j] += image_PS_lap[i * Zn + j];
				}
			}



		}

		sprintf(filename, "./image2/All_image_PP_lap.dat");
		if ((fp = fopen(filename, "wb")) != NULL)
		{
			for (i = pml; i < Xn - pml; i++)
			{
				for (j = pml; j < Zn - pml; j++)
				{
					fwrite(&All_image_PP_lap[i * Zn + j], sizeof(float), 1, fp);
				}
			}
		}fclose(fp);
		sprintf(filename, "./image2/All_image_PS_lap.dat");
		if ((fp = fopen(filename, "wb")) != NULL)
		{
			for (i = pml; i < Xn - pml; i++)
			{
				for (j = pml; j < Zn - pml; j++)
				{
					fwrite(&All_image_PS_lap[i * Zn + j], sizeof(float), 1, fp);
				}
			}
		}fclose(fp);

		
	}
	
	free(vp); free(vs); free(rou); free(vp_pml); free(vs_pml); free(rou_pml); free(ddx); free(ddz); free(frontwaveux); free(frontwaveuz); 
	free(ux); free(uz); free(upx); free(upz); free(usx); free(usz); free(record_upx); free(record_upz); free(record_usx); free(record_usz); free(record_ux); free(record_uz);
	free(ux_s); free(uz_s); free(ux_r); free(uz_r); free(image_PP); free(image_PS); free(image_PP_lap); free(image_PS_lap); free(image_PP_pyt); free(image_PS_pyt); 
	free(All_image_PP); free(All_image_PS); free(All_image_PP_lap); free(All_image_PS_lap); free(All_image_PP_pyt); free(All_image_PS_pyt);

	cudaFree(d_ddx); cudaFree(d_ddz); cudaFree(d_vp); cudaFree(d_vs); cudaFree(d_upx_next); cudaFree(d_upx_now); cudaFree(d_upx_past); cudaFree(d_upz_next); cudaFree(d_upz_now); cudaFree(d_upz_past);
	cudaFree(d_usx_next); cudaFree(d_usx_now); cudaFree(d_usx_past);cudaFree(d_usz_next); cudaFree(d_usz_now); cudaFree(d_usz_past); cudaFree(d_recordupx); cudaFree(d_recordupz); cudaFree(d_recordusx); 
	cudaFree(d_recordusz); cudaFree(d_recordux); cudaFree(d_recorduz); cudaFree(O_duz_xdz); cudaFree(O_duz_zdx); cudaFree(O_dux_zdx); cudaFree(O_dux_xdz);
	cudaFree(O_thetax); cudaFree(O_thetaz); cudaFree(O_omegaz); cudaFree(O_omegax); cudaFree(F_xx); cudaFree(F_zz); cudaFree(F_xz); cudaFree(F_zx); cudaFree(d_ux); cudaFree(d_uz); 
	cudaFree(d_theta); cudaFree(d_omega); cudaFree(d_duzdx); cudaFree(d_duzdz); cudaFree(d_duxdx); cudaFree(d_duxdz); cudaFree(d_source); 
	cudaFree(d_ux_up); cudaFree(d_uz_up); cudaFree(d_theta_up); cudaFree(d_omega_up); cudaFree(d_duzdx_up); cudaFree(d_duzdz_up); cudaFree(d_duxdx_up); cudaFree(d_duxdz_up);
	cudaFree(d_ux_dn); cudaFree(d_uz_dn); cudaFree(d_theta_dn); cudaFree(d_omega_dn); cudaFree(d_duzdx_dn); cudaFree(d_duzdz_dn); cudaFree(d_duxdx_dn); cudaFree(d_duxdz_dn); 	
	cudaFree(d_ux_lf); cudaFree(d_uz_lf); cudaFree(d_theta_lf); cudaFree(d_omega_lf); cudaFree(d_duzdx_lf); cudaFree(d_duzdz_lf); cudaFree(d_duxdx_lf); cudaFree(d_duxdz_lf); 
	cudaFree(d_ux_rt); cudaFree(d_uz_rt); cudaFree(d_theta_rt); cudaFree(d_omega_rt); cudaFree(d_duzdx_rt); cudaFree(d_duzdz_rt); cudaFree(d_duxdx_rt); cudaFree(d_duxdz_rt); 
	cudaFree(d_upx_next_up); cudaFree(d_upz_next_up); cudaFree(d_usx_next_up); cudaFree(d_usz_next_up); cudaFree(d_upx_now_up); cudaFree(d_upz_now_up); cudaFree(d_usx_now_up); cudaFree(d_usz_now_up); 
	cudaFree(d_upx_next_dn); cudaFree(d_upz_next_dn); cudaFree(d_usx_next_dn); cudaFree(d_usz_next_dn); cudaFree(d_upx_now_dn);	cudaFree(d_upz_now_dn); cudaFree(d_usx_now_dn); cudaFree(d_usz_now_dn); 
	cudaFree(d_upx_next_lf); cudaFree(d_upz_next_lf); cudaFree(d_usx_next_lf); cudaFree(d_usz_next_lf); cudaFree(d_upx_now_lf); cudaFree(d_upz_now_lf); cudaFree(d_usx_now_lf); cudaFree(d_usz_now_lf); 
	cudaFree(d_upx_next_rt); cudaFree(d_upz_next_rt); cudaFree(d_usx_next_rt); cudaFree(d_usz_next_rt); cudaFree(d_upx_now_rt); cudaFree(d_upz_now_rt); cudaFree(d_usx_now_rt); cudaFree(d_usz_now_rt); 
	cudaFree(d_ux_s); cudaFree(d_uz_s); cudaFree(d_theta_s); cudaFree(d_omega_s); cudaFree(d_duzdx_s); cudaFree(d_duzdz_s); cudaFree(d_duxdz_s);cudaFree(d_duxdx_s); 
	cudaFree(d_upx_next_s); cudaFree(d_upz_next_s); cudaFree(d_usx_next_s); cudaFree(d_usz_next_s); cudaFree(d_upx_now_s); cudaFree(d_upz_now_s); cudaFree(d_usx_now_s); cudaFree(d_usz_now_s); 
	cudaFree(d_upx_past_s); cudaFree(d_upz_past_s); cudaFree(d_usx_past_s); cudaFree(d_usz_past_s);	cudaFree(d_ux_r); cudaFree(d_uz_r); cudaFree(d_theta_r); cudaFree(d_omega_r); 
	cudaFree(d_duxdz_r); cudaFree(d_duzdz_r); cudaFree(d_duxdx_r); cudaFree(d_duzdx_r); cudaFree(d_upx_next_r); cudaFree(d_upz_next_r); cudaFree(d_usx_next_r); cudaFree(d_usz_next_r); 
	cudaFree(d_upx_now_r); cudaFree(d_upz_now_r); cudaFree(d_usx_now_r); cudaFree(d_usz_now_r); cudaFree(d_upx_past_r); cudaFree(d_upz_past_r); cudaFree(d_usx_past_r); cudaFree(d_usz_past_r); 
	cudaFree(O_duz_xdz_r); cudaFree(O_duz_zdx_r); cudaFree(O_dux_zdx_r); cudaFree(O_dux_xdz_r); cudaFree(O_thetax_r); cudaFree(O_thetaz_r); cudaFree(O_omegaz_r); cudaFree(O_omegax_r); 
	cudaFree(F_xx_r); cudaFree(F_zz_r); cudaFree(F_xz_r); cudaFree(F_zx_r); cudaFree(d_Epx_S); cudaFree(d_Epz_S); cudaFree(d_Epx_R); cudaFree(d_Epz_R); cudaFree(d_Esx_R); cudaFree(d_Esz_R);
	cudaFree(d_fenzi_PP_pyt); cudaFree(d_fenzi_PS_pyt); cudaFree(d_fenmu_P_pyt); cudaFree(d_fenzi_PP); cudaFree(d_fenzi_PS); cudaFree(d_fenmu_P); 
	cudaFree(d_RR_upx_up); cudaFree(d_RR_upx_dn); cudaFree(d_RR_upx_lf); cudaFree(d_RR_upx_rt); cudaFree(d_RR_upz_up); cudaFree(d_RR_upz_dn); cudaFree(d_RR_upz_lf); cudaFree(d_RR_upz_rt); 
	cudaFree(d_RR_usx_up); cudaFree(d_RR_usx_dn); cudaFree(d_RR_usx_lf); cudaFree(d_RR_usx_rt); cudaFree(d_RR_usz_up); cudaFree(d_RR_usz_dn); cudaFree(d_RR_usz_lf); cudaFree(d_RR_usz_rt); 
	cudaFree(d_SS_upx_up); cudaFree(d_SS_upx_dn); cudaFree(d_SS_upx_lf); cudaFree(d_SS_upx_rt);	cudaFree(d_SS_upz_up); cudaFree(d_SS_upz_dn); cudaFree(d_SS_upz_lf); cudaFree(d_SS_upz_rt); 
	cudaFree(d_image_PP); cudaFree(d_image_PS); cudaFree(d_image_PP_lap); cudaFree(d_image_PS_lap); cudaFree(d_image_PP_pyt); cudaFree(d_image_PS_pyt); 


	return 0;
}
