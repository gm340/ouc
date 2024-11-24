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
#include "cuComplex.h"
#define PI 3.1415926
#define L 200
#define N 6
#define NT 6000
#define Xn 600
#define Zn 600
#define m 2
#define M 13
#define FM 30.0
#define R 1e-3
#define a_1 30
#define a_2 30
#define shot_num 40

void creatmodel(float* vp, float* vs, float* P)
{
	for (int i = L; i < Xn - L; i++){
		for (int j = L; j < Zn - L; j++){
			P[i * Zn + j] = 1;
		}
	}
}

void xiangbian(float* P, float* Vp, float* Vs)
{
	int i, j;
	
	for (i = L; i < Xn - L; i++)
		for (j = 0; j < L; j++)
		{

			P[i * Zn + j] = P[i * Zn + L];
			Vp[i * Zn + j] = Vp[i * Zn + L];
			Vs[i * Zn + j] = Vs[i * Zn + L];
		}

	
	for (i = L; i < Xn - L; i++)
		for (j = Zn - L; j < Zn; j++)
		{

			P[i * Zn + j] = P[i * Zn + Zn - L - 1];
			Vp[i * Zn + j] = Vp[i * Zn + Zn - L - 1];
			Vs[i * Zn + j] = Vs[i * Zn + Zn - L - 1];
		}

	
	for (i = 0; i < L; i++)
		for (j = L; j < Zn - L; j++)
		{

			P[i * Zn + j] = P[L * Zn + j];
			Vp[i * Zn + j] = Vp[L * Zn + j];
			Vs[i * Zn + j] = Vs[L * Zn + j];
		}

	
	for (i = Xn - L; i < Xn; i++)
		for (j = L; j < Zn - L; j++)
		{

			P[i * Zn + j] = P[(Xn - L - 1) * Zn + j];
			Vp[i * Zn + j] = Vp[(Xn - L - 1) * Zn + j];
			Vs[i * Zn + j] = Vs[(Xn - L - 1) * Zn + j];
		}

	
	for (i = 0; i < L; i++)
		for (j = 0; j < L; j++)
		{

			P[i * Zn + j] = P[L * Zn + L];
			Vp[i * Zn + j] = Vp[L * Zn + L];
			Vs[i * Zn + j] = Vs[L * Zn + L];
		}
	
	for (i = Xn - L; i < Xn; i++)
		for (j = 0; j < L; j++)
		{

			P[i * Zn + j] = P[(Xn - L - 1) * Zn + L];
			Vp[i * Zn + j] = Vp[(Xn - L - 1) * Zn + L];
			Vs[i * Zn + j] = Vs[(Xn - L - 1) * Zn + L];
		}

	
	for (i = 0; i < L; i++)
		for (j = Zn - L; j < Zn; j++)
		{

			P[i * Zn + j] = P[L * Zn + Zn - L - 1];
			Vp[i * Zn + j] = Vp[L * Zn + Zn - L - 1];
			Vs[i * Zn + j] = Vs[L * Zn + Zn - L - 1];
		}
	
	for (i = Xn - L; i < Xn; i++)
		for (j = Zn - L; j < Zn; j++)
		{

			P[i * Zn + j] = P[(Xn - L - 1) * Zn + Zn - L - 1];
			Vp[i * Zn + j] = Vp[(Xn - L - 1) * Zn + Zn - L - 1];
			Vs[i * Zn + j] = Vs[(Xn - L - 1) * Zn + Zn - L - 1];
		}


}
__global__
void revise_xiangbian(float* Vp, float* Vs)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	
	for (i = L; i < Xn - L; i++)
		for (j = 0; j < L; j++)
		{


			Vp[i * Zn + j] = Vp[i * Zn + L];
			Vs[i * Zn + j] = Vs[i * Zn + L];
		}

	
	for (i = L; i < Xn - L; i++)
		for (j = Zn - L; j < Zn; j++)
		{


			Vp[i * Zn + j] = Vp[i * Zn + Zn - L - 1];
			Vs[i * Zn + j] = Vs[i * Zn + Zn - L - 1];
		}

	
	for (i = 0; i < L; i++)
		for (j = L; j < Zn - L; j++)
		{


			Vp[i * Zn + j] = Vp[L * Zn + j];
			Vs[i * Zn + j] = Vs[L * Zn + j];
		}

	
	for (i = Xn - L; i < Xn; i++)
		for (j = L; j < Zn - L; j++)
		{


			Vp[i * Zn + j] = Vp[(Xn - L - 1) * Zn + j];
			Vs[i * Zn + j] = Vs[(Xn - L - 1) * Zn + j];
		}

	
	for (i = 0; i < L; i++)
		for (j = 0; j < L; j++)
		{


			Vp[i * Zn + j] = Vp[L * Zn + L];
			Vs[i * Zn + j] = Vs[L * Zn + L];
		}
	
	for (i = Xn - L; i < Xn; i++)
		for (j = 0; j < L; j++)
		{


			Vp[i * Zn + j] = Vp[(Xn - L - 1) * Zn + L];
			Vs[i * Zn + j] = Vs[(Xn - L - 1) * Zn + L];
		}

	
	for (i = 0; i < L; i++)
		for (j = Zn - L; j < Zn; j++)
		{


			Vp[i * Zn + j] = Vp[L * Zn + Zn - L - 1];
			Vs[i * Zn + j] = Vs[L * Zn + Zn - L - 1];
		}
	
	for (i = Xn - L; i < Xn; i++)
		for (j = Zn - L; j < Zn; j++)
		{


			Vp[i * Zn + j] = Vp[(Xn - L - 1) * Zn + Zn - L - 1];
			Vs[i * Zn + j] = Vs[(Xn - L - 1) * Zn + Zn - L - 1];
		}
}

__global__ void forward_u(float* Vx, float* Vz, float* Vpx, float* Vpz, float* Vsx, float* Vsz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* az, float* ax, float dt, float dx, float dz, float Z_receive, float* record_vx, float* record_vz, int t, float* Vpx_now, float* Vpz_now, float* Vsx_now, float* Vsz_now,
	float* Vpx_past, float* Vpz_past, float* Vsx_past, float* Vsz_past, float* O_duz_xdz, float* O_duz_zdx, float* O_dux_zdx, float* O_dux_xdz, float* O_thetax, float* O_thetaz, float* O_omegaz, float* O_omegax) {


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
	float a[N] = { 1.2213364, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };
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

		O_duz_xdz[i * Zn + j] = O_duz_xdz[i * Zn + j] + (az[i * Zn + j] * duz_xdz - az[i * Zn + j] * O_duz_xdz[i * Zn + j]) * dt;
		O_duz_zdx[i * Zn + j] = O_duz_zdx[i * Zn + j] + (ax[i * Zn + j] * duz_zdx - ax[i * Zn + j] * O_duz_zdx[i * Zn + j]) * dt;
		O_dux_zdx[i * Zn + j] = O_dux_zdx[i * Zn + j] + (ax[i * Zn + j] * dux_zdx - ax[i * Zn + j] * O_dux_zdx[i * Zn + j]) * dt;
		O_dux_xdz[i * Zn + j] = O_dux_xdz[i * Zn + j] + (az[i * Zn + j] * dux_xdz - az[i * Zn + j] * O_dux_xdz[i * Zn + j]) * dt;

		O_thetax[i * Zn + j] = O_thetax[i * Zn + j] + (ax[i * Zn + j] * dthetadx - ax[i * Zn + j] * O_thetax[i * Zn + j]) * dt;
		O_thetaz[i * Zn + j] = O_thetaz[i * Zn + j] + (az[i * Zn + j] * dthetadz - az[i * Zn + j] * O_thetaz[i * Zn + j]) * dt;
		O_omegaz[i * Zn + j] = O_omegaz[i * Zn + j] + (az[i * Zn + j] * domegadz - az[i * Zn + j] * O_omegaz[i * Zn + j]) * dt;
		O_omegax[i * Zn + j] = O_omegax[i * Zn + j] + (ax[i * Zn + j] * domegadx - ax[i * Zn + j] * O_omegax[i * Zn + j]) * dt;

		Vpx[i * Zn + j] = 2 * Vpx_now[i * Zn + j] - Vpx_past[i * Zn + j] + (dt * dt) * (dthetadx - O_thetax[i * Zn + j] + duz_xdz - duz_zdx - O_duz_xdz[i * Zn + j] + O_duz_zdx[i * Zn + j]);
		Vpz[i * Zn + j] = 2 * Vpz_now[i * Zn + j] - Vpz_past[i * Zn + j] + (dt * dt) * (dthetadz - O_thetaz[i * Zn + j] + dux_zdx - dux_xdz - O_dux_zdx[i * Zn + j] + O_dux_xdz[i * Zn + j]);
		Vsx[i * Zn + j] = 2 * Vsx_now[i * Zn + j] - Vsx_past[i * Zn + j] + (dt * dt) * (domegadz - O_omegaz[i * Zn + j] + duz_xdz - duz_zdx - O_duz_xdz[i * Zn + j] + O_duz_zdx[i * Zn + j]);
		Vsz[i * Zn + j] = 2 * Vsz_now[i * Zn + j] - Vsz_past[i * Zn + j] + (dt * dt) * (-domegadx + O_omegax[i * Zn + j] + dux_zdx - dux_xdz - O_dux_zdx[i * Zn + j] + O_dux_xdz[i * Zn + j]);

		Vpx_past[i * Zn + j] = Vpx_now[i * Zn + j];	Vpx_now[i * Zn + j] = Vpx[i * Zn + j];
		Vpz_past[i * Zn + j] = Vpz_now[i * Zn + j];	Vpz_now[i * Zn + j] = Vpz[i * Zn + j];
		Vsx_past[i * Zn + j] = Vsx_now[i * Zn + j];	Vsx_now[i * Zn + j] = Vsx[i * Zn + j];
		Vsz_past[i * Zn + j] = Vsz_now[i * Zn + j];	Vsz_now[i * Zn + j] = Vsz[i * Zn + j];


		Vx[i * Zn + j] = Vpx[i * Zn + j] + Vsx[i * Zn + j];
		Vz[i * Zn + j] = Vpz[i * Zn + j] + Vsz[i * Zn + j];

	}

	if (j = Z_receive)
	{
		record_vx[i * NT + t] = Vx[i * Zn + j];
		record_vz[i * NT + t] = Vz[i * Zn + j];
	}


}

__global__ void forward_s(float* Vx, float* Vz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* ax, float* az, float dt, float dx, float dz, int t, int Sx, int Sz, float* Vp, float* Vs, float* source,
	float* F_xx, float* F_zz, float* F_xz, float* F_zx) {
	
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dvxdx = 0.0f;
	float dvzdz = 0.0f;
	float dvxdz = 0.0f;
	float dvzdx = 0.0f;
	float a[N] = { 1.2213364, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };
	float s;
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{
		if (i == Sx && j == Sz)
		{
			s = source[t];
		}
		else
		{
			s = 0.0;
		}

		dvxdx = (a[0] * (Vx[(i + 0) * Zn + j] - Vx[(i - 1) * Zn + j])
			+ a[1] * (Vx[(i + 1) * Zn + j] - Vx[(i - 2) * Zn + j])
			+ a[2] * (Vx[(i + 2) * Zn + j] - Vx[(i - 3) * Zn + j])
			+ a[3] * (Vx[(i + 3) * Zn + j] - Vx[(i - 4) * Zn + j])
			+ a[4] * (Vx[(i + 4) * Zn + j] - Vx[(i - 5) * Zn + j])
			+ a[5] * (Vx[(i + 5) * Zn + j] - Vx[(i - 6) * Zn + j])) / dx;

		dvxdz = (a[0] * (Vx[(i)*Zn + j + 1] - Vx[(i)*Zn + j - 0])
			+ a[1] * (Vx[(i)*Zn + j + 2] - Vx[(i)*Zn + j - 1])
			+ a[2] * (Vx[(i)*Zn + j + 3] - Vx[(i)*Zn + j - 2])
			+ a[3] * (Vx[(i)*Zn + j + 4] - Vx[(i)*Zn + j - 3])
			+ a[4] * (Vx[(i)*Zn + j + 5] - Vx[(i)*Zn + j - 4])
			+ a[5] * (Vx[(i)*Zn + j + 6] - Vx[(i)*Zn + j - 5])) / dz;

		dvzdz = (a[0] * (Vz[(i)*Zn + j + 0] - Vz[(i)*Zn + j - 1])
			+ a[1] * (Vz[(i)*Zn + j + 1] - Vz[(i)*Zn + j - 2])
			+ a[2] * (Vz[(i)*Zn + j + 2] - Vz[(i)*Zn + j - 3])
			+ a[3] * (Vz[(i)*Zn + j + 3] - Vz[(i)*Zn + j - 4])
			+ a[4] * (Vz[(i)*Zn + j + 4] - Vz[(i)*Zn + j - 5])
			+ a[5] * (Vz[(i)*Zn + j + 5] - Vz[(i)*Zn + j - 6])) / dz;

		dvzdx = (a[0] * (Vz[(i + 1) * Zn + j] - Vz[(i - 0) * Zn + j])
			+ a[1] * (Vz[(i + 2) * Zn + j] - Vz[(i - 1) * Zn + j])
			+ a[2] * (Vz[(i + 3) * Zn + j] - Vz[(i - 2) * Zn + j])
			+ a[3] * (Vz[(i + 4) * Zn + j] - Vz[(i - 3) * Zn + j])
			+ a[4] * (Vz[(i + 5) * Zn + j] - Vz[(i - 4) * Zn + j])
			+ a[5] * (Vz[(i + 6) * Zn + j] - Vz[(i - 5) * Zn + j])) / dx;

		F_xx[i * Zn + j] = F_xx[i * Zn + j] + (ax[i * Zn + j] * dvxdx - ax[i * Zn + j] * F_xx[i * Zn + j]) * dt;
		F_zz[i * Zn + j] = F_zz[i * Zn + j] + (az[i * Zn + j] * dvzdz - az[i * Zn + j] * F_zz[i * Zn + j]) * dt;
		F_xz[i * Zn + j] = F_xz[i * Zn + j] + (az[i * Zn + j] * dvxdz - az[i * Zn + j] * F_xz[i * Zn + j]) * dt;
		F_zx[i * Zn + j] = F_zx[i * Zn + j] + (ax[i * Zn + j] * dvzdx - ax[i * Zn + j] * F_zx[i * Zn + j]) * dt;

		theta[i * Zn + j] = Vp[i * Zn + j] * Vp[i * Zn + j] * (dvxdx + dvzdz - F_xx[i * Zn + j] - F_zz[i * Zn + j]);
		omega[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdz - dvzdx - F_xz[i * Zn + j] + F_zx[i * Zn + j]);
		duzdx[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvzdx - F_zx[i * Zn + j]);
		duzdz[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvzdz - F_zz[i * Zn + j]);
		duxdz[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdz - F_xz[i * Zn + j]);
		duxdx[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdx - F_xx[i * Zn + j]);

		theta[i * Zn + j] = theta[i * Zn + j] + s;		

	}
	
}





__global__
void velocity( float* Vx, float* Vz, float* Vpx, float* Vpz, float* Vsx, float* Vsz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* az, float* ax, float dt, float dx, float dz, float Z_receive, float* record_vx, float* record_vz, int t)
{
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
	float a[N] = { +1.2213e+0,-9.6931e-2,1.7448e-2,-2.9673e-3,+3.5901e-4,-2.1848e-5 };
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{
		dthetadx= (a[0] * (theta[(i+1)*Zn + j] - theta[(i - 0) * Zn + j])
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




		Vpx[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * ((1 - 0.5 * dt * ax[i * Zn + j]) * Vpx[i * Zn + j] + dt * (dthetadx + duz_xdz - duz_zdx));

		Vpz[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * ((1 - 0.5 * dt * az[i * Zn + j]) * Vpz[i * Zn + j] + dt * (dthetadz + dux_zdx - dux_xdz));

		Vsx[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * ((1 - 0.5 * dt * ax[i * Zn + j]) * Vsx[i * Zn + j] + dt * ( domegadz + duz_xdz - duz_zdx));

		Vsz[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * ((1 - 0.5 * dt * az[i * Zn + j]) * Vsz[i * Zn + j] + dt * (-domegadx + dux_zdx - dux_xdz));


		Vx[i * Zn + j] = Vpx[i * Zn + j] + Vsx[i * Zn + j];
		Vz[i * Zn + j] = Vpz[i * Zn + j] + Vsz[i * Zn + j];

	}
	if (j = Z_receive)
	{
		record_vx[i * NT + t] = Vx[i * Zn + j];
		record_vz[i * NT + t] = Vz[i * Zn + j];
	}
}
__global__
void stress(float* P, float* Vx, float* Vz, float* theta, float* omega, float* theta_x, float* omega_x, float* theta_z, float* omega_z, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* ax,float* az, float dt, float dx, float dz, int t, int Sx, int Sz, float* Vp, float* Vs, float* source)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dvxdx = 0.0f;
	float dvzdz = 0.0f;
	float dvxdz = 0.0f;
	float dvzdx = 0.0f;
	float a[N] = { +1.2213e+0,-9.6931e-2,1.7448e-2,-2.9673e-3,+3.5901e-4,-2.1848e-5 };
	float s;
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{
		if (i == Sx && j == Sz)
		{
			s = source[t];
		}
		else
		{
			s = 0.0;
		}

		dvxdx = (a[0] * (Vx[(i + 0) * Zn + j] - Vx[(i - 1) * Zn + j])
			+ a[1] * (Vx[(i + 1) * Zn + j] - Vx[(i - 2) * Zn + j])
			+ a[2] * (Vx[(i + 2) * Zn + j] - Vx[(i - 3) * Zn + j])
			+ a[3] * (Vx[(i + 3) * Zn + j] - Vx[(i - 4) * Zn + j])
			+ a[4] * (Vx[(i + 4) * Zn + j] - Vx[(i - 5) * Zn + j])
			+ a[5] * (Vx[(i + 5) * Zn + j] - Vx[(i - 6) * Zn + j])) / dx;

		dvxdz = (a[0] * (Vx[(i)*Zn + j + 1] - Vx[(i)*Zn + j - 0])
			+ a[1] * (Vx[(i)*Zn + j + 2] - Vx[(i)*Zn + j - 1])
			+ a[2] * (Vx[(i)*Zn + j + 3] - Vx[(i)*Zn + j - 2])
			+ a[3] * (Vx[(i)*Zn + j + 4] - Vx[(i)*Zn + j - 3])
			+ a[4] * (Vx[(i)*Zn + j + 5] - Vx[(i)*Zn + j - 4])
			+ a[5] * (Vx[(i)*Zn + j + 6] - Vx[(i)*Zn + j - 5])) / dz;

		dvzdz= (a[0] * (Vz[(i)*Zn + j + 0] - Vz[(i)*Zn + j - 1])
			+ a[1] * (Vz[(i)*Zn + j + 1] - Vz[(i)*Zn + j - 2])
			+ a[2] * (Vz[(i)*Zn + j + 2] - Vz[(i)*Zn + j - 3])
			+ a[3] * (Vz[(i)*Zn + j + 3] - Vz[(i)*Zn + j - 4])
			+ a[4] * (Vz[(i)*Zn + j + 4] - Vz[(i)*Zn + j - 5])
			+ a[5] * (Vz[(i)*Zn + j + 5] - Vz[(i)*Zn + j - 6])) / dz;

		dvzdx = (a[0] * (Vz[(i + 1) * Zn + j] - Vz[(i - 0) * Zn + j])
			+ a[1] * (Vz[(i + 2) * Zn + j] - Vz[(i - 1) * Zn + j])
			+ a[2] * (Vz[(i + 3) * Zn + j] - Vz[(i - 2) * Zn + j])
			+ a[3] * (Vz[(i + 4) * Zn + j] - Vz[(i - 3) * Zn + j])
			+ a[4] * (Vz[(i + 5) * Zn + j] - Vz[(i - 4) * Zn + j])
			+ a[5] * (Vz[(i + 6) * Zn + j] - Vz[(i - 5) * Zn + j])) / dx;

		theta_x[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * (dt * powf(Vp[i * Zn + j], 2) * dvxdx + (1.0 - 0.5 * dt * ax[i * Zn + j]) * theta_x[i * Zn + j]);

		theta_z[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * (dt * powf(Vp[i * Zn + j], 2) * dvzdz + (1.0 - 0.5 * dt * az[i * Zn + j]) * theta_z[i * Zn + j]);
		////
		omega_x[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * (-dt * powf(Vs[i * Zn + j], 2) * dvzdx + (1.0 - 0.5 * dt * ax[i * Zn + j]) * omega_x[i * Zn + j]);

		omega_z[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvxdz + (1.0 - 0.5 * dt * az[i * Zn + j]) * omega_z[i * Zn + j]);
		////
		duxdx[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvxdx + (1.0 - 0.5 * dt * ax[i * Zn + j]) * duxdx[i * Zn + j]);

		duxdz[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvxdz + (1.0 - 0.5 * dt * az[i * Zn + j]) * duxdz[i * Zn + j]);

		duzdx[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvzdx + (1.0 - 0.5 * dt * ax[i * Zn + j]) * duzdx[i * Zn + j]);

		duzdz[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvzdz + (1.0 - 0.5 * dt * az[i * Zn + j]) * duzdz[i * Zn + j]);

		theta[i * Zn + j] = theta_x[i * Zn + j] + theta_z[i * Zn + j] + s;
		omega[i * Zn + j] = omega_x[i * Zn + j] + omega_z[i * Zn + j];

	}

}

__global__ void rt_u_res(float* Vx, float* Vz, float* Vpx, float* Vpz, float* Vsx, float* Vsz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* az, float* ax, float dt, float dx, float dz, int t, float* Vpx_now, float* Vpz_now, float* Vsx_now, float* Vsz_now,
	float* Vpx_past, float* Vpz_past, float* Vsx_past, float* Vsz_past, float* O_duz_xdz, float* O_duz_zdx, float* O_dux_zdx, float* O_dux_xdz, float* O_thetax, float* O_thetaz, float* O_omegaz, float* O_omegax) {


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
	float a[N] = { 1.2213364, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };
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

		O_duz_xdz[i * Zn + j] = O_duz_xdz[i * Zn + j] + (az[i * Zn + j] * duz_xdz - az[i * Zn + j] * O_duz_xdz[i * Zn + j]) * dt;
		O_duz_zdx[i * Zn + j] = O_duz_zdx[i * Zn + j] + (ax[i * Zn + j] * duz_zdx - ax[i * Zn + j] * O_duz_zdx[i * Zn + j]) * dt;
		O_dux_zdx[i * Zn + j] = O_dux_zdx[i * Zn + j] + (ax[i * Zn + j] * dux_zdx - ax[i * Zn + j] * O_dux_zdx[i * Zn + j]) * dt;
		O_dux_xdz[i * Zn + j] = O_dux_xdz[i * Zn + j] + (az[i * Zn + j] * dux_xdz - az[i * Zn + j] * O_dux_xdz[i * Zn + j]) * dt;

		O_thetax[i * Zn + j] = O_thetax[i * Zn + j] + (ax[i * Zn + j] * dthetadx - ax[i * Zn + j] * O_thetax[i * Zn + j]) * dt;
		O_thetaz[i * Zn + j] = O_thetaz[i * Zn + j] + (az[i * Zn + j] * dthetadz - az[i * Zn + j] * O_thetaz[i * Zn + j]) * dt;
		O_omegaz[i * Zn + j] = O_omegaz[i * Zn + j] + (az[i * Zn + j] * domegadz - az[i * Zn + j] * O_omegaz[i * Zn + j]) * dt;
		O_omegax[i * Zn + j] = O_omegax[i * Zn + j] + (ax[i * Zn + j] * domegadx - ax[i * Zn + j] * O_omegax[i * Zn + j]) * dt;

		Vpx[i * Zn + j] = 2 * Vpx_now[i * Zn + j] - Vpx_past[i * Zn + j] + (dt * dt) * (dthetadx - O_thetax[i * Zn + j] + duz_xdz - duz_zdx - O_duz_xdz[i * Zn + j] + O_duz_zdx[i * Zn + j]);
		Vpz[i * Zn + j] = 2 * Vpz_now[i * Zn + j] - Vpz_past[i * Zn + j] + (dt * dt) * (dthetadz - O_thetaz[i * Zn + j] + dux_zdx - dux_xdz - O_dux_zdx[i * Zn + j] + O_dux_xdz[i * Zn + j]);
		Vsx[i * Zn + j] = 2 * Vsx_now[i * Zn + j] - Vsx_past[i * Zn + j] + (dt * dt) * (domegadz - O_omegaz[i * Zn + j] + duz_xdz - duz_zdx - O_duz_xdz[i * Zn + j] + O_duz_zdx[i * Zn + j]);
		Vsz[i * Zn + j] = 2 * Vsz_now[i * Zn + j] - Vsz_past[i * Zn + j] + (dt * dt) * (-domegadx + O_omegax[i * Zn + j] + dux_zdx - dux_xdz - O_dux_zdx[i * Zn + j] + O_dux_xdz[i * Zn + j]);

		Vpx_past[i * Zn + j] = Vpx_now[i * Zn + j];	Vpx_now[i * Zn + j] = Vpx[i * Zn + j];
		Vpz_past[i * Zn + j] = Vpz_now[i * Zn + j];	Vpz_now[i * Zn + j] = Vpz[i * Zn + j];
		Vsx_past[i * Zn + j] = Vsx_now[i * Zn + j];	Vsx_now[i * Zn + j] = Vsx[i * Zn + j];
		Vsz_past[i * Zn + j] = Vsz_now[i * Zn + j];	Vsz_now[i * Zn + j] = Vsz[i * Zn + j];


		Vx[i * Zn + j] = Vpx[i * Zn + j] + Vsx[i * Zn + j];
		Vz[i * Zn + j] = Vpz[i * Zn + j] + Vsz[i * Zn + j];

	}


}

__global__ void rt_s_res(float* Vx, float* Vz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* ax, float* az, float dt, float dx, float dz, float* Vp, float* Vs, 
	float* F_xx, float* F_zz, float* F_xz, float* F_zx) {

	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dvxdx = 0.0f;
	float dvzdz = 0.0f;
	float dvxdz = 0.0f;
	float dvzdx = 0.0f;
	float a[N] = { 1.2213364, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };
	float s;
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{
		
		dvxdx = (a[0] * (Vx[(i + 0) * Zn + j] - Vx[(i - 1) * Zn + j])
			+ a[1] * (Vx[(i + 1) * Zn + j] - Vx[(i - 2) * Zn + j])
			+ a[2] * (Vx[(i + 2) * Zn + j] - Vx[(i - 3) * Zn + j])
			+ a[3] * (Vx[(i + 3) * Zn + j] - Vx[(i - 4) * Zn + j])
			+ a[4] * (Vx[(i + 4) * Zn + j] - Vx[(i - 5) * Zn + j])
			+ a[5] * (Vx[(i + 5) * Zn + j] - Vx[(i - 6) * Zn + j])) / dx;

		dvxdz = (a[0] * (Vx[(i)*Zn + j + 1] - Vx[(i)*Zn + j - 0])
			+ a[1] * (Vx[(i)*Zn + j + 2] - Vx[(i)*Zn + j - 1])
			+ a[2] * (Vx[(i)*Zn + j + 3] - Vx[(i)*Zn + j - 2])
			+ a[3] * (Vx[(i)*Zn + j + 4] - Vx[(i)*Zn + j - 3])
			+ a[4] * (Vx[(i)*Zn + j + 5] - Vx[(i)*Zn + j - 4])
			+ a[5] * (Vx[(i)*Zn + j + 6] - Vx[(i)*Zn + j - 5])) / dz;

		dvzdz = (a[0] * (Vz[(i)*Zn + j + 0] - Vz[(i)*Zn + j - 1])
			+ a[1] * (Vz[(i)*Zn + j + 1] - Vz[(i)*Zn + j - 2])
			+ a[2] * (Vz[(i)*Zn + j + 2] - Vz[(i)*Zn + j - 3])
			+ a[3] * (Vz[(i)*Zn + j + 3] - Vz[(i)*Zn + j - 4])
			+ a[4] * (Vz[(i)*Zn + j + 4] - Vz[(i)*Zn + j - 5])
			+ a[5] * (Vz[(i)*Zn + j + 5] - Vz[(i)*Zn + j - 6])) / dz;

		dvzdx = (a[0] * (Vz[(i + 1) * Zn + j] - Vz[(i - 0) * Zn + j])
			+ a[1] * (Vz[(i + 2) * Zn + j] - Vz[(i - 1) * Zn + j])
			+ a[2] * (Vz[(i + 3) * Zn + j] - Vz[(i - 2) * Zn + j])
			+ a[3] * (Vz[(i + 4) * Zn + j] - Vz[(i - 3) * Zn + j])
			+ a[4] * (Vz[(i + 5) * Zn + j] - Vz[(i - 4) * Zn + j])
			+ a[5] * (Vz[(i + 6) * Zn + j] - Vz[(i - 5) * Zn + j])) / dx;

		F_xx[i * Zn + j] = F_xx[i * Zn + j] + (ax[i * Zn + j] * dvxdx - ax[i * Zn + j] * F_xx[i * Zn + j]) * dt;
		F_zz[i * Zn + j] = F_zz[i * Zn + j] + (az[i * Zn + j] * dvzdz - az[i * Zn + j] * F_zz[i * Zn + j]) * dt;
		F_xz[i * Zn + j] = F_xz[i * Zn + j] + (az[i * Zn + j] * dvxdz - az[i * Zn + j] * F_xz[i * Zn + j]) * dt;
		F_zx[i * Zn + j] = F_zx[i * Zn + j] + (ax[i * Zn + j] * dvzdx - ax[i * Zn + j] * F_zx[i * Zn + j]) * dt;

		theta[i * Zn + j] = Vp[i * Zn + j] * Vp[i * Zn + j] * (dvxdx + dvzdz - F_xx[i * Zn + j] - F_zz[i * Zn + j]);
		omega[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdz - dvzdx - F_xz[i * Zn + j] + F_zx[i * Zn + j]);
		duzdx[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvzdx - F_zx[i * Zn + j]);
		duzdz[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvzdz - F_zz[i * Zn + j]);
		duxdz[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdz - F_xz[i * Zn + j]);
		duxdx[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdx - F_xx[i * Zn + j]);

		
	}

}

__global__
void velocity_backward_propagation(float* Vx, float* Vz, float* Vpx, float* Vpz, float* Vsx, float* Vsz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* az, float* ax, float dt, float dx, float dz)
{
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
	float a[N] = { +1.2213e+0,-9.6931e-2,1.7448e-2,-2.9673e-3,+3.5901e-4,-2.1848e-5 };
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




		Vpx[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * ((1 - 0.5 * dt * ax[i * Zn + j]) * Vpx[i * Zn + j] + dt * (dthetadx + duz_xdz - duz_zdx));

		Vpz[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * ((1 - 0.5 * dt * az[i * Zn + j]) * Vpz[i * Zn + j] + dt * (dthetadz + dux_zdx - dux_xdz));

		Vsx[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * ((1 - 0.5 * dt * ax[i * Zn + j]) * Vsx[i * Zn + j] + dt * (domegadz + duz_xdz - duz_zdx));

		Vsz[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * ((1 - 0.5 * dt * az[i * Zn + j]) * Vsz[i * Zn + j] + dt * (-domegadx + dux_zdx - dux_xdz));


		Vx[i * Zn + j] = Vpx[i * Zn + j] + Vsx[i * Zn + j];
		Vz[i * Zn + j] = Vpz[i * Zn + j] + Vsz[i * Zn + j];

	}
	
}
__global__
void stress_backward_propagation(float* P, float* Vx, float* Vz, float* theta, float* omega, float* theta_x, float* omega_x, float* theta_z, float* omega_z, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float* ax, float* az, float dt, float dx, float dz, int t, float* Vp, float* Vs)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dvxdx = 0.0f;
	float dvzdz = 0.0f;
	float dvxdz = 0.0f;
	float dvzdx = 0.0f;
	float a[N] = { +1.2213e+0,-9.6931e-2,1.7448e-2,-2.9673e-3,+3.5901e-4,-2.1848e-5 };
	float s;
	if (i >= N && i < Xn - N && j >= N && j < Zn - N)
	{
		

		dvxdx = (a[0] * (Vx[(i + 0) * Zn + j] - Vx[(i - 1) * Zn + j])
			+ a[1] * (Vx[(i + 1) * Zn + j] - Vx[(i - 2) * Zn + j])
			+ a[2] * (Vx[(i + 2) * Zn + j] - Vx[(i - 3) * Zn + j])
			+ a[3] * (Vx[(i + 3) * Zn + j] - Vx[(i - 4) * Zn + j])
			+ a[4] * (Vx[(i + 4) * Zn + j] - Vx[(i - 5) * Zn + j])
			+ a[5] * (Vx[(i + 5) * Zn + j] - Vx[(i - 6) * Zn + j])) / dx;

		dvxdz = (a[0] * (Vx[(i)*Zn + j + 1] - Vx[(i)*Zn + j - 0])
			+ a[1] * (Vx[(i)*Zn + j + 2] - Vx[(i)*Zn + j - 1])
			+ a[2] * (Vx[(i)*Zn + j + 3] - Vx[(i)*Zn + j - 2])
			+ a[3] * (Vx[(i)*Zn + j + 4] - Vx[(i)*Zn + j - 3])
			+ a[4] * (Vx[(i)*Zn + j + 5] - Vx[(i)*Zn + j - 4])
			+ a[5] * (Vx[(i)*Zn + j + 6] - Vx[(i)*Zn + j - 5])) / dz;

		dvzdz = (a[0] * (Vz[(i)*Zn + j + 0] - Vz[(i)*Zn + j - 1])
			+ a[1] * (Vz[(i)*Zn + j + 1] - Vz[(i)*Zn + j - 2])
			+ a[2] * (Vz[(i)*Zn + j + 2] - Vz[(i)*Zn + j - 3])
			+ a[3] * (Vz[(i)*Zn + j + 3] - Vz[(i)*Zn + j - 4])
			+ a[4] * (Vz[(i)*Zn + j + 4] - Vz[(i)*Zn + j - 5])
			+ a[5] * (Vz[(i)*Zn + j + 5] - Vz[(i)*Zn + j - 6])) / dz;

		dvzdx = (a[0] * (Vz[(i + 1) * Zn + j] - Vz[(i - 0) * Zn + j])
			+ a[1] * (Vz[(i + 2) * Zn + j] - Vz[(i - 1) * Zn + j])
			+ a[2] * (Vz[(i + 3) * Zn + j] - Vz[(i - 2) * Zn + j])
			+ a[3] * (Vz[(i + 4) * Zn + j] - Vz[(i - 3) * Zn + j])
			+ a[4] * (Vz[(i + 5) * Zn + j] - Vz[(i - 4) * Zn + j])
			+ a[5] * (Vz[(i + 6) * Zn + j] - Vz[(i - 5) * Zn + j])) / dx;

		theta_x[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * (dt * powf(Vp[i * Zn + j], 2) * dvxdx + (1.0 - 0.5 * dt * ax[i * Zn + j]) * theta_x[i * Zn + j]);

		theta_z[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * (dt * powf(Vp[i * Zn + j], 2) * dvzdz + (1.0 - 0.5 * dt * az[i * Zn + j]) * theta_z[i * Zn + j]);
		////
		omega_x[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * (-dt * powf(Vs[i * Zn + j], 2) * dvzdx + (1.0 - 0.5 * dt * ax[i * Zn + j]) * omega_x[i * Zn + j]);

		omega_z[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvxdz + (1.0 - 0.5 * dt * az[i * Zn + j]) * omega_z[i * Zn + j]);
		////
		duxdx[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvxdx + (1.0 - 0.5 * dt * ax[i * Zn + j]) * duxdx[i * Zn + j]);

		duxdz[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvxdz + (1.0 - 0.5 * dt * az[i * Zn + j]) * duxdz[i * Zn + j]);

		duzdx[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * ax[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvzdx + (1.0 - 0.5 * dt * ax[i * Zn + j]) * duzdx[i * Zn + j]);

		duzdz[i * Zn + j] = (1.0 / (1.0 + 0.5 * dt * az[i * Zn + j])) * (dt * powf(Vs[i * Zn + j], 2) * dvzdz + (1.0 - 0.5 * dt * az[i * Zn + j]) * duzdz[i * Zn + j]);

		theta[i * Zn + j] = theta_x[i * Zn + j] + theta_z[i * Zn + j];
		omega[i * Zn + j] = omega_x[i * Zn + j] + omega_z[i * Zn + j];

	}

}


__global__ void reshot_u(float* Vx, float* Vz, float* Vpx, float* Vpz, float* Vsx, float* Vsz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float dt, float dx, float dz, float* Vpx_now, float* Vpz_now, float* Vsx_now, float* Vsz_now,
	float* Vpx_past, float* Vpz_past, float* Vsx_past, float* Vsz_past) {
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
	float a[N] = { 1.2213364, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
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

		 Vpx[i * Zn + j] = Vpx_now[i * Zn + j];Vpx_now[i * Zn + j] = Vpx_past[i * Zn + j];
		 Vpz[i * Zn + j] = Vpz_now[i * Zn + j];Vpz_now[i * Zn + j] = Vpz_past[i * Zn + j];
		 Vsx[i * Zn + j] = Vsx_now[i * Zn + j];Vsx_now[i * Zn + j] = Vsx_past[i * Zn + j];
		 Vsz[i * Zn + j] = Vsz_now[i * Zn + j];Vsz_now[i * Zn + j] = Vsz_past[i * Zn + j];

		

		Vpx_past[i * Zn + j] = 2 * Vpx_now[i * Zn + j] - Vpx[i * Zn + j] + (dt * dt) * (dthetadx + duz_xdz - duz_zdx);
		Vpz_past[i * Zn + j] = 2 * Vpz_now[i * Zn + j] - Vpz[i * Zn + j] + (dt * dt) * (dthetadz + dux_zdx - dux_xdz);
		Vsx_past[i * Zn + j] = 2 * Vsx_now[i * Zn + j] - Vsx[i * Zn + j] + (dt * dt) * (domegadz + duz_xdz - duz_zdx);
		Vsz_past[i * Zn + j] = 2 * Vsz_now[i * Zn + j] - Vsz[i * Zn + j] + (dt * dt) * (-domegadx + dux_zdx - dux_xdz);



		Vx[i * Zn + j] = Vpx_past[i * Zn + j] + Vsx_past[i * Zn + j];
		Vz[i * Zn + j] = Vpz_past[i * Zn + j] + Vsz_past[i * Zn + j];

	}

}

__global__ void reshot_s(float* Vx, float* Vz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	float dt, float dx, float dz, float* Vp, float* Vs) {

	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dvxdx = 0.0f;
	float dvzdz = 0.0f;
	float dvxdz = 0.0f;
	float dvzdx = 0.0f;
	float a[N] = { 1.2213364, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };
	float s;
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{
		

		dvxdx = (a[0] * (Vx[(i + 0) * Zn + j] - Vx[(i - 1) * Zn + j])
			+ a[1] * (Vx[(i + 1) * Zn + j] - Vx[(i - 2) * Zn + j])
			+ a[2] * (Vx[(i + 2) * Zn + j] - Vx[(i - 3) * Zn + j])
			+ a[3] * (Vx[(i + 3) * Zn + j] - Vx[(i - 4) * Zn + j])
			+ a[4] * (Vx[(i + 4) * Zn + j] - Vx[(i - 5) * Zn + j])
			+ a[5] * (Vx[(i + 5) * Zn + j] - Vx[(i - 6) * Zn + j])) / dx;

		dvxdz = (a[0] * (Vx[(i)*Zn + j + 1] - Vx[(i)*Zn + j - 0])
			+ a[1] * (Vx[(i)*Zn + j + 2] - Vx[(i)*Zn + j - 1])
			+ a[2] * (Vx[(i)*Zn + j + 3] - Vx[(i)*Zn + j - 2])
			+ a[3] * (Vx[(i)*Zn + j + 4] - Vx[(i)*Zn + j - 3])
			+ a[4] * (Vx[(i)*Zn + j + 5] - Vx[(i)*Zn + j - 4])
			+ a[5] * (Vx[(i)*Zn + j + 6] - Vx[(i)*Zn + j - 5])) / dz;

		dvzdz = (a[0] * (Vz[(i)*Zn + j + 0] - Vz[(i)*Zn + j - 1])
			+ a[1] * (Vz[(i)*Zn + j + 1] - Vz[(i)*Zn + j - 2])
			+ a[2] * (Vz[(i)*Zn + j + 2] - Vz[(i)*Zn + j - 3])
			+ a[3] * (Vz[(i)*Zn + j + 3] - Vz[(i)*Zn + j - 4])
			+ a[4] * (Vz[(i)*Zn + j + 4] - Vz[(i)*Zn + j - 5])
			+ a[5] * (Vz[(i)*Zn + j + 5] - Vz[(i)*Zn + j - 6])) / dz;

		dvzdx = (a[0] * (Vz[(i + 1) * Zn + j] - Vz[(i - 0) * Zn + j])
			+ a[1] * (Vz[(i + 2) * Zn + j] - Vz[(i - 1) * Zn + j])
			+ a[2] * (Vz[(i + 3) * Zn + j] - Vz[(i - 2) * Zn + j])
			+ a[3] * (Vz[(i + 4) * Zn + j] - Vz[(i - 3) * Zn + j])
			+ a[4] * (Vz[(i + 5) * Zn + j] - Vz[(i - 4) * Zn + j])
			+ a[5] * (Vz[(i + 6) * Zn + j] - Vz[(i - 5) * Zn + j])) / dx;

		

		theta[i * Zn + j] = Vp[i * Zn + j] * Vp[i * Zn + j] * (dvxdx + dvzdz);
		omega[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdz - dvzdx);
		duzdx[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvzdx);
		duzdz[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvzdz);
		duxdz[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdz);
		duxdx[i * Zn + j] = Vs[i * Zn + j] * Vs[i * Zn + j] * (dvxdx);

		

	}

}
__global__ void PLACE_V( float* Vpx, float* Vpz, float* Vsx, float* Vsz, 
	float* Vpx_now, float* Vpz_now, float* Vsx_now, float* Vsz_now,
	float* Vpx_past, float* Vpz_past, float* Vsx_past, float* Vsz_past) {


	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{
	
		Vpx_past[i * Zn + j] = Vpx_now[i * Zn + j];
		Vpz_past[i * Zn + j] = Vpz_now[i * Zn + j];
		Vsx_past[i * Zn + j] = Vsx_now[i * Zn + j];
		Vsz_past[i * Zn + j] = Vsz_now[i * Zn + j];

		Vpx_now[i * Zn + j] = Vpx[i * Zn + j];
		Vpz_now[i * Zn + j] = Vpz[i * Zn + j];
		Vsx_now[i * Zn + j] = Vsx[i * Zn + j];
		Vsz_now[i * Zn + j] = Vsz[i * Zn + j];

		

	}



}

__global__
void velocity_wavefield_reconstruction(float* Vx, float* Vz, float* Vpx, float* Vpz, float* Vsx, float* Vsz, float* theta, float* omega, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	 float dt, float dx, float dz)
{
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
	float a[N] = { +1.2213e+0,-9.6931e-2,1.7448e-2,-2.9673e-3,+3.5901e-4,-2.1848e-5 };
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
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




		Vpx[i * Zn + j] = Vpx[i * Zn + j] - dt * (dthetadx + duz_xdz - duz_zdx);

		Vpz[i * Zn + j] = Vpz[i * Zn + j] - dt * (dthetadz + dux_zdx - dux_xdz);

		Vsx[i * Zn + j] = Vsx[i * Zn + j] - dt * (domegadz + duz_xdz - duz_zdx);

		Vsz[i * Zn + j] = Vsz[i * Zn + j] - dt * (-domegadx + dux_zdx - dux_xdz);


		Vx[i * Zn + j] = Vpx[i * Zn + j] + Vsx[i * Zn + j];
		Vz[i * Zn + j] = Vpz[i * Zn + j] + Vsz[i * Zn + j];

	}
	
}
__global__
void stress_wavefield_reconstruction(float* Vx, float* Vz, float* theta, float* omega, float* theta_x, float* omega_x, float* theta_z, float* omega_z, float* duxdz, float* duzdz, float* duxdx, float* duzdx,
	 float dt, float dx, float dz, float* Vp, float* Vs)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float dvxdx = 0.0f;
	float dvzdz = 0.0f;
	float dvxdz = 0.0f;
	float dvzdx = 0.0f;
	float a[N] = { +1.2213e+0,-9.6931e-2,1.7448e-2,-2.9673e-3,+3.5901e-4,-2.1848e-5 };
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{

		dvxdx = (a[0] * (Vx[(i + 0) * Zn + j] - Vx[(i - 1) * Zn + j])
			+ a[1] * (Vx[(i + 1) * Zn + j] - Vx[(i - 2) * Zn + j])
			+ a[2] * (Vx[(i + 2) * Zn + j] - Vx[(i - 3) * Zn + j])
			+ a[3] * (Vx[(i + 3) * Zn + j] - Vx[(i - 4) * Zn + j])
			+ a[4] * (Vx[(i + 4) * Zn + j] - Vx[(i - 5) * Zn + j])
			+ a[5] * (Vx[(i + 5) * Zn + j] - Vx[(i - 6) * Zn + j])) / dx;

		dvxdz = (a[0] * (Vx[(i)*Zn + j + 1] - Vx[(i)*Zn + j - 0])
			+ a[1] * (Vx[(i)*Zn + j + 2] - Vx[(i)*Zn + j - 1])
			+ a[2] * (Vx[(i)*Zn + j + 3] - Vx[(i)*Zn + j - 2])
			+ a[3] * (Vx[(i)*Zn + j + 4] - Vx[(i)*Zn + j - 3])
			+ a[4] * (Vx[(i)*Zn + j + 5] - Vx[(i)*Zn + j - 4])
			+ a[5] * (Vx[(i)*Zn + j + 6] - Vx[(i)*Zn + j - 5])) / dz;

		dvzdz = (a[0] * (Vz[(i)*Zn + j + 0] - Vz[(i)*Zn + j - 1])
			+ a[1] * (Vz[(i)*Zn + j + 1] - Vz[(i)*Zn + j - 2])
			+ a[2] * (Vz[(i)*Zn + j + 2] - Vz[(i)*Zn + j - 3])
			+ a[3] * (Vz[(i)*Zn + j + 3] - Vz[(i)*Zn + j - 4])
			+ a[4] * (Vz[(i)*Zn + j + 4] - Vz[(i)*Zn + j - 5])
			+ a[5] * (Vz[(i)*Zn + j + 5] - Vz[(i)*Zn + j - 6])) / dz;

		dvzdx = (a[0] * (Vz[(i + 1) * Zn + j] - Vz[(i - 0) * Zn + j])
			+ a[1] * (Vz[(i + 2) * Zn + j] - Vz[(i - 1) * Zn + j])
			+ a[2] * (Vz[(i + 3) * Zn + j] - Vz[(i - 2) * Zn + j])
			+ a[3] * (Vz[(i + 4) * Zn + j] - Vz[(i - 3) * Zn + j])
			+ a[4] * (Vz[(i + 5) * Zn + j] - Vz[(i - 4) * Zn + j])
			+ a[5] * (Vz[(i + 6) * Zn + j] - Vz[(i - 5) * Zn + j])) / dx;

		theta_x[i * Zn + j] = -dt * powf(Vp[i * Zn + j], 2) * dvxdx + theta_x[i * Zn + j];

		theta_z[i * Zn + j] = -dt * powf(Vp[i * Zn + j], 2) * dvzdz + theta_z[i * Zn + j];
		////
		omega_x[i * Zn + j] = dt * powf(Vs[i * Zn + j], 2) * dvzdx + omega_x[i * Zn + j];

		omega_z[i * Zn + j] = -dt * powf(Vs[i * Zn + j], 2) * dvxdz + omega_z[i * Zn + j];
		////
		duxdx[i * Zn + j] = -dt * powf(Vs[i * Zn + j], 2) * dvxdx + duxdx[i * Zn + j];

		duxdz[i * Zn + j] = -dt * powf(Vs[i * Zn + j], 2) * dvxdz + duxdz[i * Zn + j];

		duzdx[i * Zn + j] = -dt * powf(Vs[i * Zn + j], 2) * dvzdx + duzdx[i * Zn + j];

		duzdz[i * Zn + j] = -dt * powf(Vs[i * Zn + j], 2) * dvzdz + duzdz[i * Zn + j];

		theta[i * Zn + j] = theta_x[i * Zn + j] + theta_z[i * Zn + j];
		omega[i * Zn + j] = omega_x[i * Zn + j] + omega_z[i * Zn + j];

	}

}
__global__
void save_wavefiled(float* d_vx_up, float* d_vz_up, float* d_theta_up, float* d_omega_up, float* d_duzdx_up, float* d_duzdz_up, float* d_duxdx_up, float* d_duxdz_up,
	float* d_vx_dn, float* d_vz_dn, float* d_theta_dn, float* d_omega_dn, float* d_duzdx_dn, float* d_duzdz_dn, float* d_duxdx_dn, float* d_duxdz_dn,
	float* d_vx_lf, float* d_vz_lf, float* d_theta_lf, float* d_omega_lf, float* d_duzdx_lf, float* d_duzdz_lf, float* d_duxdx_lf, float* d_duxdz_lf,
	float* d_vx_rt, float* d_vz_rt, float* d_theta_rt, float* d_omega_rt, float* d_duzdx_rt, float* d_duzdz_rt, float* d_duxdx_rt, float* d_duxdz_rt, 
	float* d_vx, float* d_vz, float* d_theta, float* d_omega, float* d_duzdx, float* d_duzdz, float* d_duxdx, float* d_duxdz, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= 0 && i < Xn && j >= L - N && j < L + N)
	{
		d_vx_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_vx[i * Zn + j];
		d_vz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_vz[i * Zn + j];
		d_theta_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_theta[i * Zn + j];
		d_omega_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_omega[i * Zn + j];
		d_duzdx_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_duzdx[i * Zn + j];
		d_duzdz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_duzdz[i * Zn + j];
		d_duxdx_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_duxdx[i * Zn + j];
		d_duxdz_up[t * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)] = d_duxdz[i * Zn + j];
	}
	if (i >= 0 && i < Xn && j >= Zn - L - N && j < Zn - L + N)
	{
		d_vx_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_vx[i * Zn + j];
		d_vz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_vz[i * Zn + j];
		d_theta_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_theta[i * Zn + j];
		d_omega_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_omega[i * Zn + j];
		d_duzdx_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_duzdx[i * Zn + j];
		d_duzdz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_duzdz[i * Zn + j];
		d_duxdx_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_duxdx[i * Zn + j];
		d_duxdz_dn[t * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)] = d_duxdz[i * Zn + j];
	}
	if (i >= L - N && i < L + N && j >= 0 && j < Zn)
	{
		d_vx_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_vx[i * Zn + j];
		d_vz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_vz[i * Zn + j];
		d_theta_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_theta[i * Zn + j];
		d_omega_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_omega[i * Zn + j];
		d_duzdx_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_duzdx[i * Zn + j];
		d_duzdz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_duzdz[i * Zn + j];
		d_duxdx_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_duxdx[i * Zn + j];
		d_duxdz_lf[t * (2 * N * Zn) + (i - L + 2 * N) * Zn + j] = d_duxdz[i * Zn + j];

	}
	if (i >= Xn - L - N && i < Xn - L + N && j >= 0 && j < Zn)
	{
		d_vx_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_vx[i * Zn + j];
		d_vz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_vz[i * Zn + j];
		d_theta_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_theta[i * Zn + j];
		d_omega_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_omega[i * Zn + j];
		d_duzdx_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_duzdx[i * Zn + j];
		d_duzdz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_duzdz[i * Zn + j];
		d_duxdx_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_duxdx[i * Zn + j];
		d_duxdz_rt[t * (2 * N * Zn) + (i - Xn + L) * Zn + j] = d_duxdz[i * Zn + j];

	}

}
__global__
void read_last_wavefiled(float* d_theta_s, float* d_omega_s, float* d_Vx_s, float* d_Vz_s, float* d_duzdx_s, float* d_duzdz_s, float* d_duxdz_s, float* d_duxdx_s,
	float* d_theta_x_s, float* d_omega_x_s, float* d_theta_z_s, float* d_omega_z_s, float* d_Vpx_s, float* d_Vpz_s, float* d_Vsx_s, float* d_Vsz_s,
	float* d_theta, float* d_omega, float* d_Vx, float* d_Vz, float* d_duzdx, float* d_duzdz, float* d_duxdz, float* d_duxdx,
	float* d_theta_x, float* d_omega_x, float* d_theta_z, float* d_omega_z, float* d_Vpx, float* d_Vpz, float* d_Vsx, float* d_Vsz, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L - N && i < Xn - L + N && j >= L - N && j < Zn - L + N)
	{
		d_theta_s[i * Zn + j] = d_theta[i * Zn + j];
		d_omega_s[i * Zn + j] = d_omega[i * Zn + j];
		d_Vx_s[i * Zn + j] = d_Vx[i * Zn + j];
		d_Vz_s[i * Zn + j] = d_Vz[i * Zn + j];
		d_theta_x_s[i * Zn + j] = d_theta_x[i * Zn + j];
		d_omega_x_s[i * Zn + j] = d_omega_x[i * Zn + j];
		d_theta_z_s[i * Zn + j] = d_theta_z[i * Zn + j];
		d_omega_z_s[i * Zn + j] = d_omega_z[i * Zn + j];
		d_Vpx_s[i * Zn + j] = d_Vpx[i * Zn + j];
		d_Vpz_s[i * Zn + j] = d_Vpz[i * Zn + j];
		d_Vsx_s[i * Zn + j] = d_Vsx[i * Zn + j];
		d_Vsz_s[i * Zn + j] = d_Vsz[i * Zn + j];
		d_duzdx_s[i * Zn + j] = d_duzdx[i * Zn + j];
		d_duzdz_s[i * Zn + j] = d_duzdz[i * Zn + j];
		d_duxdx_s[i * Zn + j] = d_duxdx[i * Zn + j];
		d_duxdz_s[i * Zn + j] = d_duxdz[i * Zn + j];
	}
}

__global__ void read_last_wavefiled2(float* d_theta_s, float* d_omega_s, float* d_Vx_s, float* d_Vz_s, float* d_duzdx_s, float* d_duzdz_s, float* d_duxdz_s, float* d_duxdx_s,
	float* d_Vpx_s, float* d_Vpz_s, float* d_Vsx_s, float* d_Vsz_s,
	float* d_theta, float* d_omega, float* d_Vx, float* d_Vz, float* d_duzdx, float* d_duzdz, float* d_duxdz, float* d_duxdx,
	float* d_Vpx, float* d_Vpz, float* d_Vsx, float* d_Vsz, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L - N && i < Xn - L + N && j >= L - N && j < Zn - L + N)
	{
		d_theta_s[i * Zn + j] = d_theta[i * Zn + j];
		d_omega_s[i * Zn + j] = d_omega[i * Zn + j];
		d_Vx_s[i * Zn + j] = d_Vx[i * Zn + j];
		d_Vz_s[i * Zn + j] = d_Vz[i * Zn + j];
		
		d_Vpx_s[i * Zn + j] = d_Vpx[i * Zn + j];
		d_Vpz_s[i * Zn + j] = d_Vpz[i * Zn + j];
		d_Vsx_s[i * Zn + j] = d_Vsx[i * Zn + j];
		d_Vsz_s[i * Zn + j] = d_Vsz[i * Zn + j];
		d_duzdx_s[i * Zn + j] = d_duzdx[i * Zn + j];
		d_duzdz_s[i * Zn + j] = d_duzdz[i * Zn + j];
		d_duxdx_s[i * Zn + j] = d_duxdx[i * Zn + j];
		d_duxdz_s[i * Zn + j] = d_duxdz[i * Zn + j];
	}
}

__global__
void read_wavefiled1(float* d_Vx_up, float* d_Vz_up, float* d_Vx_dn, float* d_Vz_dn, float* d_Vx_lf, float* d_Vz_lf, float* d_Vx_rt, float* d_Vz_rt, 
	float* d_Vx, float* d_Vz, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= 0 && i < Xn && j >= L - N && j < L + N)
	{

		d_Vz[i * Zn + j] = d_Vz_up[(t) * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_Vx[i * Zn + j] = d_Vx_up[(t) * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
	}
	if (i >= 0 && i < Xn && j >= Zn - L - N && j < Zn - L + N)
	{

		d_Vz[i * Zn + j] = d_Vz_dn[(t) * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_Vx[i * Zn + j] = d_Vx_dn[(t) * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
	}
	if (i >= L - N && i < L + N && j >= 0 && j < Zn)
	{

		d_Vz[i * Zn + j] = d_Vz_lf[(t) * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_Vx[i * Zn + j] = d_Vx_lf[(t) * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
	}
	if (i >= Xn - L - N && i < Xn - L + N && j >= 0 && j < Zn)
	{

		d_Vz[i * Zn + j] = d_Vz_rt[(t) * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_Vx[i * Zn + j] = d_Vx_rt[(t) * (2 * N * Zn) + (i - Xn + L) * Zn + j];
	}

}
__global__
void read_wavefiled2(float* d_theta_up, float* d_omega_up, float* d_duxdz_up, float* d_duxdx_up, float* d_duzdz_up, float* d_duzdx_up,
	float* d_theta_dn, float* d_omega_dn, float* d_duxdz_dn, float* d_duxdx_dn, float* d_duzdz_dn, float* d_duzdx_dn,
	float* d_theta_lf, float* d_omega_lf, float* d_duxdz_lf, float* d_duxdx_lf, float* d_duzdz_lf, float* d_duzdx_lf,
	float* d_theta_rt, float* d_omega_rt, float* d_duxdz_rt, float* d_duxdx_rt, float* d_duzdz_rt, float* d_duzdx_rt,
	float* d_theta, float* d_omega, float* d_duxdz, float* d_duxdx, float* d_duzdz, float* d_duzdx, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= 0 && i < Xn && j >= L - N && j < L + N)
	{
		d_theta[i * Zn + j] = d_theta_up[(t) * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_omega[i * Zn + j] = d_omega_up[(t) * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_duxdz[i * Zn + j] = d_duxdz_up[(t) * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_duxdx[i * Zn + j] = d_duxdx_up[(t) * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_duzdz[i * Zn + j] = d_duzdz_up[(t) * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
		d_duzdx[i * Zn + j] = d_duzdx_up[(t) * (2 * N * Xn) + i * (2 * N) + j - (L - 2 * N)];
	}
	if (i >= 0 && i < Xn && j >= Zn - L - N && j < Zn - L + N)
	{
		d_theta[i * Zn + j] = d_theta_dn[(t) * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_omega[i * Zn + j] = d_omega_dn[(t) * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_duxdz[i * Zn + j] = d_duxdz_dn[(t) * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_duxdx[i * Zn + j] = d_duxdx_dn[(t) * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_duzdz[i * Zn + j] = d_duzdz_dn[(t) * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
		d_duzdx[i * Zn + j] = d_duzdx_dn[(t) * (2 * N * Xn) + i * (2 * N) + j - (Zn - L)];
	}
	if (i >= L - N && i < L + N && j >= 0 && j < Zn)
	{
		d_theta[i * Zn + j] = d_theta_lf[(t) * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_omega[i * Zn + j] = d_omega_lf[(t) * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_duxdz[i * Zn + j] = d_duxdz_lf[(t) * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_duxdx[i * Zn + j] = d_duxdx_lf[(t) * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_duzdz[i * Zn + j] = d_duzdz_lf[(t) * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
		d_duzdx[i * Zn + j] = d_duzdx_lf[(t) * (2 * N * Zn) + (i - L + 2 * N) * Zn + j];
	}
	if (i >= Xn - L - N && i < Xn - L + N && j >= 0 && j < Zn)
	{
		d_theta[i * Zn + j] = d_theta_rt[(t) * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_omega[i * Zn + j] = d_omega_rt[(t) * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_duxdz[i * Zn + j] = d_duxdz_rt[(t) * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_duxdx[i * Zn + j] = d_duxdx_rt[(t) * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_duzdz[i * Zn + j] = d_duzdz_rt[(t) * (2 * N * Zn) + (i - Xn + L) * Zn + j];
		d_duzdx[i * Zn + j] = d_duzdx_rt[(t) * (2 * N * Zn) + (i - Xn + L) * Zn + j];
	}

}

__global__ void corr_v(float* fenzi_PP, float* fenzi_PS, float* fenmu_P, float* SS_Px, float* SS_Pz, float* PP_Px, float* PP_Pz, float* PP_Sx, float* PP_Sz)
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

__global__ void image_fun(float* fenzi, float* fenmu, float* image)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{
		//	if (abs(1.0*(i - X0) / (j - Z0)) < aperture)                           //偏移孔径
		image[i * Zn + j] = fenzi[i * Zn + j] / fenmu[i * Zn + j];
		/*	else
		image[i*Zn + j] = 0.0;
		*/
	}
}

__global__ void Laplace(float dh, float* image, float* image_lap)
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

__global__
void load_record(int reciver, float* vx, float* vz, float* record_vx, float* record_vz, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= L && i < Xn - L && j == reciver)
	{
		vx[i * Zn + j] = record_vx[i * NT + t];
		vz[i * Zn + j] = record_vz[i * NT + t];
		
	}
}
float adsource(float* res_recordux, float* res_recorduz, float* cur_recordux, float* cur_recorduz, float* ori_recordux, float* ori_recorduz, int t) {

	int i, j, k;
	if (i >= L && i < Xn - L && j >= 0 && j < NT)
	{
			res_recordux[i * NT + t] = cur_recordux[i * NT + t] - ori_recordux[i * NT + t];
			res_recorduz[i * NT + t] = cur_recorduz[i * NT + t] - ori_recorduz[i * NT + t];
		
	}

		

	

}

__global__
void revise_model_end(float* Vp, float* Vs, float* Vp_end, float* Vs_end, float* Grad_Vp, float* Grad_Vs, int k)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float s;
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{


		Vp_end[i * Zn + j] = Vp[i * Zn + j] - a_2 * pow(0.95, k) * Grad_Vp[i * Zn + j];
		Vs_end[i * Zn + j] = Vs[i * Zn + j] - a_2 * pow(0.95, k) * Grad_Vs[i * Zn + j];



	}
}
__global__
void gradfun(float* Grad_Vp, float* Grad_Vs, float* Unx_s, float* Unz_s, float* theta_r, float* omega_r, float* dux_z_r, float* dux_x_r, float* duz_z_r, float* duz_x_r, float* SSS, float* Vp, float* Vs, float* P, float dt, float dx, float dz, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float duxdx = 0.0f;
	float duzdz = 0.0f;
	float duzdx = 0.0f;
	float duxdz = 0.0f;
	float druxdx = 0.0f;
	float druzdz = 0.0f;
	float druzdx = 0.0f;
	float druxdz = 0.0f;
	float a[N] = { 1.2213364, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{
		duxdx = (a[0] * (Unx_s[(i)*Zn + j] - Unx_s[(i - 1) * Zn + j])
			+ a[1] * (Unx_s[(i + 1) * Zn + j] - Unx_s[(i - 2) * Zn + j])
			+ a[2] * (Unx_s[(i + 2) * Zn + j] - Unx_s[(i - 3) * Zn + j])
			+ a[3] * (Unx_s[(i + 3) * Zn + j] - Unx_s[(i - 4) * Zn + j])
			+ a[4] * (Unx_s[(i + 4) * Zn + j] - Unx_s[(i - 5) * Zn + j])
			+ a[5] * (Unx_s[(i + 5) * Zn + j] - Unx_s[(i - 6) * Zn + j])) / dx;
		duzdz = (a[0] * (Unz_s[(i)*Zn + j] - Unz_s[(i)*Zn + j - 1])
			+ a[1] * (Unz_s[(i)*Zn + j + 1] - Unz_s[(i)*Zn + j - 2])
			+ a[2] * (Unz_s[(i)*Zn + j + 2] - Unz_s[(i)*Zn + j - 3])
			+ a[3] * (Unz_s[(i)*Zn + j + 3] - Unz_s[(i)*Zn + j - 4])
			+ a[4] * (Unz_s[(i)*Zn + j + 4] - Unz_s[(i)*Zn + j - 5])
			+ a[5] * (Unz_s[(i)*Zn + j + 5] - Unz_s[(i)*Zn + j - 6])) / dz;
		duzdx = (a[0] * (Unz_s[(i + 1) * Zn + j] - Unz_s[(i)*Zn + j])
			+ a[1] * (Unz_s[(i + 2) * Zn + j] - Unz_s[(i - 1) * Zn + j])
			+ a[2] * (Unz_s[(i + 3) * Zn + j] - Unz_s[(i - 2) * Zn + j])
			+ a[3] * (Unz_s[(i + 4) * Zn + j] - Unz_s[(i - 3) * Zn + j])
			+ a[4] * (Unz_s[(i + 5) * Zn + j] - Unz_s[(i - 4) * Zn + j])
			+ a[5] * (Unz_s[(i + 6) * Zn + j] - Unz_s[(i - 5) * Zn + j])) / dx;
		duxdz = (a[0] * (Unx_s[(i)*Zn + j + 1] - Unx_s[(i)*Zn + j])
			+ a[1] * (Unx_s[(i)*Zn + j + 2] - Unx_s[(i)*Zn + j - 1])
			+ a[2] * (Unx_s[(i)*Zn + j + 3] - Unx_s[(i)*Zn + j - 2])
			+ a[3] * (Unx_s[(i)*Zn + j + 4] - Unx_s[(i)*Zn + j - 6])
			+ a[4] * (Unx_s[(i)*Zn + j + 5] - Unx_s[(i)*Zn + j - 4])
			+ a[5] * (Unx_s[(i)*Zn + j + 6] - Unx_s[(i)*Zn + j - 5])) / dz;




		Grad_Vp[i * Zn + j] -= 2 * P[i * Zn + j] * Vp[i * Zn + j] * ((duxdx + duzdz) * (theta_r[i * Zn + j]));




		Grad_Vs[i * Zn + j] -= 2 * P[i * Zn + j] * Vs[i * Zn + j] * (omega_r[i * Zn + j] * (duxdz - duzdx) + duz_x_r[i * Zn + j] * duzdx + duz_z_r[i * Zn + j] * duzdz
			+ dux_z_r[i * Zn + j] * duxdz + dux_x_r[i * Zn + j] * duxdx);


		SSS[i * Zn + j] += ((Unx_s[(i)*Zn + j] * Unx_s[(i)*Zn + j]) + (Unz_s[(i)*Zn + j] * Unz_s[(i)*Zn + j]));

	}
}
__global__
void caculate_Grad(float* Grad_Vp, float* Grad_Vs, float* Unx_s, float* Unz_s, float* theta_r, float*omega_r, float* dux_z_r, float* dux_x_r, float* duz_z_r, float* duz_x_r, float* SSS, float* Vp, float* Vs, float* P, float dt, float dx, float dz, int t)
{
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float duxdx = 0.0f;
	float duzdz = 0.0f;
	float duzdx = 0.0f;
	float duxdz = 0.0f;
	float druxdx = 0.0f;
	float druzdz = 0.0f;
	float druzdx = 0.0f;
	float druxdz = 0.0f;
	float a[N] = { +1.2213e+0,-9.6931e-2,1.7448e-2,-2.9673e-3,+3.5901e-4,-2.1848e-5 };
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{
		duxdx = (a[0] * (Unx_s[(i)*Zn + j] - Unx_s[(i - 1) * Zn + j])
			+ a[1] * (Unx_s[(i + 1) * Zn + j] - Unx_s[(i - 2) * Zn + j])
			+ a[2] * (Unx_s[(i + 2) * Zn + j] - Unx_s[(i - 3) * Zn + j])
			+ a[3] * (Unx_s[(i + 3) * Zn + j] - Unx_s[(i - 4) * Zn + j])
			+ a[4] * (Unx_s[(i + 4) * Zn + j] - Unx_s[(i - 5) * Zn + j])
			+ a[5] * (Unx_s[(i + 5) * Zn + j] - Unx_s[(i - 6) * Zn + j])) / dx;
		duzdz = (a[0] * (Unz_s[(i)*Zn + j] - Unz_s[(i)*Zn + j - 1])
			+ a[1] * (Unz_s[(i)*Zn + j + 1] - Unz_s[(i)*Zn + j - 2])
			+ a[2] * (Unz_s[(i)*Zn + j + 2] - Unz_s[(i)*Zn + j - 3])
			+ a[3] * (Unz_s[(i)*Zn + j + 3] - Unz_s[(i)*Zn + j - 4])
			+ a[4] * (Unz_s[(i)*Zn + j + 4] - Unz_s[(i)*Zn + j - 5])
			+ a[5] * (Unz_s[(i)*Zn + j + 5] - Unz_s[(i)*Zn + j - 6])) / dz;
		duzdx = (a[0] * (Unz_s[(i + 1) * Zn + j] - Unz_s[(i)*Zn + j])
			+ a[1] * (Unz_s[(i + 2) * Zn + j] - Unz_s[(i - 1) * Zn + j])
			+ a[2] * (Unz_s[(i + 3) * Zn + j] - Unz_s[(i - 2) * Zn + j])
			+ a[3] * (Unz_s[(i + 4) * Zn + j] - Unz_s[(i - 3) * Zn + j])
			+ a[4] * (Unz_s[(i + 5) * Zn + j] - Unz_s[(i - 4) * Zn + j])
			+ a[5] * (Unz_s[(i + 6) * Zn + j] - Unz_s[(i - 5) * Zn + j])) / dx;
		duxdz = (a[0] * (Unx_s[(i)*Zn + j + 1] - Unx_s[(i)*Zn + j])
			+ a[1] * (Unx_s[(i)*Zn + j + 2] - Unx_s[(i)*Zn + j - 1])
			+ a[2] * (Unx_s[(i)*Zn + j + 3] - Unx_s[(i)*Zn + j - 2])
			+ a[3] * (Unx_s[(i)*Zn + j + 4] - Unx_s[(i)*Zn + j - 6])
			+ a[4] * (Unx_s[(i)*Zn + j + 5] - Unx_s[(i)*Zn + j - 4])
			+ a[5] * (Unx_s[(i)*Zn + j + 6] - Unx_s[(i)*Zn + j - 5])) / dz;




		Grad_Vp[i * Zn + j] -= 2 * P[i * Zn + j] * Vp[i * Zn + j] * ((duxdx + duzdz) * (theta_r[i * Zn + j]));




		Grad_Vs[i * Zn + j] -= 2 * P[i * Zn + j] * Vs[i * Zn + j] * (omega_r[i * Zn + j]*(duxdz- duzdx)+ duz_x_r[i * Zn + j]* duzdx+ duz_z_r[i * Zn + j] * duzdz
			+ dux_z_r[i * Zn + j] * duxdz+ dux_x_r[i * Zn + j] * duxdx);


		SSS[i * Zn + j] += ((Unx_s[(i)*Zn + j] * Unx_s[(i)*Zn + j]) + (Unz_s[(i)*Zn + j] * Unz_s[(i)*Zn + j]));

	}
}
__global__
void remove(int Sx, int Sz, int Z_receive, int t0, float dt, float dh, float* v, float* record_vx, float* record_vz)
{
	int  t;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float distance;
	if (i >= 0 && i < Xn && j == Z_receive)
	{
		for (t = 0; t < NT; t++)
		{
			distance = sqrtf(float(abs(Sx - i) * abs(Sx - i) + abs(Z_receive - Sz) * abs(Z_receive - Sz)));
			if (t < (2 * t0 + distance * dh * 1.0 / (dt * v[Sx * Zn + j])))
			{
				record_vx[i * NT + t] = 0;
				record_vz[i * NT + t] = 0;

			}


		}

	}
}
int main()
{
	cudaSetDevice(2);
	FILE* fp;
	int i, j, t, t0, l;
	int starttime, endtime, caltime;
	char filename[2000];
	int Sx, Sz, Z_receive, viewpoint_x, viewpoint_z;
	int size = Xn * Zn;

	float* Vx, * Vz, * theta, * omega, * duzdx, * duzdz, * duxdx, * duxdz;
	float* Vpx, * Vpz, *Vsx, * Vsz, * theta_x, * omega_x, * theta_z, * omega_z;
	float* Vx_r, * Vz_r;
	float* Vx_s, * Vz_s;
	float* record_vx, * record_vz,* record_vx_r, * record_vz_r;
	float* record_vx_1, * record_vz_1;
	float* record_vx_2, * record_vz_2;
	float* Vp_end, * Vs_end;
	float* d_Vp_end, * d_Vs_end;
	float* Grad_Vp, * Grad_Vs;
	float* d_Grad_Vp, * d_Grad_Vs;
	float* source,* Vp, * Vs, * P;
	float dt, dx, dz, dh;
	float* SSS;
	float* Vpx_r, * Vpz_r, *Vsx_r, * Vsz_r;


	dt = 0.001;
	dx = 10.0;
	dz =10.0;
	dh = 10.0;
	float* begin;
	float* begin1;


	float* d_Vx, * d_Vz, * d_theta, * d_omega, * d_duzdx, * d_duzdz, * d_duxdx, * d_duxdz;
	float* d_Vpx, * d_Vpz, * d_Vsx, * d_Vsz, * d_theta_x, * d_omega_x, * d_theta_z, * d_omega_z;

	float* d_Vx_r, * d_Vz_r, * d_theta_r, * d_omega_r, * d_duzdx_r, * d_duzdz_r, * d_duxdx_r, * d_duxdz_r;
	float* d_Vpx_r, * d_Vpz_r, * d_Vsx_r, * d_Vsz_r, * d_theta_x_r, * d_omega_x_r, * d_theta_z_r, * d_omega_z_r;

	float* d_Vx_s, * d_Vz_s, * d_theta_s, * d_omega_s, * d_duzdx_s, * d_duzdz_s, * d_duxdx_s, * d_duxdz_s;
	float* d_Vpx_s, * d_Vpz_s, * d_Vsx_s, * d_Vsz_s, * d_theta_x_s, * d_omega_x_s, * d_theta_z_s, * d_omega_z_s;


	float* d_Vx_up, * d_Vz_up, * d_theta_up, * d_omega_up, * d_duzdx_up, * d_duzdz_up, * d_duxdx_up, * d_duxdz_up;
	float* d_Vx_dn, * d_Vz_dn, * d_theta_dn, * d_omega_dn, * d_duzdx_dn, * d_duzdz_dn, * d_duxdx_dn, * d_duxdz_dn;
	float* d_Vx_lf, * d_Vz_lf, * d_theta_lf, * d_omega_lf, * d_duzdx_lf, * d_duzdz_lf, * d_duxdx_lf, * d_duxdz_lf;
	float* d_Vx_rt, * d_Vz_rt, * d_theta_rt, * d_omega_rt, * d_duzdx_rt, * d_duzdz_rt, * d_duxdx_rt, * d_duxdz_rt;

	float* d_Vp, * d_Vs, * d_P, * d_source,* ax, * az;
	float* d_SSS;
	float* d_record_vx, * d_record_vz, * d_record_vx_r, * d_record_vz_r;
	float* d_record_vx_1, * d_record_vz_1;
	float* d_record_vx_2, * d_record_vz_2;

	float* O_duz_xdz, * O_duz_zdx, * O_dux_zdx, * O_dux_xdz, * O_thetax, * O_thetaz, * O_omegaz, * O_omegax;
	float* F_xx, * F_zz, * F_xz, * F_zx;
	float* O_duz_xdz_r, * O_duz_zdx_r, * O_dux_zdx_r, * O_dux_xdz_r, * O_thetax_r, * O_thetaz_r, * O_omegaz_r, * O_omegax_r;
	float* F_xx_r, * F_zz_r, * F_xz_r, * F_zx_r;
	float* Vpx_now, * Vpz_now, * Vsx_now, * Vsz_now, * Vpx_past, * Vpz_past, * Vsx_past, * Vsz_past;
	float* Vpx_now_r, * Vpz_now_r, * Vsx_now_r, * Vsz_now_r, * Vpx_past_r, * Vpz_past_r, * Vsx_past_r, * Vsz_past_r;
	float* Vpx_now_s, * Vpz_now_s, * Vsx_now_s, * Vsz_now_s, * Vpx_past_s, * Vpz_past_s, * Vsx_past_s, * Vsz_past_s;
	float* fenzi_PP, *fenzi_PS, *fenmu_P, *d_image_PP, *d_image_PS, * d_image_PP_lap, * d_image_PS_lap;
	float* image_PP, *image_PS, * image_PP_lap, * image_PS_lap;
	float* All_image_PP, *All_image_PS;
	float* All_image_PP_lap, * All_image_PS_lap;
	float* res_recordux, * res_recorduz;
	

	Vx = (float*)malloc(size * sizeof(float));
	Vz = (float*)malloc(size * sizeof(float));
	theta = (float*)malloc(size * sizeof(float));
	omega = (float*)malloc(size * sizeof(float));
	Vpx = (float*)malloc(size * sizeof(float));
	Vpz = (float*)malloc(size * sizeof(float));
	Vsx = (float*)malloc(size * sizeof(float));
	Vsz = (float*)malloc(size * sizeof(float));

	Vx_r = (float*)malloc(size * sizeof(float));
	Vz_r = (float*)malloc(size * sizeof(float));
	Vpx_r = (float*)malloc(size * sizeof(float));
	Vpz_r = (float*)malloc(size * sizeof(float));
	Vsx_r = (float*)malloc(size * sizeof(float));
	Vsz_r = (float*)malloc(size * sizeof(float));

	Vx_s = (float*)malloc(size * sizeof(float));
	Vz_s = (float*)malloc(size * sizeof(float));

	Vp = (float*)malloc(size * sizeof(float));
	Vs = (float*)malloc(size * sizeof(float));
	P = (float*)malloc(size * sizeof(float));
	ax = (float*)malloc(size * sizeof(float));
	az = (float*)malloc(size * sizeof(float));
	image_PP = (float*)calloc(size, sizeof(float));
	image_PS = (float*)calloc(size, sizeof(float));
	All_image_PP = (float*)calloc(size, sizeof(float));
	All_image_PS = (float*)calloc(size, sizeof(float));
	image_PP_lap = (float*)calloc(size, sizeof(float));
	image_PS_lap = (float*)calloc(size, sizeof(float));	
	All_image_PP_lap = (float*)calloc(size, sizeof(float));
	All_image_PS_lap = (float*)calloc(size, sizeof(float));

	source = (float*)malloc(NT * sizeof(float));
	record_vx = (float*)malloc(NT * Xn * sizeof(float));
	record_vz = (float*)malloc(NT * Xn * sizeof(float));
	record_vx_r = (float*)malloc(NT * Xn * sizeof(float));
	record_vz_r = (float*)malloc(NT * Xn * sizeof(float));

	record_vx_1 = (float*)malloc(NT * Xn * sizeof(float));
	record_vz_1 = (float*)malloc(NT * Xn * sizeof(float));
	record_vx_2 = (float*)malloc(NT * Xn * sizeof(float));
	record_vz_2 = (float*)malloc(NT * Xn * sizeof(float));


	begin = (float*)malloc(size * sizeof(float));
	begin1 = (float*)malloc(Xn * NT * sizeof(float));

	Vp_end = (float*)malloc(size * sizeof(float));
	Vs_end = (float*)malloc(size * sizeof(float));
	Grad_Vp = (float*)malloc(size * sizeof(float));
	Grad_Vs = (float*)malloc(size * sizeof(float));
	SSS = (float*)malloc(size * sizeof(float));

	cudaMalloc((void**)&d_Vx, size * sizeof(float));
	cudaMalloc((void**)&d_Vz, size * sizeof(float));
	cudaMalloc((void**)&d_theta, size * sizeof(float));
	cudaMalloc((void**)&d_omega, size * sizeof(float));
	cudaMalloc((void**)&d_duzdx, size * sizeof(float));
	cudaMalloc((void**)&d_duxdx, size * sizeof(float));	
	cudaMalloc((void**)&d_duzdz, size * sizeof(float)); 
	cudaMalloc((void**)&d_duxdz, size * sizeof(float));
	cudaMalloc((void**)&d_Vpx, size * sizeof(float));
	cudaMalloc((void**)&d_Vsx, size * sizeof(float));
	cudaMalloc((void**)&d_Vpz, size * sizeof(float));
	cudaMalloc((void**)&d_Vsz, size * sizeof(float));
	cudaMalloc((void**)&d_theta_x, size * sizeof(float));
	cudaMalloc((void**)&d_theta_z, size * sizeof(float));
	cudaMalloc((void**)&d_omega_x, size * sizeof(float));
	cudaMalloc((void**)&d_omega_z, size * sizeof(float));
	cudaMalloc((void**)&d_SSS, size * sizeof(float));

	
	cudaMalloc((void**)&d_Vx_r, size * sizeof(float));
	cudaMalloc((void**)&d_Vz_r, size * sizeof(float));
	cudaMalloc((void**)&d_theta_r, size * sizeof(float));
	cudaMalloc((void**)&d_omega_r, size * sizeof(float));
	cudaMalloc((void**)&d_duzdx_r, size * sizeof(float));
	cudaMalloc((void**)&d_duxdx_r, size * sizeof(float));
	cudaMalloc((void**)&d_duzdz_r, size * sizeof(float));
	cudaMalloc((void**)&d_duxdz_r, size * sizeof(float));
	cudaMalloc((void**)&d_Vpx_r, size * sizeof(float));
	cudaMalloc((void**)&d_Vsx_r, size * sizeof(float));
	cudaMalloc((void**)&d_Vpz_r, size * sizeof(float));
	cudaMalloc((void**)&d_Vsz_r, size * sizeof(float));
	cudaMalloc((void**)&d_theta_x_r, size * sizeof(float));
	cudaMalloc((void**)&d_theta_z_r, size * sizeof(float));
	cudaMalloc((void**)&d_omega_x_r, size * sizeof(float));
	cudaMalloc((void**)&d_omega_z_r, size * sizeof(float));

	cudaMalloc((void**)&d_Vx_s, size * sizeof(float));
	cudaMalloc((void**)&d_Vz_s, size * sizeof(float));
	cudaMalloc((void**)&d_theta_s, size * sizeof(float));
	cudaMalloc((void**)&d_omega_s, size * sizeof(float));
	cudaMalloc((void**)&d_duzdx_s, size * sizeof(float));
	cudaMalloc((void**)&d_duxdx_s, size * sizeof(float));
	cudaMalloc((void**)&d_duzdz_s, size * sizeof(float));
	cudaMalloc((void**)&d_duxdz_s, size * sizeof(float));
	cudaMalloc((void**)&d_Vpx_s, size * sizeof(float));
	cudaMalloc((void**)&d_Vsx_s, size * sizeof(float));
	cudaMalloc((void**)&d_Vpz_s, size * sizeof(float));
	cudaMalloc((void**)&d_Vsz_s, size * sizeof(float));
	cudaMalloc((void**)&d_theta_x_s, size * sizeof(float));
	cudaMalloc((void**)&d_theta_z_s, size * sizeof(float));
	cudaMalloc((void**)&d_omega_x_s, size * sizeof(float));
	cudaMalloc((void**)&d_omega_z_s, size * sizeof(float));

	cudaMalloc((void**)&d_Vp, size * sizeof(float));
	cudaMalloc((void**)&d_Vs, size * sizeof(float));
	cudaMalloc((void**)&d_P, size * sizeof(float));
	cudaMalloc((void**)&d_source, NT * sizeof(float));
	cudaMalloc((void**)&d_record_vx, Xn * NT * sizeof(float));
	cudaMalloc((void**)&d_record_vz, Xn * NT * sizeof(float));
	cudaMalloc((void**)&d_record_vx_r, Xn * NT * sizeof(float));
	cudaMalloc((void**)&d_record_vz_r, Xn * NT * sizeof(float));

	cudaMalloc((void**)&d_record_vx_1, Xn * NT * sizeof(float));
	cudaMalloc((void**)&d_record_vz_1, Xn * NT * sizeof(float));
	cudaMalloc((void**)&d_record_vx_2, Xn * NT * sizeof(float));
	cudaMalloc((void**)&d_record_vz_2, Xn * NT * sizeof(float));
	
	cudaMalloc((void**)&res_recordux, Xn* NT * sizeof(float));
	cudaMalloc((void**)&res_recorduz, Xn* NT * sizeof(float));

	cudaMalloc((void**)&d_Vx_up, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_Vz_up, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_theta_up, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_omega_up, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdx_up, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdz_up, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdx_up, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdz_up, 2 * N * Xn * (NT - 1) * sizeof(float));

	cudaMalloc((void**)&d_Vx_dn, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_Vz_dn, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_theta_dn, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_omega_dn, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdx_dn, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdz_dn, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdx_dn, 2 * N * Xn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdz_dn, 2 * N * Xn * (NT - 1) * sizeof(float));


	cudaMalloc((void**)&d_Vx_lf, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_Vz_lf, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_theta_lf, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_omega_lf, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdx_lf, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdz_lf, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdx_lf, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdz_lf, 2 * N * Zn * (NT - 1) * sizeof(float));

	cudaMalloc((void**)&d_Grad_Vp, size * sizeof(float));
	cudaMalloc((void**)&d_Grad_Vs, size * sizeof(float));

	cudaMalloc((void**)&d_Vp_end, size * sizeof(float));
	cudaMalloc((void**)&d_Vs_end, size * sizeof(float));

	cudaMalloc((void**)&d_Vx_rt, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_Vz_rt, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_theta_rt, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_omega_rt, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdx_rt, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duzdz_rt, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdx_rt, 2 * N * Zn * (NT - 1) * sizeof(float));
	cudaMalloc((void**)&d_duxdz_rt, 2 * N * Zn * (NT - 1) * sizeof(float));
	
	cudaMalloc((void**)&O_duz_xdz, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_duz_zdx, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_dux_zdx, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_dux_xdz, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_thetax, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_thetaz, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_omegaz, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&O_omegax, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&F_xx, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&F_zz, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&F_xz, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&F_zx, Xn* Zn * sizeof(float));
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

	
	cudaMalloc((void**)&Vpx_now, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpz_now, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsx_now, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsz_now, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpx_past, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpz_past, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsx_past, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsz_past, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpx_now_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpz_now_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsx_now_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsz_now_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpx_past_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpz_past_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsx_past_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsz_past_r, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpx_now_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpz_now_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsx_now_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsz_now_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpx_past_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vpz_past_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsx_past_s, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&Vsz_past_s, Xn* Zn * sizeof(float));
	
	cudaMalloc((void**)&fenzi_PP, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&fenzi_PS, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&fenmu_P, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PP, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PS, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PP_lap, Xn* Zn * sizeof(float));
	cudaMalloc((void**)&d_image_PS_lap, Xn* Zn * sizeof(float));

	float Nk = PI * PI * FM * FM * dt * dt;
	t0 = ceil(1.0 / (FM * dt));
	for (t = 0; t < NT; t++)
	{
		source[t] = (1.0 - 2.0 * Nk * (t - t0) * (t - t0)) * exp(-Nk * (t - t0) * (t - t0));
		for (j = 0; j < Zn; j++)
		{
			begin1[j * NT + t] = 0.0;
		}


	}

	for (i = 0; i < Xn; i++)
		for (j = 0; j < Zn; j++)
		{
			theta[i * Zn + j] = 0.0;
			omega[i * Zn + j] = 0.0;
			Vx[i * Zn + j] = 0.0;
			Vz[i * Zn + j] = 0.0;
			begin[i * Zn + j] = 0.0;
		}
	creatmodel(Vp, Vs, P);

	dim3 dimGrid(ceil(Xn / 8.0), ceil(Zn / 8.0), 1);
	dim3 dimBlock(8, 8, 1);

	int k;
	for (k = 0; k <1; k++)
	{
		for (i = 0; i < Xn; i++)
			for (j = 0; j < Zn; j++)
			{

				Grad_Vp[i * Zn + j] = 0.0;
				Grad_Vs[i * Zn + j] = 0.0;
				SSS[i * Zn + j] = 0.0;
			}

		cudaMemcpy(d_Grad_Vp, begin, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Grad_Vs, begin, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_SSS, begin, size * sizeof(float), cudaMemcpyHostToDevice);
		cout << k + 1 << " th iteration:" << endl;


		for (i = L; i < Xn - L; i++)
		{
			for (j = L; j < 50 + L; j++)
			{
				Vp[i * Zn + j] = 3400;

			}
		}
		for (i = L; i < Xn - L; i++)
		{
			for (j = 50 + L; j < 100 + L; j++)
			{
				Vp[i * Zn + j] = 3800;

			}
		}
		for (i = L; i < Xn - L; i++)
		{
			for (j = 100 + L; j < Zn - L; j++)
			{
				Vp[i * Zn + j] = 4500;

			}
		}
		for (i = L; i < Xn - L; i++)
		{
			for (j = L; j < 100 + L; j++)
			{

				Vs[i * Zn + j] = 2900;
				P[i * Zn + j] = 1.0;
			}
		}
		for (i = L; i < Xn - L; i++)
		{
			for (j = 100 + L; j < Zn - L; j++)
			{

				Vs[i * Zn + j] = 3800;
				P[i * Zn + j] = 1.0;
			}
		}

		//sprintf(filename, "./model/Vp.dat");
		//if ((fp = fopen(filename, "wb")) != NULL)
		//{
		//	for (i = L; i < Xn - L; i++)
		//		for (j = L; j < Zn - L; j++)
		//		{
		//			fwrite(&Vp[i * Zn + j], sizeof(float), 1, fp);

		//		}
		//}
		//fclose(fp);
		//sprintf(filename, "./model/Vs.dat");
		//if ((fp = fopen(filename, "wb")) != NULL)
		//{
		//	for (i = L; i < Xn - L; i++)
		//		for (j = L; j < Zn - L; j++)
		//		{
		//			fwrite(&Vs[i * Zn + j], sizeof(float), 1, fp);

		//		}
		//}
		//fclose(fp);

		xiangbian(P, Vp, Vs);

		for (i = 0; i < Xn; i++)
			for (j = 0; j < Zn; j++)
			{
				if (i >= 0 && i < L)
				{
					ax[i * Zn + j] = log10(1.0 / R) * 1.5 * Vp[i * Zn + j] / (L)*pow(1.0 * (L - i) / (L), 4.0);

				}
				else if (i > Xn - L && i < Xn)
				{
					ax[i * Zn + j] = log10(1.0 / R) * 1.5 * Vp[i * Zn + j] / (L)*pow(1.0 * (i - Xn + L) / (L), 4.0);

				}
				else
				{
					ax[i * Zn + j] = 0.0;

				}
			}
		for (i = 0; i < Xn; i++)
			for (j = 0; j < Zn; j++)
			{
				if (j >= 0 && j < L)
				{
					az[i * Zn + j] = log10(1.0 / R) * 1.5 * Vp[i * Zn + j] / (L)*pow(1.0 * (L - j) / (L), 4.0);

				}
				else if (j > Zn - L && j < Zn)
				{
					az[i * Zn + j] = log10(1.0 / R) * 1.5 * Vp[i * Zn + j] / (L)*pow(1.0 * (j - Zn + L) / (L), 4.0);

				}
				else
				{
					az[i * Zn + j] = 0.0;

				}

			}
		float E1 = 0, E2 = 0, E3 = 0;

		for (l = 0; l < shot_num; l++)
		{

			cout << k + 1 << "  ShotNumber:" << l + 1 << endl;
			
			Sx = L + l * 5, Sz = L;
			Z_receive = L;

			float* d_ax, * d_az;
			cudaMalloc((void**)&d_ax, size * sizeof(float));
			cudaMalloc((void**)&d_az, size * sizeof(float));
			cudaMemcpy(d_ax, ax, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_az, az, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_Vx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vpx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vpz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vsx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vsz, begin, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_theta, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_theta_x, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega_x, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_theta_z, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega_z, begin, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_duzdx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duzdz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duxdx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duxdz, begin, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_Vx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vpx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vpz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vsx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vsz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);


			cudaMemcpy(d_theta_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_theta_x_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega_x_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_theta_z_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega_z_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_duzdx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duzdz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duxdx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duxdz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_Vx_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vz_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vpx_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vpz_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vsx_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vsz_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_theta_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_theta_x_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega_x_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_theta_z_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_omega_z_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);


			cudaMemcpy(d_duzdx_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duzdz_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duxdx_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_duxdz_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_record_vz, begin1, NT * Xn * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_record_vx, begin1, NT * Xn * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_record_vz_1, begin1, NT * Xn * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_record_vx_1, begin1, NT * Xn * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_record_vz_2, begin1, NT * Xn * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_record_vx_2, begin1, NT * Xn * sizeof(float), cudaMemcpyHostToDevice);
			
			cudaMemcpy(res_recordux, begin1, NT * Xn * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(res_recorduz, begin1, NT * Xn * sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(d_Vp, Vp, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vs, Vs, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_P, P, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_source, source, NT * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_SSS, SSS, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Grad_Vp, Grad_Vp, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Grad_Vs, Grad_Vs, size * sizeof(float), cudaMemcpyHostToDevice);
			
			cudaMemcpy(O_duz_xdz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_duz_zdx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_dux_zdx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_dux_xdz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_thetax, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_thetaz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_omegaz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_omegax, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(F_xx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(F_zz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(F_xz, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(F_zx, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_duz_xdz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_duz_zdx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_dux_zdx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_dux_xdz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_thetax_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_thetaz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_omegaz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(O_omegax_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(F_xx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(F_zz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(F_xz_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(F_zx_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			
			cudaMemcpy(Vpx_now, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpz_now, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsx_now, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsz_now, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpx_past, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpz_past, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsx_past, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsz_past, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpx_now_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpz_now_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsx_now_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsz_now_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpx_past_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpz_past_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsx_past_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsz_past_r, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpx_now_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpz_now_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsx_now_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsz_now_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpx_past_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vpz_past_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsx_past_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(Vsz_past_s, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			
			cudaMemcpy(fenzi_PP, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(fenzi_PS, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(fenmu_P, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_image_PP, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_image_PS, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_image_PP_lap, begin, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_image_PS_lap, begin, size * sizeof(float), cudaMemcpyHostToDevice);

			printf("forward\n");
			for (t = 0; t < NT; t++)
			{

				forward_s << <dimGrid, dimBlock >> > (d_Vx, d_Vz, d_theta, d_omega, d_duxdz, d_duzdz, d_duxdx, d_duzdx,
					d_ax, d_az, dt, dx, dz, t, Sx, Sz, d_Vp, d_Vs, d_source, F_xx, F_zz, F_xz, F_zx);

				forward_u << <dimGrid, dimBlock >> > (d_Vx, d_Vz, d_Vpx, d_Vpz, d_Vsx, d_Vsz, d_theta, d_omega, d_duxdz, d_duzdz, d_duxdx, d_duzdx,
					d_az, d_ax, dt, dx, dz, Z_receive, d_record_vx, d_record_vz, t, Vpx_now, Vpz_now, Vsx_now, Vsz_now,
					Vpx_past, Vpz_past, Vsx_past, Vsz_past, O_duz_xdz, O_duz_zdx, O_dux_zdx, O_dux_xdz, O_thetax, O_thetaz, O_omegaz, O_omegax);
			

				if (t != NT - 1)
				{

					save_wavefiled << <dimGrid, dimBlock >> > (d_Vx_up, d_Vz_up, d_theta_up, d_omega_up, d_duzdx_up, d_duzdz_up, d_duxdx_up, d_duxdz_up,
						d_Vx_dn, d_Vz_dn, d_theta_dn, d_omega_dn, d_duzdx_dn, d_duzdz_dn, d_duxdx_dn, d_duxdz_dn,
						d_Vx_lf, d_Vz_lf, d_theta_lf, d_omega_lf, d_duzdx_lf, d_duzdz_lf, d_duxdx_lf, d_duxdz_lf,
						d_Vx_rt, d_Vz_rt, d_theta_rt, d_omega_rt, d_duzdx_rt, d_duzdz_rt, d_duxdx_rt, d_duxdz_rt,
						d_Vx, d_Vz, d_theta, d_omega, d_duzdx, d_duzdz, d_duxdx, d_duxdz, t);
				}
				if (t == NT - 1)
				{
					
					read_last_wavefiled2 << <dimGrid, dimBlock >> > (d_theta_s, d_omega_s, d_Vx_s, d_Vz_s, d_duzdx_s, d_duzdz_s, d_duxdz_s, d_duxdx_s,
						d_Vpx_s, d_Vpz_s, d_Vsx_s, d_Vsz_s,	d_theta, d_omega, d_Vx, d_Vz, d_duzdx, d_duzdz, d_duxdz, d_duxdx,
						d_Vpx, d_Vpz, d_Vsx, d_Vsz, t);
					

				}
				

			
			
				
			}
			remove << <dimGrid, dimBlock >> > (Sx, Sz, Z_receive, t0, dt, dh, d_Vp, d_record_vx, d_record_vz);
			cudaMemcpy(record_vx, d_record_vx, Xn * NT * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(record_vz, d_record_vz, Xn * NT * sizeof(float), cudaMemcpyDeviceToHost);

			


			printf("\n");


			

			printf("backward\n");

			

			cudaMemcpy(d_record_vx, record_vx, Xn * NT * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_record_vz, record_vz, Xn * NT * sizeof(float), cudaMemcpyHostToDevice);
		

			for (t = NT - 1; t >= 0; t--)
			{
				
				if (t == NT - 1)
				{

					load_record << <dimGrid, dimBlock >> > (Z_receive, d_Vx_r, d_Vz_r, d_record_vx, d_record_vz, t);

					rt_s_res << <dimGrid, dimBlock >> > (d_Vx_r, d_Vz_r, d_theta_r, d_omega_r, d_duxdz_r, d_duzdz_r, d_duxdx_r, d_duzdx_r,
						d_ax, d_az, dt, dx, dz, d_Vp, d_Vs, F_xx_r, F_zz_r, F_xz_r, F_zx_r);

					rt_u_res << <dimGrid, dimBlock >> > (d_Vx_r, d_Vz_r, d_Vpx_r, d_Vpz_r, d_Vsx_r, d_Vsz_r, d_theta_r, d_omega_r, d_duxdz_r, d_duzdz_r, d_duxdx_r, d_duzdx_r,
						d_az, d_ax, dt, dx, dz, t, Vpx_now_r, Vpz_now_r, Vsx_now_r, Vsz_now_r,
						Vpx_past_r, Vpz_past_r, Vsx_past_r, Vsz_past_r, O_duz_xdz_r, O_duz_zdx_r, O_dux_zdx_r, O_dux_xdz_r, O_thetax_r, O_thetaz_r, O_omegaz_r, O_omegax_r);


				}
				
				
				if (t < NT - 1)
				{
					reshot_u << <dimGrid, dimBlock >> > (d_Vx_s, d_Vz_s, d_Vpx_s, d_Vpz_s, d_Vsx_s, d_Vsz_s, d_theta_s, d_omega_s, d_duxdz_s, d_duzdz_s, d_duxdx_s, d_duzdx_s,
						dt, dx, dz, Vpx_now, Vpz_now, Vsx_now, Vsz_now, Vpx_past, Vpz_past, Vsx_past, Vsz_past);

					read_wavefiled1 << <dimGrid, dimBlock >> > (d_Vx_up, d_Vz_up, d_Vx_dn, d_Vz_dn, d_Vx_lf, d_Vz_lf, d_Vx_rt, d_Vz_rt,
						d_Vx_s, d_Vz_s, t);

					reshot_s << <dimGrid, dimBlock >> > (d_Vx_s, d_Vz_s, d_theta_s, d_omega_s, d_duxdz_s, d_duzdz_s, d_duxdx_s, d_duzdx_s,
						dt, dx, dz, d_Vp, d_Vs);

					read_wavefiled2 << <dimGrid, dimBlock >> > (d_theta_up, d_omega_up, d_duxdz_up, d_duxdx_up, d_duzdz_up, d_duzdx_up,
						d_theta_dn, d_omega_dn, d_duxdz_dn, d_duxdx_dn, d_duzdz_dn, d_duzdx_dn,
						d_theta_lf, d_omega_lf, d_duxdz_lf, d_duxdx_lf, d_duzdz_lf, d_duzdx_lf,
						d_theta_rt, d_omega_rt, d_duxdz_rt, d_duxdx_rt, d_duzdz_rt, d_duzdx_rt,
						d_theta_s, d_omega_s, d_duxdz_s, d_duxdx_s, d_duzdz_s, d_duzdx_s, t);
					
					load_record << <dimGrid, dimBlock >> > (Z_receive, d_Vx_r, d_Vz_r, d_record_vx, d_record_vz, t);

					rt_s_res << <dimGrid, dimBlock >> > (d_Vx_r, d_Vz_r, d_theta_r, d_omega_r, d_duxdz_r, d_duzdz_r, d_duxdx_r, d_duzdx_r,
						d_ax, d_az, dt, dx, dz, d_Vp, d_Vs, F_xx_r, F_zz_r, F_xz_r, F_zx_r);

					rt_u_res << <dimGrid, dimBlock >> > (d_Vx_r, d_Vz_r, d_Vpx_r, d_Vpz_r, d_Vsx_r, d_Vsz_r, d_theta_r, d_omega_r, d_duxdz_r, d_duzdz_r, d_duxdx_r, d_duzdx_r,
						d_az, d_ax, dt, dx, dz, t, Vpx_now_r, Vpz_now_r, Vsx_now_r, Vsz_now_r,
						Vpx_past_r, Vpz_past_r, Vsx_past_r, Vsz_past_r, O_duz_xdz_r, O_duz_zdx_r, O_dux_zdx_r, O_dux_xdz_r, O_thetax_r, O_thetaz_r, O_omegaz_r, O_omegax_r);

				}
				

				corr_v << <dimGrid, dimBlock >> > (fenzi_PP, fenzi_PS, fenmu_P, d_Vpx_s, d_Vpz_s, d_Vpx_r, d_Vpz_r, d_Vsx_r, d_Vsz_r);
				
				
			}
			
			
			image_fun << <dimGrid, dimBlock >> > (fenzi_PP, fenmu_P, d_image_PP);
			image_fun << <dimGrid, dimBlock >> > (fenzi_PS, fenmu_P, d_image_PS);

			cudaMemcpy(image_PP, d_image_PP, Xn* Zn * sizeof(float), cudaMemcpyDeviceToHost);			
			cudaMemcpy(image_PS, d_image_PS, Xn* Zn * sizeof(float), cudaMemcpyDeviceToHost);
			Laplace << <dimGrid, dimBlock >> > (dx, d_image_PP, d_image_PP_lap);
			Laplace << <dimGrid, dimBlock >> > (dx, d_image_PS, d_image_PS_lap);

			cudaMemcpy(image_PP_lap, d_image_PP_lap, Xn* Zn * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(image_PS_lap, d_image_PS_lap, Xn* Zn * sizeof(float), cudaMemcpyDeviceToHost);

			
			sprintf(filename, "./image2/image_PP_lap_%d_%d_%d.dat", (Xn - 2 * L), (Zn - 2 * L), l);
			if ((fp = fopen(filename, "wb")) != NULL)
			{
				for (i = L; i < Xn - L; i++)
				{
					for (j = L; j < Zn - L; j++)
					{
						fwrite(&image_PP_lap[i * Zn + j], sizeof(float), 1, fp);
					}
				}
			}fclose(fp);
			sprintf(filename, "./image2/image_PS_lap_%d_%d_%d.dat", (Xn - 2 * L), (Zn - 2 * L), l);
			if ((fp = fopen(filename, "wb")) != NULL)
			{
				for (i = L; i < Xn - L; i++)
				{
					for (j = L; j < Zn - L; j++)
					{
						fwrite(&image_PS_lap[i * Zn + j], sizeof(float), 1, fp);
					}
				}
			}fclose(fp);
		}
		
		for (l = 0; l < shot_num; l++)
		{


			sprintf(filename, "./image2/image_PP_lap_%d_%d_%d.dat", (Xn - 2 * L), (Zn - 2 * L), l);
			if ((fp = fopen(filename, "rb")) != NULL)
			{
				float a = 0;
				for (i = L; i < Xn - L; i++)
				{
					for (j = L; j < Zn - L; j++)
					{
						fread(&image_PP_lap[i * Zn + j], sizeof(float), 1, fp);
					}
				}
			}fclose(fp);

			sprintf(filename, "./image2/image_PS_lap_%d_%d_%d.dat", (Xn - 2 * L), (Zn - 2 * L), l);
			if ((fp = fopen(filename, "rb")) != NULL)
			{
				float a = 0;
				for (i = L; i < Xn - L; i++)
				{
					for (j = L; j < Zn - L; j++)
					{
						fread(&image_PS_lap[i * Zn + j], sizeof(float), 1, fp);
					}
				}
			}fclose(fp);


			for (i = L; i < Xn - L; i++)
			{
				for (j = L; j < Zn - L; j++)
				{
					All_image_PP_lap[i * Zn + j] += image_PP_lap[i * Zn + j];
					All_image_PS_lap[i * Zn + j] += image_PS_lap[i * Zn + j];
				}
			}

		}

		
		sprintf(filename, "./image2/All_image_PP_lap.dat");
		if ((fp = fopen(filename, "wb")) != NULL)
		{
			for (i = L; i < Xn - L; i++)
			{
				for (j = L; j < Zn - L; j++)
				{
					fwrite(&All_image_PP_lap[i * Zn + j], sizeof(float), 1, fp);
				}
			}
		}fclose(fp);
		sprintf(filename, "./image2/All_image_PS_lap.dat");
		if ((fp = fopen(filename, "wb")) != NULL)
		{
			for (i = L; i < Xn - L; i++)
			{
				for (j = L; j < Zn - L; j++)
				{
					fwrite(&All_image_PS_lap[i * Zn + j], sizeof(float), 1, fp);
				}
			}
		}fclose(fp);

	}
	





















	return 0;
}
