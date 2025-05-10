#include "optimization.h"
#include "cuda_runtime.h"
//#include "device_atomic_functions.hpp"
#include "lib.cuh"
#include "Grid.h"
#include "gpuVector.h"
#include <vector>
#include "templateMatrix.h"
//#include "gpuVector.h"

extern  __constant__  double gTemplateMatrix[24][24];
extern  __constant__  double gTemplateMatrix11[24][24];
extern  __constant__  double gTemplateMatrix12[24][24];
extern  __constant__  double gTemplateMatrix13[24][24];
extern  __constant__  double gTemplateMatrix22[24][24];
extern  __constant__  double gTemplateMatrix23[24][24];
extern  __constant__  double gTemplateMatrix33[24][24];
extern  __constant__  double gTemplateMatrix44[24][24];
extern  __constant__  double gTemplateMatrix55[24][24];
extern  __constant__  double gTemplateMatrix66[24][24];
extern  __constant__ int* gV2E[8];
extern  __constant__ int* gV2Vfine[27];
extern  __constant__ int* gV2Vcoarse[8];
extern  __constant__ int* gV2V[27];
extern  __constant__ int* gVfine2Vfine[27];
extern  __constant__ int* gV2VfineC[64];// vertex to fine grid element center 
extern  __constant__ int* gVfine2Efine[8];
extern  __constant__ int* gVfine2Effine[8];
extern  __constant__ float power_penalty[1];
extern  __constant__ double* gU[3];
extern  __constant__ double* gF[3];
extern  __constant__ double* gR[3];
extern  __constant__ double* gUworst[3];
extern  __constant__ double* gFworst[3];
extern  __constant__ double* gRfine[3];
extern  __constant__ double* gUcoarse[3];
extern  __constant__ int gGS_num[8];
extern  __constant__ int gmode[1];
extern  __constant__ int* gVflag[1];
extern  __constant__ int* gEflag[1];
extern  __constant__ int gLayerid[1];
extern  __constant__ int gDEBUG[1];


extern __device__ void loadTemplateMatrix(volatile double KE[24][24]);
extern __device__ void loadTemplateMatrix11(volatile float KE11[24][24]);
extern __device__ void loadTemplateMatrix12(volatile float KE12[24][24]);
extern __device__ void loadTemplateMatrix13(volatile float KE13[24][24]);
extern __device__ void loadTemplateMatrix22(volatile float KE22[24][24]);
extern __device__ void loadTemplateMatrix23(volatile float KE23[24][24]);
extern __device__ void loadTemplateMatrix33(volatile float KE33[24][24]);
extern __device__ void loadTemplateMatrix44(volatile float KE44[24][24]);
extern __device__ void loadTemplateMatrix55(volatile float KE55[24][24]);
extern __device__ void loadTemplateMatrix66(volatile float KE66[24][24]);

template<int N>
__device__ int gridPos2id(int x, int y, int z) {
	return x + y * N + z * N*N;
}

//  suppose Uworst, Fworst is prepared in U, F
__global__ void computeSensitivity_kernel(int nv, float* rholist, double mu, float* sens) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ double KE[24][24];

	loadTemplateMatrix(KE);

	if (tid >= nv) return;

	int vid = tid;

	int vneigh[27];
	for (int i = 0; i < 27; i++) {
		vneigh[i] = gV2V[i][vid];
	}

	// traverse incident elements of vertex
	for (int i = 0; i < 8; i++) {
		double partialSens = 0;

		int eid = gV2E[i][vid];
		if (eid == -1) continue;
		double Ui[3] = { gU[0][vid],gU[1][vid],gU[2][vid] };
		double penal = power_penalty[0] * powf(rholist[eid], power_penalty[0] - 1);

		// compute partial node force (element i's contribution) K_\rho * Uworst on vi
		double KrhoU[3] = { 0. };
		// vertex self id in neihbor element i
		int vi = 7 - i;
		// vertex neighbor id in element i, traverse them and compute the corresponding node force on self
		for (int vj = 0; vj < 8; vj++) {
			int vjlid = gridPos2id<3>(i % 2 + vj % 2, i % 4 / 2 + vj % 4 / 2, i / 4 + vj / 4);
			double Uj[3];
			for (int k = 0; k < 3; k++) Uj[k] = gU[k][vneigh[vjlid]];
			for (int krow = 0; krow < 3; krow++) {
				for (int kcol = 0; kcol < 3; kcol++) {
					KrhoU[krow] += KE[vi * 3 + krow][vj * 3 + kcol] * Uj[kcol];
				}
			}
		}

		for (int k = 0; k < 3; k++) KrhoU[k] *= penal;

#if 0
		// sensitivity  u_worst * dK/drho * u_worst
		for (int k = 0; k < 3; k++) partialSens += Ui[k] * KrhoU[k];

		// sensitivity  - 2 mu * u_worst * dK/drho * K * u_worst
		for (int k = 0; k < 3; k++) partialSens += -2 * mu * KrhoU[k] * gFworst[k][vid];

		// sensitivity  - lambda * N * dK/drho * u_worst
		for (int k = 0; k < 3; k++) {
			partialSens += -KrhoU[k] * gU[k][vid];
		}
#else
		// sensitivity  - u_worst * dK/drho * u_worst
		for (int k = 0; k < 3; k++) partialSens -= Ui[k] * KrhoU[k];

#endif

		atomicAdd(sens + eid, float(partialSens));
	}

}

__global__ void computeSensitivity_spinodal_kernel(int nv, float* rho, float* c11, float* c12, float* c13, float* c22, float* c23, float* c33, float* c44, float* c55, float* c66, float* sens) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ float KE11[24][24];
    __shared__ float KE12[24][24];
    __shared__ float KE13[24][24];
    __shared__ float KE22[24][24];
    __shared__ float KE23[24][24];
    __shared__ float KE33[24][24];
    __shared__ float KE44[24][24];
    __shared__ float KE55[24][24];
    __shared__ float KE66[24][24];

	loadTemplateMatrix11(KE11);
    loadTemplateMatrix12(KE12);
    loadTemplateMatrix13(KE13);
    loadTemplateMatrix22(KE22);
    loadTemplateMatrix23(KE23);
    loadTemplateMatrix33(KE33);
    loadTemplateMatrix44(KE44);
    loadTemplateMatrix55(KE55);
    loadTemplateMatrix66(KE66);

	if (tid >= nv) return;

	int vid = tid;

	int vneigh[27];
	for (int i = 0; i < 27; i++) {
		vneigh[i] = gV2V[i][vid];
	}

	// traverse incident elements of vertex
	for (int i = 0; i < 8; i++) {
		double partialSens = 0;

		int eid = gV2E[i][vid];
		if (eid == -1) continue;
		double Ui[3] = { gU[0][vid],gU[1][vid],gU[2][vid] };
		// double penal = power_penalty[0] * powf(rholist[eid], power_penalty[0] - 1);

		// compute partial node force (element i''s contribution) K_\rho * Uworst on vi
		double KrhoU[3] = { 0. };
		// vertex self id in neihbor element i
		int vi = 7 - i;
		// vertex neighbor id in element i, traverse them and compute the corresponding node force on self
		for (int vj = 0; vj < 8; vj++) {
			int vjlid = gridPos2id<3>(i % 2 + vj % 2, i % 4 / 2 + vj % 4 / 2, i / 4 + vj / 4);
			double Uj[3];
			for (int k = 0; k < 3; k++) Uj[k] = gU[k][vneigh[vjlid]];
			for (int krow = 0; krow < 3; krow++) {
				for (int kcol = 0; kcol < 3; kcol++) {
					// KrhoU[krow] += KE[vi * 3 + krow][vj * 3 + kcol] * Uj[kcol];
                    KrhoU[krow] += (c11[eid] * KE11[vi * 3 + krow][vj * 3 + kcol] + c12[eid] * KE12[vi * 3 + krow][vj * 3 + kcol] + 
                                    c13[eid] * KE13[vi * 3 + krow][vj * 3 + kcol] + c22[eid] * KE22[vi * 3 + krow][vj * 3 + kcol] + 
                                    c23[eid] * KE23[vi * 3 + krow][vj * 3 + kcol] + c33[eid] * KE33[vi * 3 + krow][vj * 3 + kcol] + 
                                    c44[eid] * KE44[vi * 3 + krow][vj * 3 + kcol] + c55[eid] * KE55[vi * 3 + krow][vj * 3 + kcol] + 
                                    c66[eid] * KE66[vi * 3 + krow][vj * 3 + kcol]) * Uj[kcol];
				}
			}
		}

		// for (int k = 0; k < 3; k++) KrhoU[k] *= rho[eid];


		// sensitivity  - u_worst * dK/drho * u_worst
		for (int k = 0; k < 3; k++) partialSens -= Ui[k] * KrhoU[k];

		atomicAdd(sens + eid, float(partialSens));
	}
}

void computeSensitivity(void) {
	grids[0]->use_grid();
	// now, suppose Uworst, Fworst is prepared, N^T * Lambda is in U,
	// copy Fworst=KUworst to F
	//grids[0]->v3_copy(grids[0]->getWorstForce(), grids[0]->getForce());

	// init sensitivity to zero
	if(grids._useSpinodal)
	{
		// std::cout<<"not comlete yet"<<std::endl;
		int n_gs=grids[0]->n_nodes();
		init_array(grids[0]->_gbuf.g_sens, float{ 0 }, n_gs);
        init_array(grids[0]->_gbuf.g_sens_t1, float{ 0 }, n_gs);
        init_array(grids[0]->_gbuf.g_sens_t2, float{ 0 }, n_gs);
        init_array(grids[0]->_gbuf.g_sens_t3, float{ 0 }, n_gs);
		init_array(grids[0]->_gbuf.g_sens_vol, float{params.volume_ratio}, n_gs);
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n_gs, 512);
        // density filter
        computeSensitivity_spinodal_kernel << <grid_size, block_size >> > (n_gs, grids[0]->getRho(), grids[0]->_gbuf.g_sens_C11r, grids[0]->_gbuf.g_sens_C12r, grids[0]->_gbuf.g_sens_C13r, grids[0]->_gbuf.g_sens_C22r, grids[0]->_gbuf.g_sens_C23r, grids[0]->_gbuf.g_sens_C33r, grids[0]->_gbuf.g_sens_C44r, grids[0]->_gbuf.g_sens_C55r, grids[0]->_gbuf.g_sens_C66r, grids[0]->_gbuf.g_sens);
        cudaDeviceSynchronize();
	    cuda_error_check;

        computeSensitivity_spinodal_kernel << <grid_size, block_size >> > (n_gs, grids[0]->_gbuf.t1_e, grids[0]->_gbuf.g_sens_C11t1, grids[0]->_gbuf.g_sens_C12t1, grids[0]->_gbuf.g_sens_C13t1, grids[0]->_gbuf.g_sens_C22t1, grids[0]->_gbuf.g_sens_C23t1, grids[0]->_gbuf.g_sens_C33t1, grids[0]->_gbuf.g_sens_C44t1, grids[0]->_gbuf.g_sens_C55t1, grids[0]->_gbuf.g_sens_C66t1, grids[0]->_gbuf.g_sens_t1);
        cudaDeviceSynchronize();
        cuda_error_check;

        computeSensitivity_spinodal_kernel << <grid_size, block_size >> > (n_gs, grids[0]->_gbuf.t2_e, grids[0]->_gbuf.g_sens_C11t2, grids[0]->_gbuf.g_sens_C12t2, grids[0]->_gbuf.g_sens_C13t2, grids[0]->_gbuf.g_sens_C22t2, grids[0]->_gbuf.g_sens_C23t2, grids[0]->_gbuf.g_sens_C33t2, grids[0]->_gbuf.g_sens_C44t2, grids[0]->_gbuf.g_sens_C55t2, grids[0]->_gbuf.g_sens_C66t2, grids[0]->_gbuf.g_sens_t2);
        cudaDeviceSynchronize();
        cuda_error_check;

        computeSensitivity_spinodal_kernel << <grid_size, block_size >> > (n_gs, grids[0]->_gbuf.t3_e, grids[0]->_gbuf.g_sens_C11t3, grids[0]->_gbuf.g_sens_C12t3, grids[0]->_gbuf.g_sens_C13t3, grids[0]->_gbuf.g_sens_C22t3, grids[0]->_gbuf.g_sens_C23t3, grids[0]->_gbuf.g_sens_C33t3, grids[0]->_gbuf.g_sens_C44t3, grids[0]->_gbuf.g_sens_C55t3, grids[0]->_gbuf.g_sens_C66t3, grids[0]->_gbuf.g_sens_t3);
        cudaDeviceSynchronize();
        cuda_error_check;

		grids[0]->filterSensitivitySpinodal(params.filter_radius);
	}
	else
	{
		init_array(grids[0]->getSens(), float{ 0 }, grids[0]->n_rho());

		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, grids[0]->n_nodes(), 512);

		computeSensitivity_kernel << <grid_size, block_size >> > (grids[0]->n_nodes(), grids[0]->getRho(), grids[0]->_keyvalues["mu"], grids[0]->getSens());
		cudaDeviceSynchronize();
		cuda_error_check;

		// DEBUG
		//grids[0]->sens2matlab("sens");

		// filter sensitivity
		grids[0]->filterSensitivity(params.filter_radius);

		// DEBUG
		grids[0]->sens2matlab("sensfilt");
	}
}

__global__ void targetcompliance_sens_kernel(int nv, float* rho, float* c11, float* c12, float* c13, float* c22, float* c23, float* c33, float* c44, float* c55, float* c66, float* sens,double delta_c) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ float KE11[24][24];
    __shared__ float KE12[24][24];
    __shared__ float KE13[24][24];
    __shared__ float KE22[24][24];
    __shared__ float KE23[24][24];
    __shared__ float KE33[24][24];
    __shared__ float KE44[24][24];
    __shared__ float KE55[24][24];
    __shared__ float KE66[24][24];

	loadTemplateMatrix11(KE11);
    loadTemplateMatrix12(KE12);
    loadTemplateMatrix13(KE13);
    loadTemplateMatrix22(KE22);
    loadTemplateMatrix23(KE23);
    loadTemplateMatrix33(KE33);
    loadTemplateMatrix44(KE44);
    loadTemplateMatrix55(KE55);
    loadTemplateMatrix66(KE66);

	if (tid >= nv) return;

	int vid = tid;

	int vneigh[27];
	for (int i = 0; i < 27; i++) {
		vneigh[i] = gV2V[i][vid];
	}

	// traverse incident elements of vertex
	for (int i = 0; i < 8; i++) {
		double partialSens = 0;

		int eid = gV2E[i][vid];
		if (eid == -1) continue;
		double Ui[3] = { gU[0][vid],gU[1][vid],gU[2][vid] };
		// double penal = power_penalty[0] * powf(rholist[eid], power_penalty[0] - 1);

		// compute partial node force (element i''s contribution) K_\rho * Uworst on vi
		double KrhoU[3] = { 0. };
		// vertex self id in neihbor element i
		int vi = 7 - i;
		// vertex neighbor id in element i, traverse them and compute the corresponding node force on self
		for (int vj = 0; vj < 8; vj++) {
			int vjlid = gridPos2id<3>(i % 2 + vj % 2, i % 4 / 2 + vj % 4 / 2, i / 4 + vj / 4);
			double Uj[3];
			for (int k = 0; k < 3; k++) Uj[k] = gU[k][vneigh[vjlid]];
			for (int krow = 0; krow < 3; krow++) {
				for (int kcol = 0; kcol < 3; kcol++) {
					// KrhoU[krow] += KE[vi * 3 + krow][vj * 3 + kcol] * Uj[kcol];
                    KrhoU[krow] += (c11[eid] * KE11[vi * 3 + krow][vj * 3 + kcol] + c12[eid] * KE12[vi * 3 + krow][vj * 3 + kcol] + 
                                    c13[eid] * KE13[vi * 3 + krow][vj * 3 + kcol] + c22[eid] * KE22[vi * 3 + krow][vj * 3 + kcol] + 
                                    c23[eid] * KE23[vi * 3 + krow][vj * 3 + kcol] + c33[eid] * KE33[vi * 3 + krow][vj * 3 + kcol] + 
                                    c44[eid] * KE44[vi * 3 + krow][vj * 3 + kcol] + c55[eid] * KE55[vi * 3 + krow][vj * 3 + kcol] + 
                                    c66[eid] * KE66[vi * 3 + krow][vj * 3 + kcol]) * Uj[kcol];
				}
			}
		}

		// for (int k = 0; k < 3; k++) KrhoU[k] *= rho[eid];


		// sensitivity  - u_worst * dK/drho * u_worst
		for (int k = 0; k < 3; k++) partialSens -= Ui[k] * KrhoU[k]*2*delta_c;

		atomicAdd(sens + eid, float(partialSens));
	}
}

void targetComplianceSens(double c, double c0)
{
	grids[0]->use_grid();
	int n_gs=grids[0]->n_nodes();
	init_array(grids[0]->_gbuf.g_sens, float{ 0 }, n_gs);
	init_array(grids[0]->_gbuf.g_sens_t1, float{ 0 }, n_gs);
	init_array(grids[0]->_gbuf.g_sens_t2, float{ 0 }, n_gs);
	init_array(grids[0]->_gbuf.g_sens_t3, float{ 0 }, n_gs);
	init_array(grids[0]->_gbuf.g_sens_vol, float{params.volume_ratio}, n_gs);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gs, 512);

	// density filter
	targetcompliance_sens_kernel << <grid_size, block_size >> > (n_gs, grids[0]->getRho(), grids[0]->_gbuf.g_sens_C11r, grids[0]->_gbuf.g_sens_C12r, grids[0]->_gbuf.g_sens_C13r, grids[0]->_gbuf.g_sens_C22r, grids[0]->_gbuf.g_sens_C23r, grids[0]->_gbuf.g_sens_C33r, grids[0]->_gbuf.g_sens_C44r, grids[0]->_gbuf.g_sens_C55r, grids[0]->_gbuf.g_sens_C66r, grids[0]->_gbuf.g_sens,c-c0);
	cudaDeviceSynchronize();
	cuda_error_check;

	targetcompliance_sens_kernel << <grid_size, block_size >> > (n_gs, grids[0]->_gbuf.t1_e, grids[0]->_gbuf.g_sens_C11t1, grids[0]->_gbuf.g_sens_C12t1, grids[0]->_gbuf.g_sens_C13t1, grids[0]->_gbuf.g_sens_C22t1, grids[0]->_gbuf.g_sens_C23t1, grids[0]->_gbuf.g_sens_C33t1, grids[0]->_gbuf.g_sens_C44t1, grids[0]->_gbuf.g_sens_C55t1, grids[0]->_gbuf.g_sens_C66t1, grids[0]->_gbuf.g_sens_t1,c-c0);
	cudaDeviceSynchronize();
	cuda_error_check;

	targetcompliance_sens_kernel << <grid_size, block_size >> > (n_gs, grids[0]->_gbuf.t2_e, grids[0]->_gbuf.g_sens_C11t2, grids[0]->_gbuf.g_sens_C12t2, grids[0]->_gbuf.g_sens_C13t2, grids[0]->_gbuf.g_sens_C22t2, grids[0]->_gbuf.g_sens_C23t2, grids[0]->_gbuf.g_sens_C33t2, grids[0]->_gbuf.g_sens_C44t2, grids[0]->_gbuf.g_sens_C55t2, grids[0]->_gbuf.g_sens_C66t2, grids[0]->_gbuf.g_sens_t2,c-c0);
	cudaDeviceSynchronize();
	cuda_error_check;

	targetcompliance_sens_kernel << <grid_size, block_size >> > (n_gs, grids[0]->_gbuf.t3_e, grids[0]->_gbuf.g_sens_C11t3, grids[0]->_gbuf.g_sens_C12t3, grids[0]->_gbuf.g_sens_C13t3, grids[0]->_gbuf.g_sens_C22t3, grids[0]->_gbuf.g_sens_C23t3, grids[0]->_gbuf.g_sens_C33t3, grids[0]->_gbuf.g_sens_C44t3, grids[0]->_gbuf.g_sens_C55t3, grids[0]->_gbuf.g_sens_C66t3, grids[0]->_gbuf.g_sens_t3,c-c0);
	cudaDeviceSynchronize();
	cuda_error_check;

	grids[0]->filterSensitivitySpinodal(params.filter_radius);
}

__global__ void trySensMultiplier_kernel(
	int nv, const float* rholist, float* g_sens, float g_thres, float step, float damp, float rhomin, float* newrho) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= nv) return;

	int eid = gV2E[7][tid];

	if (eid == -1) { return; }

	float g = g_sens[eid];

	if (g > 0) g = 0;

	g = abs(g);

	float rhoold = rholist[eid];

	float rhonew = rhoold * powf(g / g_thres, damp);

	rhonew = clamp(rhonew, rhoold - step, rhoold + step);

	rhonew = clamp(rhonew, rhomin, 1.f);

	if (gEflag[0][eid] & grid::Grid::Bitmask::mask_shellelement) rhonew = 1;

	newrho[eid] = rhonew;
}

float updateDensities(float Vgoal) {
	grids[0]->use_grid();

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, grids[0]->n_nodes(), 512);

	float Vratio = 2;

	float g_thres_low = 0;
	float g_thres_upp = 1;

	// compute old volume ratio
	double* sum = (double*)grid::Grid::getTempBuf(sizeof(double) * grids[0]->n_rho() / 100);
	double Vold = parallel_sum_d(grids[0]->getRho(), sum, grids[0]->n_rho()) / grids[0]->n_rho();

	// compute maximal sensitivity
	float* maxdump = (float*)grid::Grid::getTempBuf(sizeof(float)* grids[0]->n_rho() / 100);
	float g_max = parallel_maxabs(grids[0]->getSens(), maxdump, grids[0]->n_rho());

	g_thres_upp = g_max;

	printf("[sensitivity] max = %f\n", g_max);

	float g_thres = (g_thres_low + g_thres_upp) / 2;

	// iteration counter
	int itn = 0;

	// bisection search sensitivity multiplier
	do  {
		// update sensitivity threshold
		g_thres = (g_thres_low + g_thres_upp) / 2;

		printf("-- searching multiplier g = %4.4e", g_thres);

		float* newrho = (float*)grid::Grid::getTempBuf(sizeof(float)* grids[0]->n_rho());

		// update new rho
		trySensMultiplier_kernel << <grid_size, block_size >> > (
			grids[0]->n_nodes(), grids[0]->getRho(), grids[0]->getSens(), g_thres, params.design_step, params.damp_ratio, params.min_rho, newrho);
		cudaDeviceSynchronize();
		cuda_error_check;

		// compute new volume ratio
		Vratio = dump_array_sum(newrho, grids[0]->n_rho()) / grids[0]->n_rho();

		printf(", V = %f  goal %f\n", Vratio, Vgoal);

		if (Vratio > Vgoal) {
			g_thres_low = g_thres;
		}
		else if (Vratio < Vgoal) {
			g_thres_upp = g_thres;
		}
	} while (abs(Vratio - Vgoal) > 1e-4 && itn++ < 30);

	// update densities according to new sensitivity
	trySensMultiplier_kernel << <grid_size, block_size >> > (grids[0]->n_nodes(), grids[0]->getRho(), grids[0]->getSens(), g_thres, params.design_step, params.damp_ratio, params.min_rho, grids[0]->getRho());
	cudaDeviceSynchronize();
	cuda_error_check;
	
	return g_thres;
}

void initSpinodalMMA(float*&xmin,float*&xmax,float*&xold1,float*&xold2,float*&df0dx,float*&gval,float*&dgdx,float*&low,float*&upp,float*&a,float*&c,float*&d)
{
	int ne_gs = grids[0]->n_gselements;
	int nvar=4*ne_gs;
	float pi = acos(-1.);
	int nconstrain = 1;
	cudaMalloc(&xmin, nvar * sizeof(float));
	cudaMalloc(&xmax, nvar * sizeof(float));
	cudaMalloc(&xold1, nvar * sizeof(float));
	cudaMalloc(&xold2, nvar * sizeof(float));
	cudaMalloc(&df0dx, nvar * sizeof(float));
	cudaMalloc(&gval, nconstrain * sizeof(float));
	cudaMalloc(&dgdx, nvar * sizeof(float));
	cudaMalloc(&low, nvar * sizeof(float));
	cudaMalloc(&upp, nvar * sizeof(float));
	cudaMalloc(&a, nconstrain * sizeof(float));
	cudaMalloc(&c, nconstrain * sizeof(float));
	cudaMalloc(&d, nconstrain * sizeof(float));
	init_array(xmin, float{0}, nvar);
	init_array(xmax, float{0.7}, ne_gs);
	init_array(xmax + ne_gs, float{pi / 2}, 3 * ne_gs);
	init_array(xold1, float{0}, nvar);
	init_array(xold2, float{0}, nvar);
	init_array(df0dx, float{0}, nvar);
	init_array(gval, float{0}, nconstrain);
	init_array(dgdx, float{0}, nvar);
	init_array(low, float{0}, nvar);
	init_array(upp, float{0}, nvar);
	init_array(a, float{0}, nconstrain);
	init_array(c, float{100000}, nconstrain);
	init_array(d, float{1}, nconstrain);
}

__global__ void spinodal_projection_kernel(int nv,float* rho,float*t1,float*t2,float*t3)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= nv) return;

	int eid = gV2E[7][tid];

	if (eid == -1) { return; }

	float rhoold = rho[eid];
	float t1old = t1[eid];
	float t2old = t2[eid];
	float t3old = t3[eid];

	rho[eid] = rhoold / (1 + exp(-600 * (rhoold - 0.3)));
	t1[eid] = max(t1old, 1.f / 6) / (1 + exp(-32400 * (t1old - 1.f / 24)));
	t2[eid] = max(t2old, 1.f / 6) / (1 + exp(-32400 * (t2old - 1.f / 24)));
	t3[eid] = max(t3old, 1.f / 6) / (1 + exp(-32400 * (t3old - 1.f / 24)));
}

#include <dlfcn.h> 
void updateDensitySpinodal(float* xvar, int itn, float Vgoal)
{
	grids[0]->use_grid();
	// prepare mma
	typedef void(*pmmasub)(int ncontrain, int nvar, int itn, float* xvar, float* xmin, float* xmax, float* xold1, float* xold2,
		float f0val, float* df0dx, float* gval, float* dgdx, float* low, float* upp,
		float a0, float* a, float* c, float* d, float move);
	static void* handle=dlopen("./libmma.so",RTLD_LAZY);
	static pmmasub mmasub=(pmmasub)dlsym(handle,"mmasub_from_g");
	if(!mmasub)
	{
		std::cout << dlerror() << std::endl;
		return;
	}
	int ne_gs = grids[0]->n_gselements;
	int nvar=4*ne_gs;
	float pi = acos(-1.);
	int nconstrain = 1;
	static float *xmin, *xmax, *xold1, *xold2, *df0dx, *gval, *dgdx, *low, *upp, *a, *c_mma, *d;
	static bool run_once = (initSpinodalMMA(xmin,xmax,xold1,xold2,df0dx,gval,dgdx,low,upp,a,c_mma,d), true);
	float move = 0.1f, a0 = 1.f;
	// init_array(dgdx, Vgoal, ne_gs);
	cudaMemcpy(dgdx, grids[0]->_gbuf.g_sens_vol, ne_gs * sizeof(float), cudaMemcpyDeviceToDevice);
	float *dump = (float *)grid::Grid::getTempBuf(sizeof(float) * ne_gs);
	float h_gval = parallel_sum(xvar, dump, ne_gs) / ne_gs / Vgoal - 1;
	cudaMemcpy(gval, &h_gval, sizeof(float), cudaMemcpyHostToDevice);
	printf("V=%f Vgoal=%f\n", (h_gval + 1) * Vgoal, Vgoal);

	cudaMemcpy(df0dx, grids[0]->_gbuf.g_sens, ne_gs * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(df0dx + ne_gs, grids[0]->_gbuf.g_sens_t1, ne_gs * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(df0dx + 2 * ne_gs, grids[0]->_gbuf.g_sens_t2, ne_gs * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(df0dx + 3 * ne_gs, grids[0]->_gbuf.g_sens_t3, ne_gs * sizeof(float), cudaMemcpyDeviceToDevice);

	// saveGpuVec(grids.getPath("sens_t1.txt"), df0dx, ne_gs);
	// saveGpuVec(grids.getPath("sens_t2.txt"), df0dx + ne_gs, ne_gs);
	// saveGpuVec(grids.getPath("sens_t3.txt"), df0dx + 2 * ne_gs, ne_gs);
	// saveGpuVec(grids.getPath("sens_t3.txt"), df0dx + 3 * ne_gs, ne_gs);

	mmasub(nconstrain, nvar, itn, xvar, xmin, xmax, xold1, xold2, 0, df0dx, gval, dgdx, low, upp, a0, a, c_mma, d, move);

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, grids[0]->n_nodes(), 512);
	spinodal_projection_kernel<<<grid_size, block_size>>>(ne_gs, xvar, xvar + ne_gs, xvar + 2 * ne_gs, xvar + 3 * ne_gs);
	cudaDeviceSynchronize();
	cuda_error_check;
}

extern void matlab_utils_test(void);

void selfTest(void)
{
	printf("-- Self testing...\n");
	using namespace grid;

	std::vector<int> arr(10000, 0xaaaaaaaa);

	BitSAT<int> bits(arr);

	printf("-- host bits total = %d\n", bits.total());

	gBitSAT<int> gbits(bits._bitArray, bits._chunkSat);

	{
		std::vector<int> bitscheck(bits._bitArray.size());
		cudaMemcpy(bitscheck.data(), gbits._bitarray, sizeof(int) * bitscheck.size(), cudaMemcpyDeviceToHost);
		//printf("%p ->\n", gbits._bitarray);
		//for (int i = 0; i < bitscheck.size(); i++) printf("%d ", bitscheck[i]);
		//printf("\n");
	}

	devArray_t<int*, 1> gcount;
	cudaMalloc(&gcount[0], bits._bitArray.size() * BitCount<int>::value * sizeof(int));

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, bits._bitArray.size() * BitCount<int>::value, 128);

	auto kernel = [=] __device__(int tid) {
		if (blockIdx.x == 0 && threadIdx.x == 0) {
			//printf("_bitarray = %p ; _chunksat = %p\n", gbits._bitarray, gbits._chunksat);
			//printf("firstOne = %d\n", ::firstOne<sizeof(int) * 8>::value);
		}
		gcount[0][tid] = gbits[tid];
	};
	traverse_noret << <grid_size, block_size >> > (bits._bitArray.size() * BitCount<int>::value, kernel);
	cudaDeviceSynchronize();
	cuda_error_check;

	std::vector<int> counts(bits._bitArray.size() * BitCount<int>::value);
	cudaMemcpy(counts.data(), gcount[0], sizeof(int) * bits._bitArray.size() * BitCount<int>::value, cudaMemcpyDeviceToHost);

	bool pass_test = true;
	for (int i = 0; i < counts.size(); i++) {
		//printf("[%d] : %d\n", i, counts[i]);
		if (counts[i] != bits[i]) {
			pass_test = false;
			break;
		}
	}

	gbits.destroy();
	gcount.destroy();

	cuda_error_check;

	matlab_utils_test();

	// test GraftArray
	{
		std::vector<int> hostbuf(10000);
		int* _bufdev;
		cudaMalloc(&_bufdev, sizeof(int)*hostbuf.size());
		int baselen = hostbuf.size();
		auto kern = [=] __device__(int tid) {
			int rl = baselen / 125;
			GraftArray<int, 25, 5> p(_bufdev, rl);
			int id[3] = { tid % rl, tid / rl % 5,tid / rl / 5 };
			p[id[2]][id[1]][id[0]] = tid;
		};
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, hostbuf.size(), 512);
		traverse_noret << <grid_size, block_size >> > (hostbuf.size(), kern);
		cudaDeviceSynchronize();
		cuda_error_check;
		cudaMemcpy(hostbuf.data(), _bufdev, sizeof(int) * hostbuf.size(), cudaMemcpyDeviceToHost);

		bool fail = false;
		for (int i = 0; i < hostbuf.size(); i++) {
			if (hostbuf[i] != i) {
				fail = true; break;
			}
		}
		if (fail) {
			printf("-- GraftArray test failed\n");
		}
		pass_test &= !fail;

		cudaFree(_bufdev);
	}

	if (pass_test) {
		printf("-- Pass test\n");
	}
	else {
		printf("-- Test failed\n");
	}

}

// upload template matrix and power penalty coefficient
void uploadTemplateMatrix(void)
{
	double element_len = grids.elementLength();
	if(params.use_spinodal)
	{
		initTemplateMatrixOrthotropy(element_len,gpu_manager);
		const double* ke11 = getTemplateMatrixElements11();
		const double* ke12 = getTemplateMatrixElements12();
		const double* ke13 = getTemplateMatrixElements13();
		const double* ke22 = getTemplateMatrixElements22();
		const double* ke23 = getTemplateMatrixElements23();
		const double* ke33 = getTemplateMatrixElements33();
		const double* ke44 = getTemplateMatrixElements44();
		const double* ke55 = getTemplateMatrixElements55();
		const double* ke66 = getTemplateMatrixElements66();
		cudaMemcpyToSymbol(gTemplateMatrix11, ke11, sizeof(gTemplateMatrix11));
		cudaMemcpyToSymbol(gTemplateMatrix12, ke12, sizeof(gTemplateMatrix12));
		cudaMemcpyToSymbol(gTemplateMatrix13, ke13, sizeof(gTemplateMatrix13));
		cudaMemcpyToSymbol(gTemplateMatrix22, ke22, sizeof(gTemplateMatrix22));
		cudaMemcpyToSymbol(gTemplateMatrix23, ke23, sizeof(gTemplateMatrix23));
		cudaMemcpyToSymbol(gTemplateMatrix33, ke33, sizeof(gTemplateMatrix33));
		cudaMemcpyToSymbol(gTemplateMatrix44, ke44, sizeof(gTemplateMatrix44));
		cudaMemcpyToSymbol(gTemplateMatrix55, ke55, sizeof(gTemplateMatrix55));
		cudaMemcpyToSymbol(gTemplateMatrix66, ke66, sizeof(gTemplateMatrix66));
	}
	initTemplateMatrix(element_len, gpu_manager, params.youngs_modulu, params.poisson_ratio);
	const double* ke = getTemplateMatrixElements();
	cudaMemcpyToSymbol(gTemplateMatrix, ke, sizeof(gTemplateMatrix));
	cuda_error_check;

	// upload power penalty
	float power = params.power_penalty;
	cudaMemcpyToSymbol(power_penalty, &power, sizeof(power_penalty));
	cuda_error_check;
}

void setDEBUG(bool debug)
{
	int a = 0;
	if (debug) a = 1;
	cudaMemcpyToSymbol(gDEBUG, &a, sizeof(int));
}


__global__ void checkAjointKernel(int n_gsvertices, double mu, devArray_t<double*, 3> vdst) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n_gsvertices) return;

	for (int i = 0; i < 3; i++) {
		double kku = gF[i][tid];
		double f = vdst[i][tid];
		double fworst = gFworst[i][tid];
		vdst[i][tid] = 2 * fworst - f - 2 * mu * kku;
	}
}

// supporst N^T lambda is in U,  uworst is in Uworst, Kuworst is in Fworst
bool checkAdjointVariable(void) {
	grids[0]->use_grid();
	devArray_t<double*, 3> vdst, uback;
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&vdst[i], sizeof(double) * grids[0]->n_gsvertices);
		cudaMalloc(&uback[i], sizeof(double) * grids[0]->n_gsvertices);
	}
	// backup adjoint displacement
	grids[0]->v3_copy(grids[0]->getDisplacement(), uback._data);
	// KNlam in vdst
	grids[0]->applyK(grids[0]->getDisplacement(), grids[0]->getForce());
	grids[0]->v3_copy(grids[0]->getForce(), vdst._data);
	grids[0]->resetDirchlet(vdst._data);

	// KKu is in F
	grids[0]->v3_copy(grids[0]->getWorstForce(), grids[0]->getDisplacement());
	grids[0]->applyK(grids[0]->getDisplacement(), grids[0]->getForce());
	grids[0]->resetDirchlet(grids[0]->getForce());
	double mu = grids[0]->_keyvalues["mu"];
	
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, grids[0]->n_gsvertices, 512);
	checkAjointKernel<<<grid_size,block_size>>>(grids[0]->n_gsvertices, mu, vdst);
	cudaDeviceSynchronize();
	cuda_error_check;
	grids[0]->v3_toMatlab("vadj", vdst._data);
	bool vnorm = grids[0]->v3_norm(vdst._data);
	bool passCheck = vnorm < 1e-6;

	grids[0]->v3_copy(uback._data, grids[0]->getDisplacement());

	vdst.destroy();
	uback.destroy();

	return passCheck;
}


