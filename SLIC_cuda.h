//
// Created by derue on 15/12/15.
//

#ifndef SLIC_CUDA_SLIC_CUDA_H
#define SLIC_CUDA_SLIC_CUDA_H
#endif //SLIC_CUDA_SLIC_CUDA_H

/*
 * Written by Derue Francois-Xavier
 * francois-xavier.derue@polymtl.ca
 *
 * Superpixel oversegmentation
 * GPU implementation of the algorithm SLIC of
 * Achanta et al. [PAMI 2012, vol. 34, num. 11, pp. 2274-2282]
 *
 * Library required :
 * CUDA
 */

#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "funUtils.h"


#define N_ITER 5 // Kmean iteration
#define NMAX_THREAD 1024 // depend on gpu
#define BLOCK_DIM 16


class SLIC_cuda {

private:
    int m_nPx;
    int m_nSpx;
    int m_diamSpx;
    int m_wSpx, m_hSpx, m_areaSpx;
    int m_width, m_height;
    float m_wc;

    //cpu buffer
    float *m_clusters;
    float *m_labels;

    // gpu variable
   // uchar4* frameBGRA_g;
    //float4* frameLab_g;
    float* labels_g;
    float* clusters_g;
    float* accAtt_g; //accumulator for atomic
	float* accum_map; //accumulator non atomic

    //cudaArray

    cudaArray* frameBGRA_array;
    cudaArray* frameLab_array;
    cudaArray* labels_array;

    //texture object
    cudaTextureObject_t frameBGRA_tex;
    cudaSurfaceObject_t frameLab_surf;
    cudaSurfaceObject_t labels_surf;



    //========= methods ===========
    // init centroids uniformly on a grid spaced by diamSpx
    void InitClusters();
    //=subroutine =
    void InitBuffers(); // allocate buffers on gpu
    void SendFrame(cv::Mat& frameLab); //transfer frame to gpu buffer
    void Rgb2CIELab( cudaTextureObject_t inputImg, cudaSurfaceObject_t outputImg, int width, int height );


    //===== Kernel Invocation ======
    void Assignement(); //Assignment
    void Update(); // Update


public:
    SLIC_cuda(int diamSpx, float wc);
    ~SLIC_cuda();

    void Initialize(cv::Mat& frame0);
    void Segment(cv::Mat& frame); // gpu superpixel segmentation
	void GetLabels();
	void EnforceConnectivity();

    //===== Display function =====
    void displayBound(cv::Mat& image, cv::Scalar colour); // cpu draw

    // enforce connectivity between superpixel, discard orphan (optional)
    // implementation from Pascal Mettes : https://github.com/PSMM/SLIC-Superpixels
    void enforceConnectivity();

};

__global__ void k_initClusters(cudaSurfaceObject_t frameLab,float* clusters,int width, int height, int nSpxPerRow, int nSpxPerCol);
__global__ void k_assignement(int width, int height,int wSpx, int hSpx,cudaSurfaceObject_t frameLab, cudaSurfaceObject_t labels,float* clusters,float* accAtt,float wc2);
__global__ void k_update(int nSpx,float* clusters, float* accAtt);
