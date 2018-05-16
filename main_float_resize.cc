#include <cuda_runtime.h>
#include <npp.h>
#include<stdio.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define uchar unsigned char
using namespace cv;

#define PLAN
int main(int argc, char**argv)
{
    cv::Mat img = imread(argv[1]);
    int height = img.rows;
    int width = img.cols;
    
    cv::Mat outMat(height*4, width*4, CV_8UC3);

    printf("Input:(%d,%d)\n",width,height);

    size_t size = height*width*3*sizeof(float);
    float *inData = (float*)malloc(size);
    float *outData = (float*)malloc(size*16);

#ifdef PLAN
    float*pRed 		= inData;
    float*pGreen 	= pRed+width*height;
    float*pBlue 	= pGreen + width*height;
    uchar* pSrc = img.data;
    for(int i=0;i<width*height;i++){
    	*pRed++ 	= *pSrc++;
    	*pGreen++ 	= *pSrc++;
    	*pBlue++ 	= *pSrc++;
    }
#else
    uchar* pSrc = img.data;
    for(int i=0;i<height*width*3;i++){
    	inData[i] = pSrc[i];
    }
#endif



    void *gpuSrc ,*gpuDst;
    cudaMalloc(&gpuSrc,size);
    cudaMalloc(&gpuDst,size*16);
    cudaMemcpy(gpuSrc,inData,size,cudaMemcpyHostToDevice);

    NppiSize srcSize = {width, height};
    NppiRect srcROI = {0,0,width, height};
    NppiRect dstROI;
    nppiGetResizeRect(srcROI,&dstROI,4.0,4.0,0.0,0.0,NPPI_INTER_LINEAR);
    printf("dstROI:(%d,%d,%d,%d)\n",dstROI.x,dstROI.y,dstROI.width,dstROI.height);


#ifdef PLAN
    pRed 	= (float*)gpuSrc;
    pGreen 	= (float*)pRed+width*height;
    pBlue 	= (float*)pGreen + width*height;
    const Npp32f* srcPtr[3] = {(Npp32f*)pRed,(Npp32f*)pGreen,(Npp32f*)pBlue};

    pRed 	= (float*)gpuDst;
    pGreen 	= (float*)pRed+width*height*16;
    pBlue 	= (float*)pGreen + width*height*16;
    Npp32f* dstPtr[3] = {(Npp32f*)pRed,(Npp32f*)pGreen,(Npp32f*)pBlue};

    nppiResize_32f_P3R(srcPtr,width*4,srcSize,srcROI,dstPtr,width*4*4,{width*4, height*4},dstROI,NPPI_INTER_LINEAR);
#else
    nppiResize_32f_C3R((Npp32f*)gpuSrc,width*3*4,srcSize,srcROI,(Npp32f*)gpuDst,width*4*3*4,{width*4, height*4},dstROI,NPPI_INTER_LINEAR);
#endif
    cudaMemcpy(outData,gpuDst,size*16,cudaMemcpyDeviceToHost);

#ifdef PLAN
    pRed 	= outData;
    pGreen 	= pRed+width*height*16;
    pBlue 	= pGreen + width*height*16;
    uchar* pDst = outMat.data;
    for(int i=0;i<width*height*16;i++){
    	*pDst++ = (uchar)(*pRed++);
    	*pDst++ = (uchar)(*pGreen++);
        *pDst++ = (uchar)(*pBlue++);

    }
#else
    uchar* pDst = outMat.data;
    for(int i=0;i<height*width*3*16;i++){
    	pDst[i] = (uchar)(outData[i]);
    }
#endif
    imwrite(argv[2],outMat);


    free(inData);
    free(outData);

    cudaFree(gpuSrc);
    cudaFree(gpuDst);
    return 0;
}
