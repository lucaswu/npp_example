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

    size_t size = height*width*3*sizeof(uchar);
    uchar *inData = (uchar*)malloc(size);
    uchar *outData = (uchar*)malloc(size*16);

#ifdef PLAN
    uchar*pRed 		= inData;
    uchar*pGreen 	= pRed+width*height;
    uchar*pBlue 	= pGreen + width*height;
    uchar* pSrc = img.data;
    for(int i=0;i<width*height;i++){
    	*pRed++ 	= *pSrc++;
    	*pGreen++ 	= *pSrc++;
    	*pBlue++ 	= *pSrc++;
    }
#else
    memcpy(inData,img.data,size);
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
    pRed 	= (uchar*)gpuSrc;
    pGreen 	= (uchar*)pRed+width*height;
    pBlue 	= (uchar*)pGreen + width*height;
    const Npp8u* srcPtr[3] = {(Npp8u*)pRed,(Npp8u*)pGreen,(Npp8u*)pBlue};

    pRed 	= (uchar*)gpuDst;
    pGreen 	= (uchar*)pRed+width*height*16;
    pBlue 	= (uchar*)pGreen + width*height*16;
    Npp8u* dstPtr[3] = {(Npp8u*)pRed,(Npp8u*)pGreen,(Npp8u*)pBlue};

    nppiResize_8u_P3R(srcPtr,width,srcSize,srcROI,dstPtr,width*4,{width*4, height*4},dstROI,NPPI_INTER_LINEAR);
#else
    nppiResize_8u_C3R((Npp8u*)gpuSrc,width*3,srcSize,srcROI,(Npp8u*)gpuDst,width*4*3,{width*4, height*4},dstROI,NPPI_INTER_LINEAR);
#endif
    cudaMemcpy(outData,gpuDst,size*16,cudaMemcpyDeviceToHost);

#ifdef PLAN
    pRed 	= outData;
    pGreen 	= pRed+width*height*16;
    pBlue 	= pGreen + width*height*16;
    uchar* pDst = outMat.data;
    for(int i=0;i<width*height*16;i++){
    	*pDst++ = *pRed++;
    	*pDst++ = *pGreen++;
        *pDst++ = *pBlue++;

    }
#else
    memcpy(outMat.data,outData,size*16);
#endif
    imwrite(argv[2],outMat);


    free(inData);
    free(outData);

    cudaFree(gpuSrc);
    cudaFree(gpuDst);
	return 0;
}
