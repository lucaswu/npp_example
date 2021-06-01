#include <cuda_runtime.h>
#include <npp.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

static void copyYUVCPU2GPU(Npp8u*pDst,uint8_t*pSrcY,uint8_t*pSrcU,uint8_t*pSrcV,int width,int height)
{
    if(pDst == nullptr || pSrcY == nullptr || pSrcU == nullptr || pSrcV == nullptr){
        return;
    }

    uint8_t*pTemp = new uint8_t[width*height*3];
    memcpy(pTemp,pSrcY,width*height);

    uint8_t *pTempDst = pTemp + width*height;
    uint8_t *pTempSrc = pSrcU;
    for(int i=0;i<height/2;i++){
        memcpy(pTempDst,pTempSrc,width/2);
        pTempDst += width;
        pTempSrc += width/2;
    }

    pTempDst = pTemp + width*height*2;
    pTempSrc = pSrcV;
    for(int i=0;i<height/2;i++){
        memcpy(pTempDst,pTempSrc,width/2);
        pTempDst += width;
        pTempSrc += width/2;
    }
  
    cudaMemcpy(pDst,pTemp,width*height*3,cudaMemcpyHostToDevice);

    delete[] pTemp;
}

int main(int argc,char** argv)
{
    if(argc != 4){
        printf("usage: yuv2rgb input.yuv  width height");
        return 0;
    }

    char *file_yuv = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);

    size_t srcSize = width * height * 3 / 2;
    uint8_t* pInData = new uint8_t[width*height*3/2];


    Npp8u *pYUV_dev ;
    cudaMalloc((void**)&pYUV_dev,width*height*3*sizeof(Npp8u));

    Npp8u *pRGB_dev;
    cudaMalloc((void**)&pRGB_dev,width*height*3*sizeof(Npp8u));

    FILE* fp = fopen(file_yuv, "rb");
    if (fp == NULL){
        printf("open %s error!\n",file_yuv);
        return 0;
    }


    int i = 0 ;
    while (fread(pInData, 1, srcSize, fp) == srcSize)
    {
        uint8_t* pY = pInData;
        uint8_t* pU = pY + width * height;
        uint8_t* pV = pU + width * height / 4;
        
        copyYUVCPU2GPU(pYUV_dev,pY,pU,pV,width,height);

        NppiSize nppSize = {width,height};

        printf("[%s:%d],nppSize(%d,%d)\n",__FILE__,__LINE__,nppSize.width,nppSize.height);
        auto ret = nppiYUVToRGB_8u_P3R(pYUV_dev,width*3,pRGB_dev,width*3,nppSize);
        if(ret != 0 ){
            printf("nppiYUVToRGB_8u_C3R error:%d\n",ret);
            return 0;
        }
        cv::Mat img(height, width, CV_8UC3);
        cudaMemcpy(img.data,pRGB_dev,width*height*3,cudaMemcpyDeviceToHost);
        std::string name = std::to_string(i)+".png";
        cv::imwrite(name.c_str(),img);

        i++;
        if(i>0){
            break;
        }

    }

    delete[] pInData;
    cudaFree(pYUV_dev);
    cudaFree(pRGB_dev);
    fclose(fp);

}