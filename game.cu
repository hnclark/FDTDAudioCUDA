#include<stdlib.h>
#include<stdio.h>
#include<cmath>
#include<fstream>
#include<chrono>



//shared host/device constants
int gridWidth,gridHeight,gridDepth,blockWidth,blockHeight,blockDepth,gridWidthBlocks,gridHeightBlocks,gridDepthBlocks,gridArea;
__constant__ int gridWidth_d,gridHeight_d,gridDepth_d,blockWidth_d,blockHeight_d,blockDepth_d,gridWidthBlocks_d,gridHeightBlocks_d,gridDepthBlocks_d;

//host only constants
int timeSteps;



__device__ int arrayPos(const int &x,const int &y,const int &z){
    return (z*gridWidth_d*gridHeight_d)+(y*gridWidth_d)+x;
}



__global__ void solver(bool *grid,bool *grid1){
    int xpos = (blockIdx.x*blockWidth_d)+threadIdx.x;
    int ypos = (blockIdx.y*blockHeight_d)+threadIdx.y;
    int zpos = (blockIdx.z*blockDepth_d)+threadIdx.z;

    if(xpos>0 && xpos<gridWidth_d-1 && ypos>0 && ypos<gridHeight_d-1 && zpos>0 && zpos<gridDepth_d-1){
        int neighbors = grid[arrayPos(xpos+1,ypos,zpos)]+grid[arrayPos(xpos-1,ypos,zpos)]
                        +grid[arrayPos(xpos,ypos+1,zpos)]+grid[arrayPos(xpos,ypos-1,zpos)]
                        +grid[arrayPos(xpos,ypos,zpos+1)]+grid[arrayPos(xpos,ypos,zpos-1)];

        if(grid[arrayPos(xpos,ypos,zpos)]){
            if(neighbors<2 || neighbors>3){
                grid1[arrayPos(xpos,ypos,zpos)] = false;
            }else{
                grid1[arrayPos(xpos,ypos,zpos)] = true;
            }
        }else{
            if(neighbors==3){
                grid1[arrayPos(xpos,ypos,zpos)] = true;
            }else{
                grid1[arrayPos(xpos,ypos,zpos)] = false;
            }
        }
    }
}



//helper function to read grid from a text file
void readTextRepr(const std::string& filename,bool *array){
    std::ifstream file(filename);
    std::string str;
    int index=0;

    while(std::getline(file,str)){
        if(str!="---"){
            for(int i=0;i<str.length();i++){
                //stop reading if file is greater than arrayLength
                if(index<gridArea){
                    if(str[i]!='\n'){
                        if(str[i]=='#'){
                            array[index]=true;
                        }else{
                            array[index]=false;
                        }
                        index++;
                    }
                }
            }
        }
    }
    //fill in excess space with falses if file is too short
    if(index<gridArea){
        for(int i=index;i<gridArea;i++){
            array[index]=false;
        }
    }
}



//helper function to write grid to a text file
void writeTextRepr(const std::string& filename,bool *array){
    std::ofstream file(filename);
    for(int i=0;i<gridArea;i++){
        if(array[i]){
            file<<'#';
        }else{
            file<<' ';
        }
        if((i+1)%gridWidth==0){
            file<<'\n';
        }
        if((i+1)%(gridWidth*gridHeight)==0){
            file<<"---\n";
        }
    }
}



int main(int argc, const char * argv[]){
    //start clock
    auto startTime = std::chrono::high_resolution_clock::now();

    //time of simulation
    timeSteps = 100;

    //dimensions of the grid
    gridWidth = 2048;
    gridHeight = 2048;
    gridDepth = 128;

    /*
    Speed testing with varying grid and block sizes, using both 3D and 2D kernel implementations
        16x16x16 grid, 2mil steps:
            3D Kernel
            16x16x1 = 18921ms
            8x8x16  = 21150ms
            8x8x8   = 19432ms
            4x4x4   = 21614ms
            2x2x2   = 24362ms

            2D Kernel
            16x16   = 19389ms
            8x8     = 19425ms
            4x4     = 19992ms

        64x64x64 grid, 2mil steps:
            3D Kernel
            16x16x1 = 44162ms
            8x8x8   = 61221ms
            8x8x4   = 62442ms
            
            2D Kernel
            16x16   = 47543ms

        2048x2048x128 grid, 100 steps:
            3D Kernel
            16x16x1 = 15065ms
            8x8x8   = 18386ms

    The fastest block size across both small and large grids appears to be 16x16x1, using the 3D kernel
    */
    blockWidth = 16;
    blockHeight = 16;
    blockDepth = 1;

    //derived values
    gridWidthBlocks = std::ceil((float)gridWidth/(float)blockWidth);
    gridHeightBlocks = std::ceil((float)gridHeight/(float)blockHeight);
    gridDepthBlocks = std::ceil((float)gridDepth/(float)blockDepth);
    gridArea = gridWidth*gridHeight*gridDepth;

    //set device symbols to dimensions of grid,block,etc.
    cudaMemcpyToSymbol(*(&gridWidth_d),&gridWidth,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridHeight_d),&gridHeight,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridDepth_d),&gridDepth,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&blockWidth_d),&blockWidth,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&blockHeight_d),&blockHeight,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&blockDepth_d),&blockDepth,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridWidthBlocks_d),&gridWidthBlocks,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridHeightBlocks_d),&gridHeightBlocks,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridDepthBlocks_d),&gridDepthBlocks,sizeof(int),0,cudaMemcpyHostToDevice);

    printf("grid dim = %dx%dx%d\n",gridWidth,gridHeight,gridDepth);
    printf("block dim = %dx%dx%d\n",blockWidth,blockHeight,blockDepth);
    printf("grid in blocks = %dx%dx%d\n",gridWidthBlocks,gridHeightBlocks,gridDepthBlocks);

    dim3 numBlocks(gridWidthBlocks,gridHeightBlocks,gridDepthBlocks);
    dim3 blockSize(blockWidth,blockHeight,blockDepth);

    size_t gridSize = gridWidth*gridHeight*gridDepth;

    //device+host grid arrays
    bool *grid_h;
    bool *grid_d,*grid1_d;

    //allocate host memory
    grid_h = (bool *)calloc(gridSize,sizeof(bool));

    //allocate device memory
    cudaMalloc((void **)&grid_d, gridSize);
    cudaMalloc((void **)&grid1_d, gridSize);

    //load grid
    readTextRepr("in.txt",grid_h);
    
    //only copy first grid to device, second one is computed by kernel
    cudaMemcpy(grid_d,grid_h,gridSize,cudaMemcpyHostToDevice);

    for(int i=0;i<timeSteps;i++){
        solver<<<numBlocks,blockSize>>>(grid_d,grid1_d);
        cudaDeviceSynchronize();
        std::swap(grid_d,grid1_d);

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("%s",cudaGetErrorString(error));
        }
    }

    //only copy first grid to host, since it was computed and then swapped by kernel
    cudaMemcpy(grid_h,grid_d,gridSize,cudaMemcpyDeviceToHost);

    //output grid
    writeTextRepr("out.txt",grid_h);

    //free host memory
    free(grid_h);
    //free device memory
    cudaFree(grid_d);
    cudaFree(grid1_d);

    //end clock
    auto endTime = std::chrono::high_resolution_clock::now();
    auto timePassed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    printf("Ran in %ld ms\n",timePassed);
}