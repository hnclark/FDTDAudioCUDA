#include<stdlib.h>
#include<stdio.h>
#include<cmath>
#include<fstream>



//shared host/device constants
int gridWidth,gridHeight,gridDepth,blockWidth,blockHeight,gridWidthBlocks,gridHeightBlocks,gridArea;
__constant__ int gridWidth_d,gridHeight_d,gridDepth_d,blockWidth_d,blockHeight_d,gridWidthBlocks_d,gridHeightBlocks_d;

//host only constants
int timeSteps;



__device__ int arrayPos(const int &x,const int &y,const int &z){
    return (z*gridWidth_d*gridHeight_d)+(y*gridWidth_d)+x;
}



__global__ void solver(bool *grid,bool *grid1){
    int xpos = ((blockIdx.x%gridWidthBlocks_d)*blockWidth_d)+threadIdx.x;
    int ypos = (blockIdx.y*blockHeight_d)+threadIdx.y;
    int zpos = blockIdx.x/gridWidthBlocks_d;

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
    //time of simulation
    timeSteps = 1;

    //dimensions of the grid
    gridWidth = 16;
    gridHeight = 16;
    gridDepth = 16;

    //find the most efficient block_width/height for the kernel(probably 16x16)
    blockWidth = 16;
    blockHeight = 16;

    //derived values
    gridWidthBlocks = std::ceil((float)gridWidth/(float)blockWidth);
    gridHeightBlocks = std::ceil((float)gridHeight/(float)blockHeight);
    gridArea = gridWidth*gridHeight*gridDepth;

    //set device symbols to dimensions of grid,block,etc.
    cudaMemcpyToSymbol(*(&gridWidth_d),&gridWidth,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridHeight_d),&gridHeight,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridDepth_d),&gridDepth,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&blockWidth_d),&blockWidth,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&blockHeight_d),&blockHeight,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridWidthBlocks_d),&gridWidthBlocks,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&gridHeightBlocks_d),&gridHeightBlocks,sizeof(int),0,cudaMemcpyHostToDevice);

    printf("%d %d %d %d %d %d %d\n",gridWidth,gridHeight,gridDepth,blockWidth,blockHeight,gridWidthBlocks,gridHeightBlocks);

    dim3 numBlocks(gridWidthBlocks*gridDepth,gridHeightBlocks,1);
    dim3 blockSize(blockWidth,blockHeight,1);

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
}