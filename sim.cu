#include<stdlib.h>
#include<iostream>
#include<cmath>
#include<fstream>

#include<chrono>

#define SIM_STATE_NAME "/sim_state.bin"



//shared host/device constants
int gridWidth,gridHeight,gridDepth,blockWidth,blockHeight,blockDepth,gridWidthBlocks,gridHeightBlocks,gridDepthBlocks,gridArea;
__constant__ int gridWidth_d,gridHeight_d,gridDepth_d,blockWidth_d,blockHeight_d,blockDepth_d,gridWidthBlocks_d,gridHeightBlocks_d,gridDepthBlocks_d;

//host only constants
int timeSteps;



__device__ int arrayPos(const int &x,const int &y,const int &z){
    return (z*gridWidth_d*gridHeight_d)+(y*gridWidth_d)+x;
}



__global__ void solver(double *grid,double *grid1){
    int xpos = (blockIdx.x*blockWidth_d)+threadIdx.x;
    int ypos = (blockIdx.y*blockHeight_d)+threadIdx.y;
    int zpos = (blockIdx.z*blockDepth_d)+threadIdx.z;

    if(xpos>0 && xpos<gridWidth_d-1 && ypos>0 && ypos<gridHeight_d-1 && zpos>0 && zpos<gridDepth_d-1){
        //the grid1 value at each point contains the value from 2 steps ago(but is overwritten once used)
        grid1[arrayPos(xpos,ypos,zpos)] = grid[arrayPos(xpos+1,ypos,zpos)]+grid[arrayPos(xpos-1,ypos,zpos)]
                                        +grid[arrayPos(xpos,ypos+1,zpos)]+grid[arrayPos(xpos,ypos-1,zpos)]
                                        +grid[arrayPos(xpos,ypos,zpos+1)]+grid[arrayPos(xpos,ypos,zpos-1)]
                                        -grid1[arrayPos(xpos,ypos,zpos)];
    }
}



//helper function to read header from a binary file
void readHeaderBinary(FILE *fileIn,int *w,int *h,int *d){
    fread(w,sizeof(int),1,fileIn);
    fread(h,sizeof(int),1,fileIn);
    fread(d,sizeof(int),1,fileIn);
}

//helper function to write header to a binary file
void writeHeaderBinary(FILE *fileOut,int *w,int *h,int *d){
    fwrite(w,sizeof(int),1,fileOut);
    fwrite(h,sizeof(int),1,fileOut);
    fwrite(d,sizeof(int),1,fileOut);
}

//helper function to read grid from a binary file
void readDoublesBinary(FILE *fileIn,double *array,int arrayLen){
    fread(array,sizeof(double),arrayLen,fileIn);
}

//helper function to write grid to a binary file
void writeDoublesBinary(FILE *fileOut,double *array,int arrayLen){
    fwrite(array,sizeof(double),arrayLen,fileOut);
}



//helper function to read grid from a text file
void readTextRepr(const std::string& filename,double *array){
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
                            array[index]=1;
                        }else{
                            array[index]=0;
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
void writeTextRepr(const std::string& filename,double *array){
    std::ofstream file(filename);
    for(int i=0;i<gridArea;i++){
        if(array[i]>0){
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



//check for and print cuda errors
void checkCudaError(){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::cout << cudaGetErrorString(error) << std::endl;
    }
}



int main(int argc, const char * argv[]){
    //start clock
    auto startTime = std::chrono::high_resolution_clock::now();

    //default input and output files
    std::string inFolder = "input";
    std::string outFolder = "output";

    //default simulation time
    timeSteps = 1;

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

    //default grid configuration
    gridWidth = 16;
    gridHeight = 16;
    gridDepth = 16;



    //handle command line arguments to modify default configuration
    int optionLen = 0;
    for(int i=1;i<argc;i+=optionLen){
        if(strcmp(argv[i],"-i")==0){
            optionLen = 2;
            if(i+optionLen<=argc){
                inFolder = argv[i+1];
            }else{
                printf("Error: Missing arguments for -i\n");
                return 1;
            }
        }else if(strcmp(argv[i],"-o")==0){
            optionLen = 2;
            if(i+optionLen<=argc){
                outFolder = argv[i+1];
            }else{
                printf("Error: Missing arguments for -o\n");
                return 1;
            }
        }else if(strcmp(argv[i],"-t")==0){
            optionLen = 2;
            if(i+optionLen<=argc){
                timeSteps = strtol(argv[i+1],NULL,10);
            }else{
                printf("Error: Missing arguments for -t\n");
                return 1;
            }
        }else if(strcmp(argv[i],"-g")==0){
            optionLen = 4;
            if(i+optionLen<=argc){
                gridWidth = strtol(argv[i+1],NULL,10);
                gridHeight = strtol(argv[i+2],NULL,10);
                gridDepth = strtol(argv[i+3],NULL,10);
            }else{
                printf("Error: Missing arguments for -g\n");
                return 1;
            }
        }else if(strcmp(argv[i],"-b")==0){
            optionLen = 4;
            if(i+optionLen<=argc){
                blockWidth = strtol(argv[i+1],NULL,10);
                blockHeight = strtol(argv[i+2],NULL,10);
                blockDepth = strtol(argv[i+3],NULL,10);
            }else{
                printf("Error: Missing arguments for -b\n");
                return 1;
            }
        }else{
            printf("Error: Parameters must be of form:\n");
            printf("./game [-i infile] [-o outfile] [-t timesteps] [-g gridsize] [-b blockdimensions]\n");
            return 1;
        }
    }

    

    //print for debugging purposes
    std::cout << "In folder = " << inFolder << "\n";
    std::cout << "Out folder = " << outFolder << "\n";
    printf("Time steps = %d\n",timeSteps);
    printf("Block dimensions = %dx%dx%d\n",blockWidth,blockHeight,blockDepth);



    //read binary header
    char *inFile = (char *)calloc(inFolder.length()+strlen(SIM_STATE_NAME)+1, sizeof(char));
    strcpy(inFile,inFolder.c_str());
    strcat(inFile,SIM_STATE_NAME);

    FILE *inGridFile = fopen(inFile,"rb");
    if(inGridFile!=NULL){
        readHeaderBinary(inGridFile,&gridWidth,&gridHeight,&gridDepth);

        //print for debugging purposes
        printf("Using grid settings from file...\n");
    }else{
        //print for debugging purposes
        printf("No file found, using default grid settings...\n");
    }

    //derived values
    gridWidthBlocks = std::ceil((float)gridWidth/(float)blockWidth);
    gridHeightBlocks = std::ceil((float)gridHeight/(float)blockHeight);
    gridDepthBlocks = std::ceil((float)gridDepth/(float)blockDepth);
    gridArea = gridWidth*gridHeight*gridDepth;



    //print for debugging purposes
    printf("Grid dimensions = %dx%dx%d\n",gridWidth,gridHeight,gridDepth);
    printf("Grid in blocks = %dx%dx%d\n",gridWidthBlocks,gridHeightBlocks,gridDepthBlocks);

    

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

    dim3 numBlocks(gridWidthBlocks,gridHeightBlocks,gridDepthBlocks);
    dim3 blockSize(blockWidth,blockHeight,blockDepth);

    size_t gridSize = gridWidth*gridHeight*gridDepth;

    //device+host grid arrays
    double *grid_h,*grid1_h;
    double *grid_d,*grid1_d;

    //allocate host memory
    grid_h = (double *)calloc(gridSize,sizeof(double));
    grid1_h = (double *)calloc(gridSize,sizeof(double));

    //grid1_h is initialized with all zeros but in the future it may need to be set
    //(the n-2th time value it stores is used in calculation)

    //load grid_h from file
    if(inGridFile!=NULL){
        readDoublesBinary(inGridFile,grid_h,gridArea);
        fclose(inGridFile);

        //print for debugging purposes
        printf("Read grid from file...\n");
    }else{
        //print for debugging purposes
        printf("No file found, using an empty grid...\n");
    }

    //allocate device memory
    cudaMalloc((void **)&grid_d, gridSize);
    cudaMalloc((void **)&grid1_d, gridSize);
    
    //copy both grids to device
    cudaMemcpy(grid_d,grid_h,gridSize,cudaMemcpyHostToDevice);
    cudaMemcpy(grid1_d,grid1_h,gridSize,cudaMemcpyHostToDevice);

    for(int i=0;i<timeSteps;i++){
        solver<<<numBlocks,blockSize>>>(grid_d,grid1_d);
        cudaDeviceSynchronize();
        std::swap(grid_d,grid1_d);

        checkCudaError();
    }

    //only copy first grid to host, since it was computed and then swapped by kernel
    cudaMemcpy(grid_h,grid_d,gridSize,cudaMemcpyDeviceToHost);



    //output grid in text form for debugging
    writeTextRepr(outFolder+"/text_repr.txt",grid_h);

    
    
    //write output binary file
    char *outFile = (char *)calloc(outFolder.length()+strlen(SIM_STATE_NAME)+1, sizeof(char));
    strcpy(outFile,outFolder.c_str());
    strcat(outFile,SIM_STATE_NAME);

    FILE *outGridFile = fopen(outFile,"wb");
    if(outGridFile!=NULL){
        writeHeaderBinary(outGridFile,&gridWidth,&gridHeight,&gridDepth);
        writeDoublesBinary(outGridFile,grid_h,gridArea);
        fclose(outGridFile);
    }

    //free host memory
    free(grid_h);
    free(grid1_h);
    //free device memory
    cudaFree(grid_d);
    cudaFree(grid1_d);

    //end clock
    auto endTime = std::chrono::high_resolution_clock::now();
    auto timePassed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    printf("Ran in %ld ms\n",timePassed);

    return 0;
}