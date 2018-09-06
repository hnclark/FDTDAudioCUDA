/* 
    -Uses libsndfile, covered by the GNU LGPL
*/
#include<sndfile.h> //handling sound files
#include<stdlib.h> //math functions
#include<iostream> //strings+cout
#include<sys/stat.h> //mkdir
#include<chrono> //timing

#define SIM_STATE_NAME "sim_state.bin"
#define NO_FOLDER "NONE"
#define FOLDER_DEFAULT_PERMISSIONS S_IRWXU | S_IRWXG | S_IRWXO


typedef struct{
    std::string name;
    SNDFILE *file;
    SF_INFO info;
}audioFile;

typedef struct{
    int x;
    int y;
    int z;
}coord;



//shared host/device constants
int gridWidth,gridHeight,gridDepth,blockWidth,blockHeight,blockDepth,gridWidthBlocks,gridHeightBlocks,gridDepthBlocks,audioSourceCount;
__constant__ int gridWidth_d,gridHeight_d,gridDepth_d,blockWidth_d,blockHeight_d,blockDepth_d,gridWidthBlocks_d,gridHeightBlocks_d,gridDepthBlocks_d,audioSourceCount_d;

//host only constants
int timeSteps,gridArea;



__device__ int arrayPos(const int &x,const int &y,const int &z){
    return (z*gridWidth_d*gridHeight_d)+(y*gridWidth_d)+x;
}



__global__ void solver(double *grid,double *grid1){
    int xpos = (blockIdx.x*blockWidth_d)+threadIdx.x;
    int ypos = (blockIdx.y*blockHeight_d)+threadIdx.y;
    int zpos = (blockIdx.z*blockDepth_d)+threadIdx.z;

    if(xpos>0 && xpos<gridWidth_d-1 && ypos>0 && ypos<gridHeight_d-1 && zpos>0 && zpos<gridDepth_d-1){
        //the grid1 value at each point contains the value from 2 steps ago(but is overwritten once used)
        grid1[arrayPos(xpos,ypos,zpos)] = ((grid[arrayPos(xpos+1,ypos,zpos)]+grid[arrayPos(xpos-1,ypos,zpos)]
                                        +grid[arrayPos(xpos,ypos+1,zpos)]+grid[arrayPos(xpos,ypos-1,zpos)]
                                        +grid[arrayPos(xpos,ypos,zpos+1)]+grid[arrayPos(xpos,ypos,zpos-1)]
                                        -grid1[arrayPos(xpos,ypos,zpos)])/6);
    }
}



__global__ void loadAudioSource(double *grid,coord pos,double val){
    grid[arrayPos(pos.x,pos.y,pos.z)] = val;
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



    //audio positions and files
    coord *audioInPos;
    audioFile *audioFiles;
    //default audio source settings
    audioSourceCount = 0;



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
        }else if(strcmp(argv[i],"-s")==0){
            optionLen = 2;
            if(i+optionLen<=argc){
                audioSourceCount = strtol(argv[i+1],NULL,10);
                audioInPos = (coord *)malloc(audioSourceCount*sizeof(coord));
                audioFiles = (audioFile *)malloc(audioSourceCount*sizeof(audioFile));
                
                int argsPerAudioFile = 4;

                for(int j=0;j<audioSourceCount;j++){
                    optionLen += argsPerAudioFile;
                    if(i+optionLen<=argc){
                        //set audio source file name
                        audioFiles[j].name = argv[i+(j*argsPerAudioFile)+1+1];
                        //set audio source coordinates
                        audioInPos[j].x = strtol(argv[i+(j*argsPerAudioFile)+1+2],NULL,10);
                        audioInPos[j].y = strtol(argv[i+(j*argsPerAudioFile)+1+3],NULL,10);
                        audioInPos[j].z = strtol(argv[i+(j*argsPerAudioFile)+1+4],NULL,10);
                    }else{
                        printf("Error: Missing arguments for -s sublist\n");
                        return 1;
                    }
                }

            }else{
                printf("Error: Missing arguments for -s\n");
                return 1;
            }
        }else{
            printf("Error: Parameters must be of form:\n");
            printf("./sim [-i infile] [-o outfile] [-t timesteps] [-g gridsize] [-b blockdimensions] [-s sourcecount [[sourcefile sourcepos]...]]\n");
            return 1;
        }
    }

    

    //print for debugging purposes
    std::cout << "In folder = " << inFolder << "\n";
    std::cout << "Out folder = " << outFolder << "\n";
    printf("Time steps = %d\n",timeSteps);
    printf("Block dimensions = %dx%dx%d\n",blockWidth,blockHeight,blockDepth);



    //load audio source files
    if(inFolder!=NO_FOLDER){
        for(int i=0;i<audioSourceCount;i++){
            std::string inAudFilePath = inFolder+"/"+audioFiles[i].name;

            //print for debugging purposes
            printf("Loading audio source: %s...\n",inAudFilePath.c_str());

            audioFiles[i].file = sf_open(inAudFilePath.c_str(),SFM_READ,&audioFiles[i].info);

            if(sf_error(audioFiles[i].file)==SF_ERR_NO_ERROR){
                //print for debugging purposes
                printf("    Loaded %ld frames\n",audioFiles[i].info.frames);
            }else{
                //print for debugging purposes
                printf("    Error: %s\n",sf_strerror(audioFiles[i].file));
            }
        }
    }

    //read binary header
    FILE *inGridFile = NULL;
    if(inFolder!=NO_FOLDER){
        //load binary file
        std::string inGridFileName = inFolder+"/"+SIM_STATE_NAME;
        inGridFile = fopen(inGridFileName.c_str(),"rb");
    }

    //read binary file
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
    printf("    Grid dimensions = %dx%dx%d\n",gridWidth,gridHeight,gridDepth);
    printf("    Grid in blocks = %dx%dx%d\n",gridWidthBlocks,gridHeightBlocks,gridDepthBlocks);

    

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
    cudaMemcpyToSymbol(*(&audioSourceCount_d),&audioSourceCount,sizeof(int),0,cudaMemcpyHostToDevice);

    dim3 numBlocks(gridWidthBlocks,gridHeightBlocks,gridDepthBlocks);
    dim3 blockSize(blockWidth,blockHeight,blockDepth);

    size_t gridSize = gridWidth*gridHeight*gridDepth*sizeof(double);

    //device+host memory
    double *grid_h,*grid1_h;
    double *grid_d,*grid1_d;

    //allocate host memory
    grid_h = (double *)calloc(gridSize,1);
    grid1_h = (double *)calloc(gridSize,1);

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

    //grid1_h is initialized with all zeros but in the future it may need to be set
    //(the n-2th time value it stores is used in calculation)

    //allocate device memory
    cudaMalloc((void **)&grid_d, gridSize);
    cudaMalloc((void **)&grid1_d, gridSize);

    //copy host memory to device
    cudaMemcpy(grid_d,grid_h,gridSize,cudaMemcpyHostToDevice);
    cudaMemcpy(grid1_d,grid1_h,gridSize,cudaMemcpyHostToDevice);

    for(int i=0;i<timeSteps;i++){
        //load in audio sources
        for(int j=0;j<audioSourceCount;j++){
            //load audio value from each source
            double val = ((2*(i%2))-1);
            //
            //TODO: AUDIO VALUE SHOULD BE LOADED TO val HERE
            //

            loadAudioSource<<<1,1>>>(grid_d,audioInPos[j],val);
            cudaDeviceSynchronize();
        }

        //solve FDTD
        solver<<<numBlocks,blockSize>>>(grid_d,grid1_d);
        cudaDeviceSynchronize();
        std::swap(grid_d,grid1_d);

        checkCudaError();
    }

    //only copy first grid to host, since it was computed and then swapped by kernel
    cudaMemcpy(grid_h,grid_d,gridSize,cudaMemcpyDeviceToHost);    
    
    //write output
    FILE *outGridFile = NULL;
    if(outFolder!=NO_FOLDER){
        mkdir(outFolder.c_str(),FOLDER_DEFAULT_PERMISSIONS);

        //create binary file
        std::string outGridFileName = outFolder+"/"+SIM_STATE_NAME;
        outGridFile = fopen(outGridFileName.c_str(),"wb");
    }

    //write binary file
    if(outGridFile!=NULL){
        writeHeaderBinary(outGridFile,&gridWidth,&gridHeight,&gridDepth);
        writeDoublesBinary(outGridFile,grid_h,gridArea);
        fclose(outGridFile);

        //print for debugging purposes
        printf("Outputting to file...\n");
    }else{
        //print for debugging purposes
        printf("No output...\n");
    }

    //free host memory
    free(grid_h);
    free(grid1_h);

    //free device memory
    cudaFree(grid_d);
    cudaFree(grid1_d);

    //free audio position data
    free(audioInPos);

    //free audio file data
    for(int j=0;j<audioSourceCount;j++){
        sf_close(audioFiles[j].file);
    }
    free(audioFiles);

    //end clock
    auto endTime = std::chrono::high_resolution_clock::now();
    auto timePassed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    printf("Ran in %ld ms\n",timePassed);

    return 0;
}