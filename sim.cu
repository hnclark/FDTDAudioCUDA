/* 
    -Uses libsndfile, covered by the GNU LGPL
*/
#include<stdlib.h> //malloc/realloc
#include<stdio.h> //printf
#include<sndfile.h> //handling sound files
#include<sys/stat.h> //mkdir
#include<string> //strings
#include<chrono> //timing

#define SIM_STATE_NAME "sim_state.bin"
#define AUDIO_LEDGER_NAME "audio_ledger.txt"
#define AUDIO_OUT_LEDGER_NAME "audio_out_ledger.txt"

#define AUDIO_DEFAULT_EXTENSION ".wav"
#define AUDIO_DEFAULT_FORMAT SF_FORMAT_WAV | SF_FORMAT_DOUBLE | SF_ENDIAN_CPU

#define NO_FOLDER "NONE"
#define FOLDER_DEFAULT_PERMISSIONS S_IRWXU | S_IRWXG | S_IRWXO



//audio file struct
typedef struct{
    SNDFILE *file;
    SF_INFO info;
}audioFile;

//coordinate struct
typedef struct{
    int x;
    int y;
    int z;
}coord;



//shared host/device constants
int gridWidth,gridHeight,gridDepth,blockWidth,blockHeight,blockDepth,gridWidthBlocks,gridHeightBlocks,gridDepthBlocks,timeSteps,audioSourceCount,audioOutputCount;
__constant__ int gridWidth_d,gridHeight_d,gridDepth_d,blockWidth_d,blockHeight_d,blockDepth_d,gridWidthBlocks_d,gridHeightBlocks_d,gridDepthBlocks_d,timeSteps_d,audioSourceCount_d,audioOutputCount_d;

//host only constants
int gridArea;



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

__global__ void loadAudioSources(double *grid,coord *pos,double *source,int frame){
    int index = blockIdx.x;
    int frameIndex = (index*timeSteps_d)+frame;

    grid[arrayPos(pos[index].x,pos[index].y,pos[index].z)] = source[frameIndex];
}

__global__ void readAudioOutputs(double *grid,coord *pos,double *output,int frame){
    int index = blockIdx.x;
    int frameIndex = (index*timeSteps_d)+frame;

    output[frameIndex] = grid[arrayPos(pos[index].x,pos[index].y,pos[index].z)];
}



//check for and print cuda errors
void checkCudaError(){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("%s\n",cudaGetErrorString(error));
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

//helper function to read a line from an audio ledger file
char *readAudioLedgerLine(FILE *fileIn,int *x,int *y,int *z){
    char *audioFileName;
    char *line;
    size_t lineBuffer = 0;
    
    if(getline(&line,&lineBuffer,fileIn)>=7){
        char *wordPtr;
        audioFileName = strtok_r(line," ",&wordPtr);
        *x = strtol(strtok_r(NULL," ",&wordPtr),NULL,10);
        *y = strtol(strtok_r(NULL," ",&wordPtr),NULL,10);
        *z = strtol(strtok_r(NULL," ",&wordPtr),NULL,10);
        
        return audioFileName;
    }else{
        return NULL;
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
            printf("./sim [-i infolder] [-o outfolder] [-t timesteps] [-g gridsize] [-b blockdimensions]\n");
            return 1;
        }
    }

    

    //print for debugging purposes
    printf("Basic settings...\n");
    printf("    In folder = %s\n",inFolder.c_str());
    printf("    Out folder = %s\n",outFolder.c_str());
    printf("    Block dimensions = %dx%dx%d\n",blockWidth,blockHeight,blockDepth);
    printf("    Time steps = %d\n",timeSteps);



    //create output folder
    mkdir(outFolder.c_str(),FOLDER_DEFAULT_PERMISSIONS);



    //audio positions and files
    audioSourceCount = 0;
    coord *audioSourcePos_h = NULL;
    audioFile *audioSourceFiles = NULL;

    //load audio ledger file
    FILE *inAudioLedgerFile = NULL;
    if(inFolder!=NO_FOLDER){
        std::string inAudioLedgerName = inFolder+"/"+AUDIO_LEDGER_NAME;
        inAudioLedgerFile = fopen(inAudioLedgerName.c_str(),"r");
    }

    //read audio ledger file
    if(inAudioLedgerFile!=NULL){
        //print for debugging purposes
        printf("Using audio ledger file...\n");
        
        int x,y,z,index;
        char *audioFileName;

        //load the next line from the audio ledger and continue if it's not NULL
        while((audioFileName = readAudioLedgerLine(inAudioLedgerFile,&x,&y,&z))!=NULL){
            index = audioSourceCount;
            audioSourceCount++;

            //load file
            audioSourceFiles = (audioFile *)realloc(audioSourceFiles,audioSourceCount*sizeof(audioFile));

            std::string audioFilePath = inFolder+"/"+audioFileName;
            audioSourceFiles[index].file = sf_open(audioFilePath.c_str(),SFM_READ,&audioSourceFiles[index].info);

            //load pos
            audioSourcePos_h = (coord *)realloc(audioSourcePos_h,audioSourceCount*sizeof(coord));
            audioSourcePos_h[index].x = x;
            audioSourcePos_h[index].y = y;
            audioSourcePos_h[index].z = z;

            //print for debugging purposes
            printf("    Loading audio source: %s... ",audioFilePath.c_str());

            if(sf_error(audioSourceFiles[index].file)==SF_ERR_NO_ERROR){
                if(audioSourceFiles[index].info.channels!=1){
                    free(audioSourceFiles[index].file);
                    audioSourceFiles[index].file = NULL;

                    //print for debugging purposes
                    printf(">1 channels, skipped file\n");
                }else{
                    //print for debugging purposes
                    printf("Loaded file\n");
                }
            }else{
                //print for debugging purposes
                printf("Could not load file\n");
            }

            free(audioFileName);
        }

        fclose(inAudioLedgerFile);
    }else{
        //print for debugging purposes
        printf("No audio ledger file found...\n");
    }



    //default audio output settings
    int audioOutSamplerate = 44100; //TODO: this should be set to the simulation samplerate

    //audio positions and files
    audioOutputCount = 0;
    coord *audioOutputPos_h = NULL;
    audioFile *audioOutputFiles = NULL;

    //load audio ledger file
    FILE *outAudioLedgerFile = NULL;
    if(outFolder!=NO_FOLDER){
        std::string outAudioLedgerName = inFolder+"/"+AUDIO_OUT_LEDGER_NAME;
        outAudioLedgerFile = fopen(outAudioLedgerName.c_str(),"r");
    }

    //read audio ledger file
    if(outAudioLedgerFile!=NULL){
        //print for debugging purposes
        printf("Using audio out ledger file...\n");
                
        int x,y,z,index;
        char *audioFileName;

        //load the next line from the audio ledger and continue if it's not NULL
        while((audioFileName = readAudioLedgerLine(outAudioLedgerFile,&x,&y,&z))!=NULL){
            index = audioOutputCount;
            audioOutputCount++;

            //create file
            audioOutputFiles = (audioFile *)realloc(audioOutputFiles,audioOutputCount*sizeof(audioFile));

            std::string audioFilePath = outFolder+"/"+audioFileName+AUDIO_DEFAULT_EXTENSION;
            audioOutputFiles[index].info.samplerate = audioOutSamplerate;
            audioOutputFiles[index].info.channels = 1;
            audioOutputFiles[index].info.format = AUDIO_DEFAULT_FORMAT;
            audioOutputFiles[index].file = sf_open(audioFilePath.c_str(),SFM_WRITE,&audioOutputFiles[index].info);

            //load pos
            audioOutputPos_h = (coord *)realloc(audioOutputPos_h,audioOutputCount*sizeof(coord));
            audioOutputPos_h[index].x = x;
            audioOutputPos_h[index].y = y;
            audioOutputPos_h[index].z = z;

            //print for debugging purposes
            printf("    Creating audio output: %s... ",audioFilePath.c_str());

            if(sf_error(audioOutputFiles[index].file)==SF_ERR_NO_ERROR){
                //print for debugging purposes
                printf("Created file\n");
            }else{
                //print for debugging purposes
                printf("Could not create file\n");
            }

            free(audioFileName);
        }

        fclose(outAudioLedgerFile);
    }else{
        //print for debugging purposes
        printf("No audio out ledger file found...\n");
    }



    size_t audioSourceSize = audioSourceCount*timeSteps*sizeof(double);
    size_t audioOutputSize = audioOutputCount*timeSteps*sizeof(double);

    size_t audioSourcePosSize = audioSourceCount*sizeof(coord);
    size_t audioOutputPosSize = audioOutputCount*sizeof(coord);

    //device+host memory pointers
    double *audioSource_h,*audioOutput_h;
    double *audioSource_d,*audioOutput_d;

    coord *audioSourcePos_d,*audioOutputPos_d;

    //allocate host memory
    audioSource_h = (double *)calloc(audioSourceSize,1);
    audioOutput_h = (double *)calloc(audioOutputSize,1);

    //load audio sources
    for(int j=0;j<audioSourceCount;j++){
        sf_read_double(audioSourceFiles[j].file,audioSource_h+(timeSteps*j),timeSteps);
        sf_close(audioSourceFiles[j].file);
    }

    free(audioSourceFiles);

    //allocate device memory
    cudaMalloc((void **)&audioSource_d, audioSourceSize);
    cudaMalloc((void **)&audioOutput_d, audioOutputSize);

    cudaMalloc((void **)&audioSourcePos_d, audioSourcePosSize);
    cudaMalloc((void **)&audioOutputPos_d, audioOutputPosSize);

    //copy host memory to device
    cudaMemcpy(audioSource_d,audioSource_h,audioSourceSize,cudaMemcpyHostToDevice);
    cudaMemcpy(audioOutput_d,audioOutput_h,audioOutputSize,cudaMemcpyHostToDevice);

    cudaMemcpy(audioSourcePos_d,audioSourcePos_h,audioSourcePosSize,cudaMemcpyHostToDevice);
    cudaMemcpy(audioOutputPos_d,audioOutputPos_h,audioOutputPosSize,cudaMemcpyHostToDevice);



    //default grid configuration
    gridWidth = 16;
    gridHeight = 16;
    gridDepth = 16;

    //read binary header
    FILE *inGridFile = NULL;
    if(inFolder!=NO_FOLDER){
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

    

    dim3 numBlocks(gridWidthBlocks,gridHeightBlocks,gridDepthBlocks);
    dim3 blockSize(blockWidth,blockHeight,blockDepth);

    size_t gridSize = gridWidth*gridHeight*gridDepth*sizeof(double);

    //device+host memory pointers
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

    //allocate device memory
    cudaMalloc((void **)&grid_d, gridSize);
    cudaMalloc((void **)&grid1_d, gridSize);

    //copy host memory to device
    cudaMemcpy(grid_d,grid_h,gridSize,cudaMemcpyHostToDevice);
    cudaMemcpy(grid1_d,grid1_h,gridSize,cudaMemcpyHostToDevice);



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
    cudaMemcpyToSymbol(*(&timeSteps_d),&timeSteps,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&audioSourceCount_d),&audioSourceCount,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(*(&audioOutputCount_d),&audioOutputCount,sizeof(int),0,cudaMemcpyHostToDevice);



    //run the main loop
    for(int i=0;i<timeSteps;i++){
        //load in audio sources
        loadAudioSources<<<audioSourceCount,1>>>(grid_d,audioSourcePos_d,audioSource_d,i);
        cudaDeviceSynchronize();

        //solve FDTD
        solver<<<numBlocks,blockSize>>>(grid_d,grid1_d);
        cudaDeviceSynchronize();
        std::swap(grid_d,grid1_d);

        //read out audio sources
        readAudioOutputs<<<audioOutputCount,1>>>(grid1_d,audioOutputPos_d,audioOutput_d,i);
        cudaDeviceSynchronize();

        checkCudaError();
    }



    //copy first grid to host, since it was computed and then swapped by kernel
    cudaMemcpy(grid_h,grid_d,gridSize,cudaMemcpyDeviceToHost);
    
    //create grid output file
    FILE *outGridFile = NULL;
    if(outFolder!=NO_FOLDER){
        //create binary file
        std::string outGridFileName = outFolder+"/"+SIM_STATE_NAME;
        outGridFile = fopen(outGridFileName.c_str(),"wb");
    }

    //write output file
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



    //copy audio outputs to host
    cudaMemcpy(audioOutput_h,audioOutput_d,audioOutputSize,cudaMemcpyDeviceToHost);

    //close audio output file data
    for(int j=0;j<audioOutputCount;j++){
        sf_write_double(audioOutputFiles[j].file,audioOutput_h+(timeSteps*j),timeSteps);
        sf_write_sync(audioOutputFiles[j].file);
        sf_close(audioOutputFiles[j].file);
    }
    free(audioOutputFiles);



    //free host memory
    free(grid_h);
    free(grid1_h);
    free(audioSourcePos_h);
    free(audioOutputPos_h);
    free(audioSource_h);
    free(audioOutput_h);

    //free device memory
    cudaFree(grid_d);
    cudaFree(grid1_d);
    cudaFree(audioSourcePos_d);
    cudaFree(audioOutputPos_d);
    cudaFree(audioSource_d);
    cudaFree(audioOutput_d);



    //end clock
    auto endTime = std::chrono::high_resolution_clock::now();
    auto timePassed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    printf("Ran in %ld ms\n",timePassed);

    return 0;
}