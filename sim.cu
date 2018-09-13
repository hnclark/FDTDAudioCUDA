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


typedef struct{
    std::string name;
    SNDFILE *file;
    SF_INFO info;
    bool loaded;
}audioFile;

typedef struct{
    int x;
    int y;
    int z;
}coord;



//shared host/device constants
int gridWidth,gridHeight,gridDepth,blockWidth,blockHeight,blockDepth,gridWidthBlocks,gridHeightBlocks,gridDepthBlocks,audioSourceCount,audioOutputCount;
__constant__ int gridWidth_d,gridHeight_d,gridDepth_d,blockWidth_d,blockHeight_d,blockDepth_d,gridWidthBlocks_d,gridHeightBlocks_d,gridDepthBlocks_d,audioSourceCount_d,audioOutputCount_d;

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



__device__ double audioVal;
__global__ void readAudioSource(double *grid,coord pos){
    audioVal = grid[arrayPos(pos.x,pos.y,pos.z)];
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



//check for and print cuda errors
void checkCudaError(){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("%s\n",cudaGetErrorString(error));
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
    printf("In folder = %s\n",inFolder.c_str());
    printf("Out folder = %s\n",outFolder.c_str());
    printf("Time steps = %d\n",timeSteps);
    printf("Block dimensions = %dx%dx%d\n",blockWidth,blockHeight,blockDepth);


    //create output folder
    mkdir(outFolder.c_str(),FOLDER_DEFAULT_PERMISSIONS);
    
    //default grid configuration
    gridWidth = 16;
    gridHeight = 16;
    gridDepth = 16;

    //default audio source settings
    audioSourceCount = 0;

    //audio positions and files
    coord *audioInPos = NULL;
    audioFile *audioFiles = NULL;

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

            //allocate memory for new audioFile and position
            audioInPos = (coord *)realloc(audioInPos,audioSourceCount*sizeof(coord));
            audioFiles = (audioFile *)realloc(audioFiles,audioSourceCount*sizeof(audioFile));

            std::string inAudFilePath = inFolder+"/"+audioFileName;

            //print for debugging purposes
            printf("    Loading audio source: %s... ",inAudFilePath.c_str());

            audioFiles[index].file = sf_open(inAudFilePath.c_str(),SFM_READ,&audioFiles[index].info);

            if(sf_error(audioFiles[index].file)==SF_ERR_NO_ERROR && audioFiles[index].info.channels==1){
                //print for debugging purposes
                printf("Loaded %ld frames\n",audioFiles[index].info.frames);

                audioFiles[index].loaded = true;
            }else{
                //print for debugging purposes
                printf("Could not load file\n");

                audioFiles[index].loaded = false;
            }

            audioInPos[index].x = x;
            audioInPos[index].y = y;
            audioInPos[index].z = z;
        }

        fclose(inAudioLedgerFile);
    }else{
        //print for debugging purposes
        printf("No audio ledger file found...\n");
    }



    //default audio source settings
    int audioOutSamplerate = 44100; //TODO: this should be set to the simulation samplerate

    audioOutputCount = 0;

    //audio positions and files
    coord *audioOutPos = NULL;
    audioFile *audioOutFiles = NULL;

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

            //allocate memory for new audioFile and position
            audioOutPos = (coord *)realloc(audioOutPos,audioOutputCount*sizeof(coord));
            audioOutFiles = (audioFile *)realloc(audioOutFiles,audioOutputCount*sizeof(audioFile));

            std::string outAudFilePath = outFolder+"/"+audioFileName+AUDIO_DEFAULT_EXTENSION;

            //print for debugging purposes
            printf("    Creating audio output: %s... ",outAudFilePath.c_str());

            //set sfinfo
            audioOutFiles[index].info.samplerate = audioOutSamplerate;
            audioOutFiles[index].info.channels = 1;
            audioOutFiles[index].info.format = AUDIO_DEFAULT_FORMAT;

            audioOutFiles[index].file = sf_open(outAudFilePath.c_str(),SFM_WRITE,&audioFiles[index].info);

            if(sf_error(audioOutFiles[index].file)==SF_ERR_NO_ERROR){
                //print for debugging purposes
                printf("Created file\n");

                audioOutFiles[index].loaded = true;
            }else{
                //print for debugging purposes
                printf("Could not create file\n");

                audioOutFiles[index].loaded = false;
            }

            audioOutPos[index].x = x;
            audioOutPos[index].y = y;
            audioOutPos[index].z = z;
        }

        fclose(outAudioLedgerFile);
    }else{
        //print for debugging purposes
        printf("No audio out ledger file found...\n");
    }



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
    cudaMemcpyToSymbol(*(&audioOutputCount_d),&audioOutputCount,sizeof(int),0,cudaMemcpyHostToDevice);

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
            double val = 0;
            if(audioFiles[j].loaded && sf_read_double(audioFiles[j].file,&val,1)!=0){
                loadAudioSource<<<1,1>>>(grid_d,audioInPos[j],val);
                cudaDeviceSynchronize();
            }
        }

        //solve FDTD
        solver<<<numBlocks,blockSize>>>(grid_d,grid1_d);
        cudaDeviceSynchronize();
        std::swap(grid_d,grid1_d);

        checkCudaError();

        //read out audio sources
        for(int j=0;j<audioOutputCount;j++){
            //read audio value from each point
            double val = 0;
            if(audioOutFiles[j].loaded){
                readAudioSource<<<1,1>>>(grid1_d,audioOutPos[j]); //TODO:make sure this is actually reading values to val
                cudaMemcpyFromSymbol(&val,audioVal,sizeof(double),0,cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                sf_write_double(audioOutFiles[j].file,&val,1);
            }
        }
    }

    //only copy first grid to host, since it was computed and then swapped by kernel
    cudaMemcpy(grid_h,grid_d,gridSize,cudaMemcpyDeviceToHost);    
    
    //write output
    FILE *outGridFile = NULL;
    if(outFolder!=NO_FOLDER){
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

    //close audio input file data
    for(int j=0;j<audioSourceCount;j++){
        sf_close(audioFiles[j].file);
    }
    free(audioFiles);

    //close audio output file data
    for(int j=0;j<audioOutputCount;j++){
        sf_write_sync(audioOutFiles[j].file);
        sf_close(audioOutFiles[j].file);
    }
    free(audioOutFiles);

    //end clock
    auto endTime = std::chrono::high_resolution_clock::now();
    auto timePassed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    printf("Ran in %ld ms\n",timePassed);

    return 0;
}