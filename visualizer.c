#include<gtk/gtk.h> //displaying graphics
#include<stdio.h> //file loading
#include<math.h> //math functions

#define SIM_STATE_NAME "sim_state.bin"
#define AUDIO_LEDGER_NAME "audio_ledger.txt"
#define AUDIO_OUT_LEDGER_NAME "audio_out_ledger.txt"

#define SAMPLES_PER_PIXEL 3
#define BITS_PER_SAMPLE 8

#define DEFAULT_PADDING 3

#define BASE_COMMAND "./sim"

#define FLAG_FRONT " -"
#define FLAG_BACK " "

#define INPUT_FLAG "i"
#define OUTPUT_FLAG "o"
#define TIMESTEP_FLAG "t"
#define GRIDSIZE_FLAG "g"
#define BLOCKSIZE_FLAG "b"


//data for audio list store
enum{
    COLUMN_ICON,
    COLUMN_SOURCE,
    COLUMN_X,
    COLUMN_Y,
    COLUMN_Z,
    COLUMN_NAME,
    N_COLUMNS
};


//pixel struct
typedef struct{
    guchar red;
    guchar green;
    guchar blue;
}pixel;



//Main window
GtkWidget* window;

//Menu bar
GtkWidget* menuBar;
GtkWidget* fileMenuItem;
GtkWidget* fileMenu;
GtkWidget* newItem;
GtkWidget* openItem;
GtkWidget* saveItem;
GtkWidget* saveAndRunItem;
GtkWidget* quitItem;

//Display image drawing
GtkWidget* imageWindow;
GtkAllocation* displayImageAllocation;
GtkWidget* displayImage;
double imageRatio;

//Grid dimensions
int gridWidth = 1;
int gridHeight = 1;
int gridDepth = 1;

int gridArea;
size_t gridSize;

//grid images
double *grid;
guchar *gridImageData;

GdkPixbuf **gridPixbufs;

//grid color gain factor
double colorGainFactor = (double)1/3;

//Cursor positioning/drawing
int cursorX = 0;
int cursorY = 0;
int cursorZ = 0;

GtkWidget* cursorButtonX;
GtkWidget* cursorButtonY;
GtkWidget* cursorButtonZ;

GdkPixbuf* cursorPixbuf;

//New Grid dialog options
int newGridWidth = 1;
int newGridHeight = 1;
int newGridDepth = 1;

GtkWidget* gridSizeX;
GtkWidget* gridSizeY;
GtkWidget* gridSizeZ;

//Simulation setting dialog options
GtkWidget* simSettingFolderButton;
GtkWidget* simSettingFolderText;
GtkWidget* simSettingTimesteps;
GtkWidget* simSettingBlockWidth;
GtkWidget* simSettingBlockHeight;
GtkWidget* simSettingBlockDepth;

//list store stuff
GtkListStore* audioListStore;
GtkWidget* audioListView;

GtkWidget* audioListNewSourceItem;
GtkWidget* audioListNewOutputItem;
GtkWidget* audioListRemoveItem;

//indicates whether a file is currently completely loaded into memory. Don't redraw stuff if one isn't
gboolean fileOpen;

//name of the folder currently loaded. NULL if a new grid is loaded or no folder is loaded.
char *currentInFolder = NULL;

//name of the folder the simulator will output to
char *currentOutFolder = "output";

//timesteps of simulation
int timesteps = 1;

//block size to be used in simulation
int blockWidth = 16;
int blockHeight = 16;
int blockDepth = 1;

//whether the file has been saved or not
gboolean fileSaved;


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
char *readAudioLedgerLine(FILE *file,int *x,int *y,int *z){
    char *audioFileName;
    char *line;
    size_t lineBuffer = 0;
    
    if(getline(&line,&lineBuffer,file)>=7){
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

//helper function to write a line to an audio ledger file
void writeAudioLedgerLine(FILE *file,const char *name,int x,int y,int z){
    fprintf(file,"%s %d %d %d\n",name,x,y,z);
}



//helper function to append command line flags to commands
char *appendCommandLineFlag(char *command,char *flag,char *val){
    command = (char *)realloc(command,(strlen(command)+strlen(FLAG_FRONT)+strlen(flag)+strlen(FLAG_BACK)+strlen(val))*sizeof(char));
    strcat(command,FLAG_FRONT);
    strcat(command,flag);
    strcat(command,FLAG_BACK);
    strcat(command,val);

    return command;
}



pixel doubleToPixel(double val){
    pixel pix;
    pix.green = 0;
    if(val>0){
        pix.red = (guchar)abs(pow(val,colorGainFactor)*255);
        pix.blue = 0;
    }else{
        pix.red = 0;
        pix.blue = (guchar)abs(pow(val,colorGainFactor)*255);
    }

    return pix;
}

//helper function to convert an array of doubles to an array of guchars
void doublesToGuchar(double *arrayIn,guchar *arrayOut,int arrayLen){
    for(int i=0;i<arrayLen;i++){
        pixel pix = doubleToPixel(arrayIn[i]);
        arrayOut[i*3]=pix.red;
        arrayOut[i*3+1]=pix.green;
        arrayOut[i*3+2]=pix.blue;
    }
}

//helper function to convert an array of guchars to an array of pixbufs, one per image layer
void gucharToPixbufs(guchar *arrayIn,GdkPixbuf *pixbufs[],int imageWidth,int imageHeight,int imageCount){
    for(int i=0;i<imageCount;i++){
        guchar *imagePointer = arrayIn+(i*imageWidth*imageHeight*SAMPLES_PER_PIXEL);
        pixbufs[i] = gdk_pixbuf_new_from_data(imagePointer,GDK_COLORSPACE_RGB,FALSE,BITS_PER_SAMPLE,imageWidth,imageHeight,imageWidth*3,NULL,NULL);
    }
}



void updateDisplayImage(){
    if(fileOpen){
        gtk_widget_get_allocation(imageWindow, displayImageAllocation);

        GdkPixbuf* scaledPixbuf;
        if(displayImageAllocation->width/displayImageAllocation->height<imageRatio){
            //image is wider than it is tall, compared to the viewport size
            //width is the limiting factor so it should be used to determine scale
            scaledPixbuf = gdk_pixbuf_scale_simple(gridPixbufs[cursorZ],displayImageAllocation->width,displayImageAllocation->width/imageRatio,GDK_INTERP_TILES);
        }else{
            //image is taller than it is wide, compared to the viewport size
            //height is the limiting factor so it should be used to determine scale
            scaledPixbuf = gdk_pixbuf_scale_simple(gridPixbufs[cursorZ],imageRatio*displayImageAllocation->height,displayImageAllocation->height,GDK_INTERP_TILES);
        }

        double imageTileSize = (double)gdk_pixbuf_get_width(scaledPixbuf)/(double)gridWidth;
        double imageTileToCursorRatio = imageTileSize/(double)gdk_pixbuf_get_width(cursorPixbuf);
        gdk_pixbuf_composite(cursorPixbuf,scaledPixbuf,((double)cursorX*imageTileSize),((double)cursorY*imageTileSize),imageTileSize,imageTileSize,((double)cursorX*imageTileSize),((double)cursorY*imageTileSize),imageTileToCursorRatio,imageTileToCursorRatio,GDK_INTERP_TILES,255);

        gtk_image_set_from_pixbuf(GTK_IMAGE(displayImage),scaledPixbuf);
    }
}



void audioListCursorUpdate(){
    GtkTreeSelection* selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(audioListView));

    if(gtk_tree_selection_count_selected_rows(selection)){
        gtk_widget_set_sensitive(audioListRemoveItem,TRUE);
    }else{
        gtk_widget_set_sensitive(audioListRemoveItem,FALSE);
    }
}

void fileSavedUpdate(gboolean val){
    fileSaved = val;
}

void fileOpenUpdate(gboolean val){
    fileOpen = val;

    //enable/disable save widgets based on whether a file is open or not
    gtk_widget_set_sensitive(saveItem,val);
    gtk_widget_set_sensitive(saveAndRunItem,val);

    gtk_widget_set_sensitive(audioListNewSourceItem,val);
    gtk_widget_set_sensitive(audioListNewOutputItem,val);

    //just disable remove item widget(until a new selection is made)
    gtk_widget_set_sensitive(audioListRemoveItem,FALSE);
    
}

void gridSizeXUpdate(){
    newGridWidth = gtk_spin_button_get_value(GTK_SPIN_BUTTON(gridSizeX));
}

void gridSizeYUpdate(){
    newGridHeight = gtk_spin_button_get_value(GTK_SPIN_BUTTON(gridSizeY));
}

void gridSizeZUpdate(){
    newGridDepth = gtk_spin_button_get_value(GTK_SPIN_BUTTON(gridSizeZ));
}



void cursorXUpdate(){
    cursorX = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonX));
}

void cursorYUpdate(){
    cursorY = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonY));
}

void cursorZUpdate(){
    cursorZ = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonZ));
}



void cursorButtonXUpdate(){
    cursorXUpdate();
    updateDisplayImage();
}

void cursorButtonYUpdate(){
    cursorYUpdate();
    updateDisplayImage();
}

void cursorButtonZUpdate(){
    cursorZUpdate();
    updateDisplayImage();
}




void timestepsUpdate(){
    timesteps = gtk_spin_button_get_value(GTK_SPIN_BUTTON(simSettingTimesteps));
}

void blockWidthUpdate(){
    blockWidth = gtk_spin_button_get_value(GTK_SPIN_BUTTON(simSettingBlockWidth));
}

void blockHeightUpdate(){
    blockHeight = gtk_spin_button_get_value(GTK_SPIN_BUTTON(simSettingBlockHeight));
}

void blockDepthUpdate(){
    blockDepth = gtk_spin_button_get_value(GTK_SPIN_BUTTON(simSettingBlockDepth));
}

void folderUpdate(){
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Select Output Folder",GTK_WINDOW(window),GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,"Cancel",GTK_RESPONSE_CANCEL,"Open",GTK_RESPONSE_ACCEPT,NULL);

    if(gtk_dialog_run(GTK_DIALOG(dialog))==GTK_RESPONSE_ACCEPT){
        currentOutFolder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        gtk_label_set_text(GTK_LABEL(simSettingFolderText),currentOutFolder);
    }
    gtk_widget_destroy(dialog);
}



void audioListStoreAppend(gboolean source,int x,int y,int z,const char *name){
    GtkTreeIter iter;
    gtk_list_store_append(audioListStore,&iter);

    if(source){
        gtk_list_store_set(audioListStore,&iter,COLUMN_ICON,"audio-speakers-symbolic",COLUMN_SOURCE,TRUE,COLUMN_X,x,COLUMN_Y,y,COLUMN_Z,z,COLUMN_NAME,name,-1);
    }else{
        gtk_list_store_set(audioListStore,&iter,COLUMN_ICON,"audio-input-microphone-symbolic",COLUMN_SOURCE,FALSE,COLUMN_X,x,COLUMN_Y,y,COLUMN_Z,z,COLUMN_NAME,name,-1);
    }
    fileSavedUpdate(FALSE);
}

void audioListStoreRemoveSelected(){
    GtkTreeIter iter;
    GtkTreeSelection* selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(audioListView));
    gtk_tree_selection_get_selected(selection,NULL,&iter);
    gtk_list_store_remove(audioListStore,&iter);
    fileSavedUpdate(FALSE);
}

void audioListStoreClear(){
    gtk_list_store_clear(audioListStore);
    fileSavedUpdate(FALSE);
}



void loadFolder(char *inFolder){
    fileOpenUpdate(FALSE);
    fileSavedUpdate(FALSE);

    currentInFolder = inFolder;

    audioListStoreClear();

    //open audio ledger
    FILE *audioLedgerFile = NULL;
    if(inFolder!=NULL){
        char *inFile = (char *)calloc(strlen(inFolder)+strlen(AUDIO_LEDGER_NAME)+2, sizeof(char));
        strcpy(inFile,inFolder);
        strcat(inFile,"/");
        strcat(inFile,AUDIO_LEDGER_NAME);

        audioLedgerFile = fopen(inFile,"r");
        free(inFile);
    }

    if(audioLedgerFile!=NULL){   
        int x,y,z;
        char *audioFileName;

        //load the next line from the output audio ledger and continue if it's not NULL
        while((audioFileName = readAudioLedgerLine(audioLedgerFile,&x,&y,&z))!=NULL){
            audioListStoreAppend(TRUE,x,y,z,audioFileName);
            free(audioFileName);
        }
        fclose(audioLedgerFile);
    }

    //open audio output ledger
    FILE *audioOutputLedgerFile = NULL;
    if(inFolder!=NULL){
        char *inFile = (char *)calloc(strlen(inFolder)+strlen(AUDIO_OUT_LEDGER_NAME)+2, sizeof(char));
        strcpy(inFile,inFolder);
        strcat(inFile,"/");
        strcat(inFile,AUDIO_OUT_LEDGER_NAME);

        audioOutputLedgerFile = fopen(inFile,"r");
        free(inFile);
    }

    if(audioOutputLedgerFile!=NULL){   
        int x,y,z;
        char *audioFileName;

        //load the next line from the output audio ledger and continue if it's not NULL
        while((audioFileName = readAudioLedgerLine(audioOutputLedgerFile,&x,&y,&z))!=NULL){
            audioListStoreAppend(FALSE,x,y,z,audioFileName);
            free(audioFileName);
        }
        fclose(audioOutputLedgerFile);
    }

    //open grid file
    FILE *inGridFile = NULL;
    if(inFolder!=NULL){
        char *inFile = (char *)calloc(strlen(inFolder)+strlen(SIM_STATE_NAME)+2, sizeof(char));
        strcpy(inFile,inFolder);
        strcat(inFile,"/");
        strcat(inFile,SIM_STATE_NAME);

        inGridFile = fopen(inFile,"rb");
        free(inFile);
    }

    if(inGridFile!=NULL){
        readHeaderBinary(inGridFile,&gridWidth,&gridHeight,&gridDepth);
    }
    
    gridArea = gridWidth*gridHeight*gridDepth;
    gridSize = gridWidth*gridHeight*gridDepth;

    free(grid);
    grid = (double *)calloc(gridSize,sizeof(double));

    if(inGridFile!=NULL){
        readDoublesBinary(inGridFile,grid,gridArea);
        fclose(inGridFile);

        fileSavedUpdate(TRUE);
    }

    free(gridImageData);
    gridImageData = (guchar *)malloc(gridSize*SAMPLES_PER_PIXEL*sizeof(guchar));

    doublesToGuchar(grid,gridImageData,gridArea);

    free(gridPixbufs);
    gridPixbufs = (GdkPixbuf **)malloc(gridDepth*sizeof(GdkPixbuf *));

    gucharToPixbufs(gridImageData,gridPixbufs,gridWidth,gridHeight,gridDepth);
    imageRatio = (double)gridWidth/(double)gridHeight;
    
    gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonX),0,gridWidth-1);
    gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonY),0,gridHeight-1);
    gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonZ),0,gridDepth-1);

    gtk_spin_button_set_value(GTK_SPIN_BUTTON(cursorButtonX),0);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(cursorButtonY),0);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(cursorButtonZ),0);

    cursorXUpdate();
    cursorYUpdate();
    cursorZUpdate();

    fileOpenUpdate(TRUE);

    updateDisplayImage();
}

void saveFolder(char *outFolder){
    //open audio ledger
    FILE *audioLedgerFile = NULL;
    FILE *audioOutputLedgerFile = NULL;
    if(outFolder!=NULL){
        char *ledgerFile = (char *)calloc(strlen(outFolder)+strlen(AUDIO_LEDGER_NAME)+2, sizeof(char));
        strcpy(ledgerFile,outFolder);
        strcat(ledgerFile,"/");
        strcat(ledgerFile,AUDIO_LEDGER_NAME);

        audioLedgerFile = fopen(ledgerFile,"w");
        free(ledgerFile);

        char *ledgerOutputFile = (char *)calloc(strlen(outFolder)+strlen(AUDIO_OUT_LEDGER_NAME)+2, sizeof(char));
        strcpy(ledgerOutputFile,outFolder);
        strcat(ledgerOutputFile,"/");
        strcat(ledgerOutputFile,AUDIO_OUT_LEDGER_NAME);

        audioOutputLedgerFile = fopen(ledgerOutputFile,"w");
        free(ledgerOutputFile);
    }

    if(audioLedgerFile!=NULL && audioOutputLedgerFile!=NULL){
        GtkTreeIter iter;
        if(gtk_tree_model_get_iter_first(GTK_TREE_MODEL(audioListStore),&iter)){
            do{
                gboolean source;
                int x,y,z;
                char *name;
                gtk_tree_model_get(GTK_TREE_MODEL(audioListStore),&iter,COLUMN_SOURCE,&source,COLUMN_X,&x,COLUMN_Y,&y,COLUMN_Z,&z,COLUMN_NAME,&name,-1);

                if(source){
                    writeAudioLedgerLine(audioLedgerFile,name,x,y,z);
                }else{
                    writeAudioLedgerLine(audioOutputLedgerFile,name,x,y,z);
                }
                free(name);
            }while(gtk_tree_model_iter_next(GTK_TREE_MODEL(audioListStore),&iter));
        }

        fclose(audioLedgerFile);
        fclose(audioOutputLedgerFile);
    }

    FILE *outGridFile;
    if(outFolder!=NULL){
        char *outFile = (char *)calloc(strlen(outFolder)+strlen(SIM_STATE_NAME)+2, sizeof(char));
        strcpy(outFile,outFolder);
        strcat(outFile,"/");
        strcat(outFile,SIM_STATE_NAME);
        outGridFile = fopen(outFile,"wb");
        free(outFile);
    }

    if(outGridFile!=NULL){
        writeHeaderBinary(outGridFile,&gridWidth,&gridHeight,&gridDepth);
        writeDoublesBinary(outGridFile,grid,gridArea);
        fclose(outGridFile);
    }

    currentInFolder = outFolder;
    fileSavedUpdate(TRUE);
}



void openItemFunction(){
    //prompt user to select folder to open
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Select Simulation Folder",GTK_WINDOW(window),GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,"Cancel",GTK_RESPONSE_CANCEL,"Open",GTK_RESPONSE_ACCEPT,NULL);

    if(gtk_dialog_run(GTK_DIALOG(dialog))==GTK_RESPONSE_ACCEPT){
        //load the folder
        char *inFolder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        loadFolder(inFolder);
    }
    gtk_widget_destroy(dialog);
}

void newItemFunction(){
    //prompt user to choose basic settings
    GtkWidget* settingDialog = gtk_dialog_new_with_buttons("Simulation Settings",GTK_WINDOW(window),GTK_DIALOG_MODAL|GTK_DIALOG_DESTROY_WITH_PARENT,"Cancel",GTK_RESPONSE_CANCEL,"Create",GTK_RESPONSE_ACCEPT,NULL);
    GtkWidget* settingDialogOptionBox = gtk_dialog_get_content_area(GTK_DIALOG(settingDialog));

    gridSizeX = gtk_spin_button_new_with_range(1,INT_MAX,1);
    gtk_box_pack_start(GTK_BOX(settingDialogOptionBox),gridSizeX,TRUE,TRUE,0);
    g_signal_connect(G_OBJECT(gridSizeX),"value-changed",G_CALLBACK(gridSizeXUpdate),NULL);
    gridSizeXUpdate();

    gridSizeY = gtk_spin_button_new_with_range(1,INT_MAX,1);
    gtk_box_pack_start(GTK_BOX(settingDialogOptionBox),gridSizeY,TRUE,TRUE,0);
    g_signal_connect(G_OBJECT(gridSizeY),"value-changed",G_CALLBACK(gridSizeYUpdate),NULL);
    gridSizeYUpdate();

    gridSizeZ = gtk_spin_button_new_with_range(1,INT_MAX,1);
    gtk_box_pack_start(GTK_BOX(settingDialogOptionBox),gridSizeZ,TRUE,TRUE,0);
    g_signal_connect(G_OBJECT(gridSizeZ),"value-changed",G_CALLBACK(gridSizeZUpdate),NULL);
    gridSizeZUpdate();

    gtk_window_set_resizable(GTK_WINDOW(settingDialog), FALSE);

    gtk_widget_show_all(settingDialog);

    if(gtk_dialog_run(GTK_DIALOG(settingDialog))==GTK_RESPONSE_ACCEPT){
        //set header info
        gridWidth = newGridWidth;
        gridHeight = newGridHeight;
        gridDepth = newGridDepth;

        //load everything using an empty grid, as opposed to a grid from a file
        loadFolder(NULL);
    }
    gtk_widget_destroy(settingDialog);
}

void saveItemFunction(){
    //prompt user to select folder to open
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Select Simulation Folder",GTK_WINDOW(window),GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,"Cancel",GTK_RESPONSE_CANCEL,"Save",GTK_RESPONSE_ACCEPT,NULL);
    
    if(currentInFolder!=NULL){
        gtk_file_chooser_set_filename(GTK_FILE_CHOOSER(dialog),currentInFolder);
    }

    if(gtk_dialog_run(GTK_DIALOG(dialog))==GTK_RESPONSE_ACCEPT){
        //save the folder
        char *outFolder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        saveFolder(outFolder);
    }
    gtk_widget_destroy(dialog);
}

void saveAndRunItemFunction(){
    if(!fileSaved){
        saveItemFunction();
    }

    if(fileSaved){
        //prompt user to choose basic settings
        GtkWidget* settingDialog = gtk_dialog_new_with_buttons("Simulation Settings",GTK_WINDOW(window),GTK_DIALOG_MODAL|GTK_DIALOG_DESTROY_WITH_PARENT,"Cancel",GTK_RESPONSE_CANCEL,"Run",GTK_RESPONSE_ACCEPT,NULL);
        GtkWidget* settingDialogOptionBox = gtk_dialog_get_content_area(GTK_DIALOG(settingDialog));

        //the output folder settings
        GtkWidget* simSettingFolderFrame = gtk_frame_new("Output Folder");
        gtk_container_set_border_width(GTK_CONTAINER(simSettingFolderFrame),DEFAULT_PADDING);
        gtk_box_pack_start(GTK_BOX(settingDialogOptionBox),simSettingFolderFrame,TRUE,TRUE,0);

        GtkWidget* simSettingFolderBox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL,0);
        gtk_container_add(GTK_CONTAINER(simSettingFolderFrame),simSettingFolderBox);

        simSettingFolderButton = gtk_button_new_from_icon_name("folder-new",GTK_ICON_SIZE_BUTTON);
        gtk_box_pack_start(GTK_BOX(simSettingFolderBox),simSettingFolderButton,FALSE,FALSE,0);
        g_signal_connect(G_OBJECT(simSettingFolderButton),"clicked",G_CALLBACK(folderUpdate),NULL);

        simSettingFolderText = gtk_label_new(currentOutFolder);
        gtk_box_pack_start(GTK_BOX(simSettingFolderBox),simSettingFolderText,FALSE,FALSE,DEFAULT_PADDING);

        //the timestep settings
        GtkWidget* simSettingTimestepFrame = gtk_frame_new("Time Steps");
        gtk_container_set_border_width(GTK_CONTAINER(simSettingTimestepFrame),DEFAULT_PADDING);
        gtk_box_pack_start(GTK_BOX(settingDialogOptionBox),simSettingTimestepFrame,TRUE,TRUE,0);

        simSettingTimesteps = gtk_spin_button_new_with_range(0,INT_MAX,1);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(simSettingTimesteps),timesteps);
        gtk_container_add(GTK_CONTAINER(simSettingTimestepFrame),simSettingTimesteps);
        g_signal_connect(G_OBJECT(simSettingTimesteps),"value-changed",G_CALLBACK(timestepsUpdate),NULL);
        timestepsUpdate();

        //the block size settings
        GtkWidget* simSettingBlockSizeFrame = gtk_frame_new("Block Dimensions");
        gtk_container_set_border_width(GTK_CONTAINER(simSettingBlockSizeFrame),DEFAULT_PADDING);
        gtk_box_pack_start(GTK_BOX(settingDialogOptionBox),simSettingBlockSizeFrame,TRUE,TRUE,0);

        GtkWidget* simSettingBlockSizeBox = gtk_box_new(GTK_ORIENTATION_VERTICAL,0);
        gtk_container_add(GTK_CONTAINER(simSettingBlockSizeFrame),simSettingBlockSizeBox);

        simSettingBlockWidth = gtk_spin_button_new_with_range(0,INT_MAX,1);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(simSettingBlockWidth),blockWidth);
        gtk_box_pack_start(GTK_BOX(simSettingBlockSizeBox),simSettingBlockWidth,TRUE,TRUE,0);
        g_signal_connect(G_OBJECT(simSettingBlockWidth),"value-changed",G_CALLBACK(blockWidthUpdate),NULL);
        blockWidthUpdate();

        simSettingBlockHeight = gtk_spin_button_new_with_range(0,INT_MAX,1);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(simSettingBlockHeight),blockHeight);
        gtk_box_pack_start(GTK_BOX(simSettingBlockSizeBox),simSettingBlockHeight,TRUE,TRUE,0);
        g_signal_connect(G_OBJECT(simSettingBlockHeight),"value-changed",G_CALLBACK(blockHeightUpdate),NULL);
        blockHeightUpdate();

        simSettingBlockDepth = gtk_spin_button_new_with_range(0,INT_MAX,1);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(simSettingBlockDepth),blockDepth);
        gtk_box_pack_start(GTK_BOX(simSettingBlockSizeBox),simSettingBlockDepth,TRUE,TRUE,0);
        g_signal_connect(G_OBJECT(simSettingBlockDepth),"value-changed",G_CALLBACK(blockDepthUpdate),NULL);
        blockDepthUpdate();

        gtk_window_set_resizable(GTK_WINDOW(settingDialog), FALSE);

        gtk_widget_show_all(settingDialog);

        if(gtk_dialog_run(GTK_DIALOG(settingDialog))==GTK_RESPONSE_ACCEPT){
            char *runCommand = (char *)malloc(strlen(BASE_COMMAND)*sizeof(char));
            strcpy(runCommand,BASE_COMMAND);

            if(currentInFolder!=NULL){
                runCommand = appendCommandLineFlag(runCommand,INPUT_FLAG,currentInFolder);
            }
            if(currentOutFolder!=NULL){
                runCommand = appendCommandLineFlag(runCommand,OUTPUT_FLAG,currentOutFolder);
            }
            if(timesteps){
                int len = snprintf(NULL,0,"%d",timesteps);
                char* intStr = (char *)malloc(len+1);
                snprintf(intStr,len+1,"%d",timesteps);

                runCommand = appendCommandLineFlag(runCommand,TIMESTEP_FLAG,intStr);
            }
            if(blockWidth && blockHeight && blockDepth){
                int len = snprintf(NULL,0,"%d %d %d",blockWidth,blockHeight,blockDepth);
                char* intStr = (char *)malloc(len+1);
                snprintf(intStr,len+1,"%d %d %d",blockWidth,blockHeight,blockDepth);

                runCommand = appendCommandLineFlag(runCommand,BLOCKSIZE_FLAG,intStr);
            }

            g_print("---\n");
            g_print("%s\n",runCommand);
            g_print("---\n");

            system(runCommand);

            g_print("---\n");
            
            //load folder of output
            loadFolder(currentOutFolder);
        }
        gtk_widget_destroy(settingDialog);
    }
}

void audioListNewSourceItemFunction(){
    //prompt user to name new audio file
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Select Audio Input File",GTK_WINDOW(window),GTK_FILE_CHOOSER_ACTION_OPEN,"Cancel",GTK_RESPONSE_CANCEL,"Add",GTK_RESPONSE_ACCEPT,NULL);
    
    if(gtk_dialog_run(GTK_DIALOG(dialog))==GTK_RESPONSE_ACCEPT){
        char *name = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        audioListStoreAppend(TRUE,cursorX,cursorY,cursorZ,name);
    }
    gtk_widget_destroy(dialog);
}

void audioListNewOutputItemFunction(){
    //prompt user to select audio file to open
    GtkWidget* settingDialog = gtk_dialog_new_with_buttons("Set Audio Output Name",GTK_WINDOW(window),GTK_DIALOG_MODAL|GTK_DIALOG_DESTROY_WITH_PARENT,"Cancel",GTK_RESPONSE_CANCEL,"Add",GTK_RESPONSE_ACCEPT,NULL);
    GtkWidget* settingDialogOptionBox = gtk_dialog_get_content_area(GTK_DIALOG(settingDialog));
    
    GtkWidget* nameEntry = gtk_entry_new();
    gtk_box_pack_start(GTK_BOX(settingDialogOptionBox),nameEntry,TRUE,TRUE,0);

    gtk_window_set_resizable(GTK_WINDOW(settingDialog), FALSE);
    
    gtk_widget_show_all(settingDialog);
    if(gtk_dialog_run(GTK_DIALOG(settingDialog))==GTK_RESPONSE_ACCEPT){
        GtkEntryBuffer* nameEntryBuffer = gtk_entry_get_buffer(GTK_ENTRY(nameEntry));
        const char *name = gtk_entry_buffer_get_text(nameEntryBuffer);
        audioListStoreAppend(FALSE,cursorX,cursorY,cursorZ,name);
    }
    gtk_widget_destroy(settingDialog);
}

void audioListRemoveItemFunction(){
    audioListStoreRemoveSelected();
}


int main(int argc,char *argv[]){
    gridArea = gridWidth*gridHeight*gridDepth;
    gridSize = gridWidth*gridHeight*gridDepth;

    grid = (double *)calloc(gridSize,sizeof(double));
    gridImageData = (guchar *)malloc(gridSize*sizeof(guchar));
    gridPixbufs = (GdkPixbuf **)malloc(gridDepth*sizeof(GdkPixbuf *));



    gtk_init(&argc,&argv);
    
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_default_size(GTK_WINDOW(window),1200,600);
    gtk_window_set_title(GTK_WINDOW(window),"FDTDAudioCUDA Visualizer");
    g_signal_connect(G_OBJECT(window),"destroy",G_CALLBACK(gtk_main_quit),NULL);

    GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL,0);
    gtk_container_add(GTK_CONTAINER(window),box);



    menuBar = gtk_menu_bar_new();

    fileMenuItem = gtk_menu_item_new_with_label("File");
    gtk_menu_shell_append(GTK_MENU_SHELL(menuBar),fileMenuItem);

    fileMenu = gtk_menu_new();
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(fileMenuItem),fileMenu);

    newItem = gtk_menu_item_new_with_label("New");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),newItem);
    g_signal_connect(G_OBJECT(newItem),"activate",G_CALLBACK(newItemFunction),NULL);

    openItem = gtk_menu_item_new_with_label("Open");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),openItem);
    g_signal_connect(G_OBJECT(openItem),"activate",G_CALLBACK(openItemFunction),NULL);
    
    GtkWidget* sepItem = gtk_separator_menu_item_new();
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),sepItem);

    saveItem = gtk_menu_item_new_with_label("Save");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),saveItem);
    g_signal_connect(G_OBJECT(saveItem),"activate",G_CALLBACK(saveItemFunction),NULL);

    saveAndRunItem = gtk_menu_item_new_with_label("Save + Run");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),saveAndRunItem);
    g_signal_connect(G_OBJECT(saveAndRunItem),"activate",G_CALLBACK(saveAndRunItemFunction),NULL);

    GtkWidget* sepItem2 = gtk_separator_menu_item_new();
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),sepItem2);

    quitItem = gtk_menu_item_new_with_label("Quit");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),quitItem);
    g_signal_connect(G_OBJECT(quitItem),"activate",G_CALLBACK(gtk_main_quit),NULL);

    gtk_box_pack_start(GTK_BOX(box),menuBar,FALSE,FALSE,0);
    


    GtkWidget* viewBox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL,0);
    //gtk_container_set_border_width(GTK_CONTAINER(viewBox),DEFAULT_PADDING);
    gtk_box_pack_start(GTK_BOX(box),viewBox,TRUE,TRUE,DEFAULT_PADDING);

    imageWindow = gtk_scrolled_window_new(NULL,NULL);
    gtk_box_pack_start(GTK_BOX(viewBox),imageWindow,TRUE,TRUE,DEFAULT_PADDING);

    displayImage = gtk_image_new();
    gtk_container_add(GTK_CONTAINER(imageWindow),displayImage);
    g_signal_connect(G_OBJECT(window),"size-allocate",G_CALLBACK(updateDisplayImage),NULL);

    displayImageAllocation = g_new(GtkAllocation, 1);
    gtk_widget_get_allocation(imageWindow, displayImageAllocation);

    cursorPixbuf = gdk_pixbuf_new_from_file("cursor.png",NULL);



    //frame for all audio list related widgets
    GtkWidget* audioListWindow = gtk_frame_new("Audio Sources/Outputs");
    gtk_box_pack_start(GTK_BOX(viewBox),audioListWindow,FALSE,FALSE,0);

    //audio list window box contains table and add/remove buttons
    GtkWidget* audioListWindowBox = gtk_box_new(GTK_ORIENTATION_VERTICAL,0);
    gtk_container_add(GTK_CONTAINER(audioListWindow),audioListWindowBox);

    GtkWidget* audioListButtons = gtk_box_new(GTK_ORIENTATION_HORIZONTAL,0);
    gtk_box_pack_start(GTK_BOX(audioListWindowBox),audioListButtons,FALSE,FALSE,0);

    audioListNewSourceItem = gtk_button_new_from_icon_name("audio-speakers-symbolic",GTK_ICON_SIZE_BUTTON);
    gtk_box_pack_start(GTK_BOX(audioListButtons),audioListNewSourceItem,FALSE,FALSE,0);
    g_signal_connect(G_OBJECT(audioListNewSourceItem),"clicked",G_CALLBACK(audioListNewSourceItemFunction),NULL);

    audioListNewOutputItem = gtk_button_new_from_icon_name("audio-input-microphone-symbolic",GTK_ICON_SIZE_BUTTON);
    gtk_box_pack_start(GTK_BOX(audioListButtons),audioListNewOutputItem,FALSE,FALSE,0);
    g_signal_connect(G_OBJECT(audioListNewOutputItem),"clicked",G_CALLBACK(audioListNewOutputItemFunction),NULL);

    audioListRemoveItem = gtk_button_new_from_icon_name("list-remove",GTK_ICON_SIZE_BUTTON);
    gtk_box_pack_start(GTK_BOX(audioListButtons),audioListRemoveItem,FALSE,FALSE,0);
    g_signal_connect(G_OBJECT(audioListRemoveItem),"clicked",G_CALLBACK(audioListRemoveItemFunction),NULL);

    GtkWidget* sep = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_pack_start(GTK_BOX(audioListWindowBox),sep,FALSE,FALSE,0);

    //view list
    audioListStore = gtk_list_store_new(N_COLUMNS,G_TYPE_STRING,G_TYPE_BOOLEAN,G_TYPE_INT,G_TYPE_INT,G_TYPE_INT,G_TYPE_STRING);

    audioListView = gtk_tree_view_new_with_model(GTK_TREE_MODEL(audioListStore));
    gtk_box_pack_start(GTK_BOX(audioListWindowBox),audioListView,TRUE,TRUE,0);
    g_signal_connect(G_OBJECT(audioListView),"cursor-changed",G_CALLBACK(audioListCursorUpdate),NULL);

    GtkCellRenderer* renderer;

    renderer = gtk_cell_renderer_pixbuf_new();
    GtkTreeViewColumn* columnIcon = gtk_tree_view_column_new();
    //gtk_tree_view_column_set_title(GTK_TREE_VIEW_COLUMN(columnIcon),"Type");
    gtk_tree_view_column_pack_start(columnIcon,renderer,FALSE);
    gtk_tree_view_column_set_attributes(columnIcon,renderer,"icon-name",COLUMN_ICON,NULL);
    gtk_tree_view_append_column(GTK_TREE_VIEW(audioListView),columnIcon);

    renderer = gtk_cell_renderer_text_new();
    GtkTreeViewColumn* columnX = gtk_tree_view_column_new_with_attributes("X",renderer,"text",COLUMN_X,NULL);
    gtk_tree_view_append_column(GTK_TREE_VIEW(audioListView),columnX);

    renderer = gtk_cell_renderer_text_new();
    GtkTreeViewColumn* columnY = gtk_tree_view_column_new_with_attributes("Y",renderer,"text",COLUMN_Y,NULL);
    gtk_tree_view_append_column(GTK_TREE_VIEW(audioListView),columnY);

    renderer = gtk_cell_renderer_text_new();
    GtkTreeViewColumn* columnZ = gtk_tree_view_column_new_with_attributes("Z",renderer,"text",COLUMN_Z,NULL);
    gtk_tree_view_append_column(GTK_TREE_VIEW(audioListView),columnZ);

    renderer = gtk_cell_renderer_text_new();
    GtkTreeViewColumn* columnName = gtk_tree_view_column_new_with_attributes("Name",renderer,"text",COLUMN_NAME,NULL);
    gtk_tree_view_append_column(GTK_TREE_VIEW(audioListView),columnName);



    GtkWidget* cursorBar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL,0);
    gtk_box_pack_end(GTK_BOX(box),cursorBar,FALSE,FALSE,0);

    cursorButtonX = gtk_spin_button_new_with_range(0,gridWidth-1,1);
    gtk_box_pack_start(GTK_BOX(cursorBar),cursorButtonX,TRUE,TRUE,0);
    g_signal_connect(G_OBJECT(cursorButtonX),"value-changed",G_CALLBACK(cursorButtonXUpdate),NULL);

    cursorButtonY = gtk_spin_button_new_with_range(0,gridHeight-1,1);
    gtk_box_pack_start(GTK_BOX(cursorBar),cursorButtonY,TRUE,TRUE,0);
    g_signal_connect(G_OBJECT(cursorButtonY),"value-changed",G_CALLBACK(cursorButtonYUpdate),NULL);

    cursorButtonZ = gtk_spin_button_new_with_range(0,gridDepth-1,1);
    gtk_box_pack_start(GTK_BOX(cursorBar),cursorButtonZ,TRUE,TRUE,0);
    g_signal_connect(G_OBJECT(cursorButtonZ),"value-changed",G_CALLBACK(cursorButtonZUpdate),NULL);



    fileOpenUpdate(FALSE);
    fileSavedUpdate(FALSE);
    
    gtk_widget_show_all(window);
    gtk_main();

    //Free memory
    free(grid);
    free(gridImageData);
    free(gridPixbufs);

    return 1;
}