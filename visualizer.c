#include<gtk/gtk.h> //displaying graphics
#include<stdio.h> //file loading
#include<math.h> //math functions

#define SIM_STATE_NAME "sim_state.bin"
#define AUDIO_LEDGER_NAME "audio_ledger.txt"
#define AUDIO_OUT_LEDGER "audio_out_ledger.txt"

#define SAMPLES_PER_PIXEL 3
#define BITS_PER_SAMPLE 8

#define BASE_COMMAND "./sim"

#define FLAG_FRONT " -"
#define FLAG_BACK " "

#define INPUT_FLAG "i"
#define OUTPUT_FLAG "o"
#define TIMESTEP_FLAG "t"
#define GRIDSIZE_FLAG "g"
#define BLOCKSIZE_FLAG "b"



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

//indicates whether a file is currently completely loaded into memory. Don't redraw stuff if one isn't
gboolean fileOpen;

//name of the folder currently loaded. NULL if a new grid is loaded or no folder is loaded.
char *currentInFolder = NULL;

//name of the folder the simulator will output to
char *currentOutFolder = "output";

//timesteps of simulation
int timesteps = 0;

//block size to be used in simulation
int blockWidth = 0;
int blockHeight = 0;
int blockDepth = 0;

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



void fileSavedUpdate(gboolean val){
    fileSaved = val;
}

void fileOpenUpdate(gboolean val){
    fileOpen = val;

    //enable/disable save widgets based on whether a file is open or not
    gtk_widget_set_sensitive(saveItem,val);
    gtk_widget_set_sensitive(saveAndRunItem,val);
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



void loadFolder(char *inFolder){
    fileOpenUpdate(FALSE);
    fileSavedUpdate(FALSE);

    currentInFolder = inFolder;

    FILE *inGridFile;

    if(inFolder!=NULL){
        char *inFile = (char *)calloc(strlen(inFolder)+strlen(SIM_STATE_NAME)+2, sizeof(char));
        strcpy(inFile,inFolder);
        strcat(inFile,"/");
        strcat(inFile,SIM_STATE_NAME);

        inGridFile = fopen(inFile,"rb");
        free(inFile);
    }

    if(inFolder!=NULL && inGridFile!=NULL){
        readHeaderBinary(inGridFile,&gridWidth,&gridHeight,&gridDepth);
    }
    
    gridArea = gridWidth*gridHeight*gridDepth;
    gridSize = gridWidth*gridHeight*gridDepth;

    free(grid);
    grid = (double *)calloc(gridSize,sizeof(double));

    if(inFolder!=NULL && inGridFile!=NULL){
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
    FILE *outGridFile;

    if(outFolder!=NULL){
        char *outFile = (char *)calloc(strlen(outFolder)+strlen(SIM_STATE_NAME)+2, sizeof(char));
        strcpy(outFile,outFolder);
        strcat(outFile,"/");
        strcat(outFile,SIM_STATE_NAME);

        outGridFile = fopen(outFile,"wb");

        writeHeaderBinary(outGridFile,&gridWidth,&gridHeight,&gridDepth);
        writeDoublesBinary(outGridFile,grid,gridArea);
        fclose(outGridFile);
        free(outFile);

        currentInFolder = outFolder;
        fileSavedUpdate(TRUE);
    }
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
        //TODO:add option to change output folder, timesteps, and block size

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
            int lenW = snprintf(NULL,0,"%d ",blockWidth);
            int lenH = snprintf(NULL,0,"%d ",blockHeight);
            int lenD = snprintf(NULL,0,"%d",blockDepth);

            char* intStr = (char *)malloc(lenW+lenH+lenD+1);

            snprintf(intStr,lenW+1,"%d ",blockWidth);
            snprintf(intStr+lenW,lenH+1,"%d ",blockHeight);
            snprintf(intStr+lenW+lenH,lenD+1,"%d",blockDepth);

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
}



int main(int argc,char *argv[]){
    gridArea = gridWidth*gridHeight*gridDepth;
    gridSize = gridWidth*gridHeight*gridDepth;

    grid = (double *)calloc(gridSize,sizeof(double));
    gridImageData = (guchar *)malloc(gridSize*sizeof(guchar));
    gridPixbufs = (GdkPixbuf **)malloc(gridDepth*sizeof(GdkPixbuf *));



    gtk_init(&argc,&argv);
    
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_default_size(GTK_WINDOW(window),600,400);
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
    


    imageWindow = gtk_scrolled_window_new(NULL,NULL);
    gtk_box_pack_start(GTK_BOX(box),imageWindow,TRUE,TRUE,0);

    displayImage = gtk_image_new();
    gtk_container_add(GTK_CONTAINER(imageWindow),displayImage);
    g_signal_connect(G_OBJECT(window),"size-allocate",G_CALLBACK(updateDisplayImage),NULL);

    displayImageAllocation = g_new(GtkAllocation, 1);
    gtk_widget_get_allocation(imageWindow, displayImageAllocation);



    cursorPixbuf = gdk_pixbuf_new_from_file("cursor.png",NULL);

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