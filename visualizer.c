/* 
    -Uses libsndfile, covered by the GNU LGPL
*/
#include<sndfile.h>

#include<gtk/gtk.h>
#include<stdio.h>

#define SIM_STATE_NAME "sim_state.bin"
#define SAMPLES_PER_PIXEL 3
#define BITS_PER_SAMPLE 8



typedef struct{
    guchar red;
    guchar green;
    guchar blue;
}pixel;



//Main window
GtkWidget* window;

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
gboolean fileOpen = FALSE;

//name of the folder currently loaded. NULL if a new grid is loaded or no folder is loaded.
char *currentInFolder = NULL;



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



pixel doubleToPixel(double val){
    pixel pix;
    pix.green = 0;
    if(val>0){
        pix.red = (guchar)abs((int)val*255);
        pix.blue = 0;
    }else{
        pix.red = 0;
        pix.blue = (guchar)abs((int)val*255);
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
    fileOpen = FALSE;
    currentInFolder = inFolder;

    FILE *inGridFile;

    if(inFolder!=NULL){
        char *inFile = (char *)calloc(strlen(inFolder)+strlen(SIM_STATE_NAME)+2, sizeof(char));
        strcpy(inFile,inFolder);
        strcat(inFile,"/");
        strcat(inFile,SIM_STATE_NAME);

        inGridFile = fopen(inFile,"rb");
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
    }

    free(gridImageData);
    gridImageData = (guchar *)calloc(gridSize*SAMPLES_PER_PIXEL,sizeof(guchar));

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

    fileOpen = TRUE;

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
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Select Simulation Folder",GTK_WINDOW(window),GTK_FILE_CHOOSER_ACTION_CREATE_FOLDER,"Cancel",GTK_RESPONSE_CANCEL,"Save",GTK_RESPONSE_ACCEPT,NULL);
    
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
    saveItemFunction();
    //TODO:add code to run cuda script. this should run using an already saved folder. in this case the one previously saved using saveItemFunction
}



int main(int argc,char *argv[]){
    gridArea = gridWidth*gridHeight*gridDepth;
    gridSize = gridWidth*gridHeight*gridDepth;

    grid = (double *)calloc(gridSize,sizeof(double));
    gridImageData = (guchar *)calloc(gridSize,sizeof(guchar));
    gridPixbufs = (GdkPixbuf **)malloc(gridDepth*sizeof(GdkPixbuf *));



    gtk_init(&argc,&argv);
    
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_default_size(GTK_WINDOW(window),600,400);
    gtk_window_set_title(GTK_WINDOW(window),"FDTDAudioCUDA Visualizer");
    g_signal_connect(G_OBJECT(window),"destroy",G_CALLBACK(gtk_main_quit),NULL);

    GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL,0);
    gtk_container_add(GTK_CONTAINER(window),box);



    GtkWidget* menuBar = gtk_menu_bar_new();

    GtkWidget* fileMenuItem = gtk_menu_item_new_with_label("File");
    gtk_menu_shell_append(GTK_MENU_SHELL(menuBar),fileMenuItem);

    GtkWidget* fileMenu = gtk_menu_new();
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(fileMenuItem),fileMenu);

    GtkWidget* newItem = gtk_menu_item_new_with_label("New");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),newItem);
    g_signal_connect(G_OBJECT(newItem),"activate",G_CALLBACK(newItemFunction),NULL);

    GtkWidget* openItem = gtk_menu_item_new_with_label("Open");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),openItem);
    g_signal_connect(G_OBJECT(openItem),"activate",G_CALLBACK(openItemFunction),NULL);
    
    GtkWidget* sepItem = gtk_separator_menu_item_new();
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),sepItem);

    GtkWidget* saveItem = gtk_menu_item_new_with_label("Save");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),saveItem);
    g_signal_connect(G_OBJECT(saveItem),"activate",G_CALLBACK(saveItemFunction),NULL);

    GtkWidget* saveAndRunItem = gtk_menu_item_new_with_label("Save + Run");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),saveAndRunItem);
    g_signal_connect(G_OBJECT(saveAndRunItem),"activate",G_CALLBACK(saveAndRunItemFunction),NULL);

    GtkWidget* sepItem2 = gtk_separator_menu_item_new();
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),sepItem2);

    GtkWidget* quitItem = gtk_menu_item_new_with_label("Quit");
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



    
    gtk_widget_show_all(window);
    gtk_main();

    free(grid);

    return 1;
}