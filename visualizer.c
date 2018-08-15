#include<gtk/gtk.h>
#include<stdio.h>

#define SIM_STATE_NAME "/sim_state.bin"
#define BYTES_PER_PIXEL 3


GtkWidget* window;

GtkWidget* imageWindow;
GtkAllocation* displayImageAllocation;
GtkWidget* displayImage;
double imageRatio;

int gridWidth = 1;
int gridHeight = 1;
int gridDepth = 1;

int gridArea;
size_t gridSize;

double *grid;
guchar *gridImageData;

GdkPixbuf **gridPixbufs;

int cursorX = 0;
int cursorY = 0;
int cursorZ = 0;

GtkWidget* cursorButtonX;
GtkWidget* cursorButtonY;
GtkWidget* cursorButtonZ;

gboolean fileOpen = FALSE;



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



guchar doubleToGucharRepresentation(double val){
    return (guchar)(int)val*255;
}

//helper function to convert an array of doubles to an array of guchars
void doublesToGuchar(double *arrayIn,guchar *arrayOut,int arrayLen){
    for(int i=0;i<arrayLen;i++){
        guchar value = doubleToGucharRepresentation(arrayIn[i]);
        arrayOut[i*3]=value;
        arrayOut[i*3+1]=value;
        arrayOut[i*3+2]=value;
    }
}

//helper function to convert an array of guchars to an array of pixbufs, one per image layer
void gucharToPixbufs(guchar *arrayIn,GdkPixbuf *pixbufs[],int imageWidth,int imageHeight,int imageCount){
    for(int i=0;i<imageCount;i++){
        guchar *imagePointer = arrayIn+(i*imageWidth*imageHeight*BYTES_PER_PIXEL);
        pixbufs[i] = gdk_pixbuf_new_from_data(imagePointer,GDK_COLORSPACE_RGB,FALSE,8,imageWidth,imageHeight,imageWidth*3,NULL,NULL);
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
        
        gtk_image_set_from_pixbuf(GTK_IMAGE(displayImage),scaledPixbuf);
    }
}



void cursorButtonXUpdate(){
    cursorX = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonX));
    g_print("x = %d\n",cursorX);
}

void cursorButtonYUpdate(){
    cursorY = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonY));
    g_print("y = %d\n",cursorY);
}

void cursorButtonZUpdate(){
    cursorZ = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonZ));
    updateDisplayImage();
    g_print("z = %d\n",cursorZ);
}

void openItemFunction(){
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Open Folder",GTK_WINDOW(window),GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,"Cancel",GTK_RESPONSE_CANCEL,"Open",GTK_RESPONSE_ACCEPT,NULL);

    if(gtk_dialog_run(GTK_DIALOG(dialog))==GTK_RESPONSE_ACCEPT){
        char *inFolder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        char *inFile = (char *)calloc(strlen(inFolder)+strlen(SIM_STATE_NAME)+1, sizeof(char));
        strcpy(inFile,inFolder);
        strcat(inFile,SIM_STATE_NAME);

        FILE *inGridFile;
        inGridFile = fopen(inFile,"rb");

        if(inGridFile!=NULL){
            readHeaderBinary(inGridFile,&gridWidth,&gridHeight,&gridDepth);

            gridHeight = gridHeight;

            gridArea = gridWidth*gridHeight*gridDepth;
            gridSize = gridWidth*gridHeight*gridDepth;

            free(grid);
            grid = (double *)calloc(gridSize,sizeof(double));

            readDoublesBinary(inGridFile,grid,gridArea);
            fclose(inGridFile);

            free(gridImageData);
            gridImageData = (guchar *)calloc(gridSize*BYTES_PER_PIXEL,sizeof(guchar));

            doublesToGuchar(grid,gridImageData,gridArea);

            free(gridPixbufs);
            gridPixbufs = (GdkPixbuf **)malloc(gridDepth*sizeof(GdkPixbuf *));

            gucharToPixbufs(gridImageData,gridPixbufs,gridWidth,gridHeight,gridDepth);
            imageRatio = (double)gridWidth/(double)gridHeight;            

            gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonX),0,gridWidth-1);
            gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonY),0,gridHeight-1);
            gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonZ),0,gridDepth-1);

            fileOpen = TRUE;
            
            updateDisplayImage();
        }
    }
    gtk_widget_destroy(dialog);
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

    GtkWidget* openItem = gtk_menu_item_new_with_label("Open");
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),openItem);
    g_signal_connect(G_OBJECT(openItem),"activate",G_CALLBACK(openItemFunction),NULL);
    
    GtkWidget* sepItem = gtk_separator_menu_item_new();
    gtk_menu_shell_append(GTK_MENU_SHELL(fileMenu),sepItem);

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