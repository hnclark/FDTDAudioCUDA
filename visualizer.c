#include<gtk/gtk.h>
#include<stdio.h>

GtkWidget* window;

GtkWidget* displayImage;

int gridWidth = 1;
int gridHeight = 1;
int gridDepth = 1;

int gridArea = 1;
size_t gridSize = 1;

double *grid;

int cursorX = 0;
int cursorY = 0;
int cursorZ = 0;

GtkWidget* cursorButtonX;
GtkWidget* cursorButtonY;
GtkWidget* cursorButtonZ;



//helper function to read header from a binary file
void readHeaderBinary_c(FILE *fileIn,int *w,int *h,int *d){
    fread(w,sizeof(int),1,fileIn);
    fread(h,sizeof(int),1,fileIn);
    fread(d,sizeof(int),1,fileIn);
}

//helper function to write header to a binary file
void writeHeaderBinary_c(FILE *fileOut,int *w,int *h,int *d){
    fwrite(w,sizeof(int),1,fileOut);
    fwrite(h,sizeof(int),1,fileOut);
    fwrite(d,sizeof(int),1,fileOut);
}

//helper function to read grid from a binary file
void readDoublesBinary_c(FILE *fileIn,double *array,int arrayLen){
    fread(array,sizeof(double),arrayLen,fileIn);
}

//helper function to write grid to a binary file
void writeDoublesBinary_c(FILE *fileOut,double *array,int arrayLen){
    fwrite(array,sizeof(double),arrayLen,fileOut);
}



void cursorButtonXUpdate(){
    cursorX = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonX));
    g_print("x = %d\n",cursorX);
}

void cursorButtonYUpdate(){
    int cursorY = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonY));
    g_print("y = %d\n",cursorY);
}

void cursorButtonZUpdate(){
    int cursorZ = gtk_spin_button_get_value(GTK_SPIN_BUTTON(cursorButtonZ));
    gtk_image_set_from_file(GTK_IMAGE(displayImage),"test.jpg");
    g_print("z = %d\n",cursorZ);
}

void openItemFunction(){
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Open Folder",GTK_WINDOW(window),GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,"Cancel",GTK_RESPONSE_CANCEL,"Open",GTK_RESPONSE_ACCEPT,NULL);

    if(gtk_dialog_run(GTK_DIALOG(dialog))==GTK_RESPONSE_ACCEPT){
        char *inFolder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        char *inFile = (char *)calloc(strlen(inFolder)+strlen("/sim_state.bin")+1, sizeof(char));
        strcpy(inFile,inFolder);
        strcat(inFile,"/sim_state.bin");

        FILE *inGridFile;
        inGridFile = fopen(inFile,"rb");

        if(inGridFile!=NULL){
            readHeaderBinary_c(inGridFile,&gridWidth,&gridHeight,&gridDepth);

            gridArea = gridWidth*gridHeight*gridDepth;
            gridSize = gridWidth*gridHeight*gridDepth;

            free(grid);
            grid = (double *)calloc(gridSize,sizeof(double));

            readDoublesBinary_c(inGridFile,grid,gridArea);
            fclose(inGridFile);

            gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonX),0,gridWidth-1);
            gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonY),0,gridHeight-1);
            gtk_spin_button_set_range(GTK_SPIN_BUTTON(cursorButtonZ),0,gridDepth-1);
        }
    }
    gtk_widget_destroy(dialog);
}



int main(int argc,char *argv[]){
    grid = (double *)calloc(gridSize,sizeof(double));
    
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
    


    displayImage = gtk_image_new();
    gtk_box_pack_start(GTK_BOX(box),displayImage,FALSE,FALSE,0);



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