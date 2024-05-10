import tkinter as tk
from PIL import ImageTk,Image




class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        global reset
        super().__init__(*args, **kwargs)
        self.title("Plantilla")
        self.geometry('800x600')
    
        self.contenedor=tk.Frame(self)
        self.contenedor.pack(side="top", fill="both",expand="true")
            

        self.contenedor.grid_rowconfigure(0,weight=1)
        self.contenedor.grid_columnconfigure(0,weight=1)	


        self.frames = {}

        BuildFrame=main_window
        frame =BuildFrame(self.contenedor, self)
        self.frames[BuildFrame]=frame
        frame.grid(row=0,column=0,sticky="nsew")
        self.show_frame(main_window)

    def show_frame(self,cont):
            frame = self.frames[cont]
            frame.tkraise()

class main_window(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        frame_men=tk.Frame(self,parent)
        frame_men.pack(side="top",anchor="center",pady=5)
    
        
        # Configuracion del menu ventana procipal
        self.opc_menu=tk.Menu(controller.contenedor)

        archivo_menu=tk.Menu(self.opc_menu, tearoff=0)
        archivo_menu.add_command(label="Abrir")
        archivo_menu.add_command(label="Salir")
        self.opc_menu.add_cascade(label="Archivo",menu=archivo_menu)

        acerca_menu=tk.Menu(self.opc_menu, tearoff=0)
        acerca_menu.add_command(label="Acerca de plantilla")
        self.opc_menu.add_cascade(label="Ayuda",menu=acerca_menu)

        controller.config(menu=self.opc_menu)	

        #Frames

        frame_main=tk.Frame(self,parent)
        frame_main.pack(side="top",anchor="center",pady=5)


        #frame izquierdo
        frame_left=tk.Frame(self,parent)
        frame_left.pack(side="left",anchor="n",pady=5)
        img = ImageTk.PhotoImage(Image.open("imgp.png"))  
        w_img = tk.Label(frame_left, image=img)
        w_img.image=img
        w_img.pack()
        #----------------------------------------------


        #frame derecho
        objects = ["Casa","Carro","Perro","Gato"]
        labels=[]
        count=0
        frame_right=tk.Frame(self,parent)
        frame_right.pack(side="right",anchor="n")

        frame_title=tk.Frame(frame_right,parent)
        frame_title.pack(side="top",anchor="center",padx=10)

        frame_list=tk.Frame(frame_right,parent)
        frame_list.pack(side="left",anchor="center",padx=20)


        label_title = tk.Label(frame_title, text="Objetos detectados",justify='center')
        label_title.pack(side="top",anchor="center",pady=10)
        label_title.config(font=("TkDefaultFont",20))
        
        for object in objects:	
             labels.append(tk.Label(frame_list,text=object,justify='left'))
             labels[count].grid(row = count, column = 1,pady=2)
             labels[count].config(font=("TkDefaultFont",12))
             count += 1
             	 
        #------------------------------------------------------------


        #frame_top=tk.Frame(self,parent)





App().mainloop()