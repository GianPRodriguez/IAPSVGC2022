import multiprocessing
import queue
from threading import Thread
import tkinter as tk
from tkinter import ttk
from run import *
import subprocess
import asyncio
from PIL import ImageTk, Image

class LoremIpsum(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		# Container
		container = tk.Frame(self)
		container.pack(side="top", fill="both", expand=True)
		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)
		# Configuración Ventana
		self.title("Lorem ipsum dolor sit amet")
		self.geometry("500x309")
		self.resizable(False, False)
		self.iconbitmap("icon.ico")
		# Frames
		self.frames = {}
		for F in (StartPage, Modulo1, Modulo2, Modulo3):
			page_name = F.__name__
			frame = F(parent=container, controller=self)
			self.frames[page_name] = frame
			frame.grid(row=0, column=0, sticky="nsew")
		self.show_frame("StartPage")

	def show_frame(self, page_name):
		frame = self.frames[page_name]
		frame.tkraise()

class StartPage(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.controller = controller
		# Canvas Head
		canvasHead = tk.Canvas(self, width=500, height=50, bg="#154e72")
		canvasHead.place(x=0, y=0, anchor = tk.CENTER)
		canvasHead.create_text(250, 25, text="Lorem ipsum dolor sit amet", fill="white", font=('Verdana 15 bold'))
		canvasHead.pack()
		# Canvas Body
		canvasBody = tk.Canvas(self, width=500, height=259, bg="#154e72")
		canvasBody.place(x=0, y=50, anchor = tk.CENTER)
		canvasBody.pack()

		button1 = tk.Button(self, text='Módulo de Batallas contra Agentes', anchor="center", command=lambda: controller.show_frame("Modulo1"))
		button1.configure(width = 50, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button1_widget = canvasBody.create_window(70, 35, anchor=tk.NW, window=button1)

		button2 = tk.Button(self, text='Módulo de Análisis de Batallas', anchor="center", command=lambda: controller.show_frame("Modulo2"))
		button2.configure(width = 50, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button2_widget = canvasBody.create_window(70, 105, anchor=tk.NW, window=button2)

		button3 = tk.Button(self, text='Módulo de Entrenamiento', anchor="center", command=lambda: controller.show_frame("Modulo3"))
		button3.configure(width = 50, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button3_widget = canvasBody.create_window(70, 175, anchor=tk.NW, window=button3)

class Modulo1(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		# Canvas Head
		self.canvasHead = tk.Canvas(self, width=500, height=50, bg="#154e72")
		self.canvasHead.place(x=0, y=0, anchor = tk.CENTER)
		self.canvasHead.create_text(250, 25, text="Módulo de Batallas contra Agentes", fill="white", font=('Verdana 15 bold'))
		self.canvasHead.pack()
		# Canvas Body
		self.canvasBody = tk.Canvas(self, width=500, height=259, bg="#154e72")
		self.canvasBody.pack()
		# Botón Iniciar Batalla
		self.button1 = tk.Button(self, text='Iniciar Batalla', anchor="center")
		self.button1.configure(width = 23, height = 2, activebackground = "#33B5E5", relief = tk.FLAT, command= self.IniciarBatalla)
		self.button1_widget = self.canvasBody.create_window(250, 180, window=self.button1)
		# Label Nombre de Usuario
		self.labelUsername = tk.Label(self,text="Nombre de Usuario", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
		self.LabelUsername_widget = self.canvasBody.create_window(250, 30, window=self.labelUsername)
		# Label Equipo del Agente
		self.labelTeam = tk.Label(self,text="Equipo del Agente", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
		self.LabelTeam_widget = self.canvasBody.create_window(250, 85, window=self.labelTeam)
		# Entrada Nombre de Usuario
		self.strName = tk.StringVar()
		self.nameEntry = ttk.Entry(self, textvariable=self.strName, width=28)
		self.strName_widget = self.canvasBody.create_window(250, 55, window=self.nameEntry)
		# ListBox Equipo
		self.textVar = tk.StringVar(self)
		self.textVar.set('')
		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=["Python", "classic", "alternative", "Java"], width=25)
		self.listbox_widget = self.canvasBody.create_window(250, 110, window=self.cbbTeam)
		# IMG Salir
		self.salirPNG = Image.open("salir.png")
		self.newSalirPNG = self.salirPNG.convert(mode="RGBA")
		self.resized_image= self.newSalirPNG.resize((40,40), Image.Resampling.LANCZOS)
		self.salirTK = ImageTk.PhotoImage(self.resized_image)
		self.buttonSalir = tk.Button(self, image=self.salirTK, command=lambda: controller.show_frame("StartPage"), bg="#154e72", activebackground="#154e72", highlightthickness = 0, bd = 0)
		self.buttonSalir_widget = self.canvasHead.create_window(470, 28, window=self.buttonSalir)

	def monitor(self, thread, q):
		if thread.is_alive():
			# check the thread every 100ms
			self.after(100, lambda: self.monitor(thread, q))
			try:
				self.cola = q.get(0)
				if self.cola[0] == -1:	# battle finished
					# Widgets are available
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
				elif self.cola[0] == 0:
					self.button1.configure(text='Batalla en Curso')
			except queue.Empty:
				# None
				...
		else:
			# None
			...

	def IniciarBatalla(self):
		# Parameters
		q1 = multiprocessing.Queue()
		username = self.nameEntry.get()
		password = None
		teamname = self.textVar.get()
		botmode = "CHALLENGE_USER"
		# Widgets are blocked
		self.nameEntry.config(state='disabled')
		self.cbbTeam.config(state='disabled')
		# Start Battle
		self.button1['state'] = tk.DISABLED
		self.button1.configure(text='Esperando Confirmación')
		# Thread
		p = Thread(target=AsyncDownload, args=(q1,username,password,teamname,botmode,))
		p.start()
		# Monitor
		self.monitor(p, q1)

class Modulo2(tk.Frame):
    def __init__(self, parent, controller):
    	tk.Frame.__init__(self, parent)
    	self.controller = controller
    	# Canvas Head
    	canvasHead = tk.Canvas(self, width=500, height=50, bg="#154e72")
    	canvasHead.place(x=0, y=0, anchor = tk.CENTER)
    	canvasHead.create_text(250, 25, text="Módulo de Análisis de Batallas", fill="white", font=('Verdana 15 bold'))
    	canvasHead.pack()
    	# Canvas Body
    	self.canvasBody = tk.Canvas(self, width=500, height=259, bg="#154e72")
    	self.canvasBody.create_rectangle(0,0,310,259, outline='white', width=3)
    	self.canvasBody.pack()
    	# Botón Iniciar Batalla
    	self.button1 = tk.Button(self, text='Iniciar Batalla', anchor="center", command= self.IniciarModulo)
    	self.button1.configure(width = 16, height = 1, activebackground = "#33B5E5", relief = tk.FLAT)
    	self.button1_widget = self.canvasBody.create_window(400, 225, window=self.button1)
    	# Botón Ayuda
    	self.button2 = tk.Button(self, text='Ayuda', anchor="center", command= self.OcultarDatos)
    	self.button2.configure(width = 16, height = 1, activebackground = "#33B5E5", relief = tk.FLAT)
    	self.button2_widget = self.canvasBody.create_window(400, 185, window=self.button2)
    	# Label Movimientos
    	labelMove = tk.Label(self,text="Movimientos", bg = "#154e72", font=('Verdana 12 bold'), fg="white")
    	LabelMove_widget = self.canvasBody.create_window(80, 30-5, window=labelMove)
    	# Label Rendimiento
    	labelR = tk.Label(self,text="Rendimiento", bg = "#154e72", font=('Verdana 12 bold'), fg="white")
    	LabelR_widget = self.canvasBody.create_window(80, 165-5, window=labelR)
    	labelR1 = tk.Label(self,text="Último turno", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
    	LabelR1_widget = self.canvasBody.create_window(68, 238-5, window=labelR1)
    	labelR2 = tk.Label(self,text="Histórico", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
    	LabelR2_widget = self.canvasBody.create_window(188, 238-5, window=labelR2)
    	# Rectangulo Move
    	self.rectangle1 = self.canvasBody.create_rectangle(28,50-5,148,90-5, fill='white', outline='white')
    	self.rectangle2 = self.canvasBody.create_rectangle(160,50-5,280,90-5, fill='white', outline='white')
    	self.rectangle3 = self.canvasBody.create_rectangle(28,102-5,148,142-5, fill='white', outline='white')
    	self.rectangle4 = self.canvasBody.create_rectangle(160,102-5,280,142-5, fill='white', outline='white')
    	# Rectangulo Rendimiento
    	self.rectangle5 = self.canvasBody.create_rectangle(28,185-5,148,225-5, fill='white', outline='white')
    	self.rectangle6 = self.canvasBody.create_rectangle(160,185-5,280,225-5, fill='white', outline='white')
    	# Label Nombre de Usuario
    	self.labelUsername = tk.Label(self,text="Nombre de Usuario", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
    	self.LabelUsername_widget = self.canvasBody.create_window(400, 15, window=self.labelUsername)
    	# Label Password
    	self.labelPassword = tk.Label(self,text="Contraseña", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
    	self.LabelPassword_widget = self.canvasBody.create_window(400, 65, window=self.labelPassword)
    	# Label Equipo
    	self.labelTeam = tk.Label(self,text="Equipo", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
    	self.LabelTeam_widget = self.canvasBody.create_window(400, 115, window=self.labelTeam)
    	# Entrada Nombre de Usuario
    	self.strName = tk.StringVar()
    	self.nameEntry = ttk.Entry(self, textvariable=self.strName, width=20)
    	self.strName_widget = self.canvasBody.create_window(400, 40, window=self.nameEntry)
    	# Entrada Contraseña
    	self.strPassword = tk.StringVar()
    	self.passwordEntry = ttk.Entry(self, show="*", textvariable=self.strPassword, width=20)
    	self.strPassword_widget = self.canvasBody.create_window(400, 90, window=self.passwordEntry)
    	# ListBox Equipo
    	self.textVar = tk.StringVar(self)
    	self.textVar.set('')
    	self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=["Python", "classic", "alternative", "Java"], width=17)
    	self.listbox_widget = self.canvasBody.create_window(400, 140, window=self.cbbTeam)
    	# IMG Salir
    	self.salirPNG = Image.open("salir.png")
    	self.newSalirPNG = self.salirPNG.convert(mode="RGBA")
    	self.resized_image= self.newSalirPNG.resize((40,40), Image.Resampling.LANCZOS)
    	self.salirTK = ImageTk.PhotoImage(self.resized_image)
    	self.buttonSalir = tk.Button(self, image=self.salirTK, command=lambda: controller.show_frame("StartPage"), bg="#154e72", activebackground="#154e72", highlightthickness = 0, bd = 0)
    	self.buttonSalir_widget = canvasHead.create_window(470, 28, window=self.buttonSalir)

    def monitor(self, thread, q):
        if thread.is_alive():
            # check the thread every 100ms
            self.after(100, lambda: self.monitor(thread, q))
            try:
            	self.cola = q.get(0)
            	if self.cola[0] == 1:
            		# Nombre de los Pokémon
            		namePokemon = self.cola
            		namePokemon0 = namePokemon[1]
            		namePokemon1 = namePokemon[2]
            		namePokemon2 = namePokemon[3]
            		namePokemon3 = namePokemon[4]
            		# Actualizo los Rectangulo Move
            		self.text1 = self.canvasBody.create_text(28+60, 45+20, text=namePokemon0, fill="black", font=('Verdana 10'))
            		self.text2 = self.canvasBody.create_text(160+60, 45+20, text=namePokemon1, fill="black", font=('Verdana 10'))
            		self.text3 = self.canvasBody.create_text(28+60, 97+20, text=namePokemon2, fill="black", font=('Verdana 10'))
            		self.text4 = self.canvasBody.create_text(160+60, 97+20, text=namePokemon3, fill="black", font=('Verdana 10'))
            		self.text5 = self.canvasBody.create_text(28+60, 180+20, text="0", fill="black", font=('Verdana 10'))
            		self.text6 = self.canvasBody.create_text(160+60, 180+20, text="0", fill="black", font=('Verdana 10'))
            	elif self.cola[0] == 0:
            		self.button1.configure(text='Batalla en Curso')
            	elif self.cola[0] == 2:
            		contenidoCola = self.cola
            		text1 = contenidoCola[1]
            		color1 = contenidoCola[2]
            		text2 = contenidoCola[3]
            		color2 = contenidoCola[4]
            		text3 = contenidoCola[5]
            		color3 = contenidoCola[6]
            		text4 = contenidoCola[7]
            		color4 = contenidoCola[8]
            		text5 = contenidoCola[9]
            		text6 = contenidoCola[10]
            		# Actualizo los Rectangulo Move
            		self.canvasBody.itemconfig(self.text1, text=text1)
            		self.canvasBody.itemconfig(self.text2, text=text2)
            		self.canvasBody.itemconfig(self.text3, text=text3)
            		self.canvasBody.itemconfig(self.text4, text=text4)
            		self.canvasBody.itemconfig(self.text5, text=text5)
            		self.canvasBody.itemconfig(self.text6, text=text6)
            		self.canvasBody.itemconfig(self.rectangle1, fill=color1, outline=color1)
            		self.canvasBody.itemconfig(self.rectangle2, fill=color2, outline=color2)
            		self.canvasBody.itemconfig(self.rectangle3, fill=color3, outline=color3)
            		self.canvasBody.itemconfig(self.rectangle4, fill=color4, outline=color4)
            	elif self.cola[0] == -1:
            		self.canvasBody.itemconfig(self.text1, text='')
            		self.canvasBody.itemconfig(self.text2, text='')
            		self.canvasBody.itemconfig(self.text3, text='')
            		self.canvasBody.itemconfig(self.text4, text='')
            		self.canvasBody.itemconfig(self.text5, text='')
            		self.canvasBody.itemconfig(self.text6, text='')
            		self.button2.configure(text='Ayuda')
            		self.button1['state'] = tk.NORMAL
            		self.button1.configure(text='Iniciar Batalla')
            	else:
            		...
            except queue.Empty:
            	...
        else:
        	...

    def IniciarModulo(self):
    	# Parameters
    	q2 = multiprocessing.Queue()
    	username = self.nameEntry.get()
    	password = self.passwordEntry.get()
    	teamname = self.textVar.get()
    	botmode = "SEARCH_LADDER"
    	# Creo los Rectangulo Move
    	self.text1 = self.canvasBody.create_text(28+55, 45+20, text='', fill="black", font=('Verdana 10'))
    	self.text2 = self.canvasBody.create_text(160+55, 45+20, text='', fill="black", font=('Verdana 10'))
    	self.text3 = self.canvasBody.create_text(28+55, 97+20, text='', fill="black", font=('Verdana 10'))
    	self.text4 = self.canvasBody.create_text(160+55, 97+20, text='', fill="black", font=('Verdana 10'))
    	self.text5 = self.canvasBody.create_text(28+55, 180+20, text='', fill="black", font=('Verdana 10'))
    	self.text6 = self.canvasBody.create_text(160+55, 180+20, text='', fill="black", font=('Verdana 10'))
    	# Start Battle
    	self.button1['state'] = tk.DISABLED
    	self.button1.configure(text='Buscando Oponente')
    	p = Thread(target=AsyncDownload, args=(q2, username, password, teamname, botmode, ))
    	p.start()
    	self.monitor(p, q2)

    def OcultarDatos(self):
    	# Cambiar Nombre de Botón
    	nameButton = self.button2.config('text')[-1]
    	# Muestra u oculta los campos
    	if nameButton == 'Ayuda':
    		self.canvasBody.itemconfig(self.text1, text="???")
    		self.canvasBody.itemconfig(self.text2, text="???")
    		self.canvasBody.itemconfig(self.text3, text="???")
    		self.canvasBody.itemconfig(self.text4, text="???")
    		self.canvasBody.itemconfig(self.text5, text="???")
    		self.canvasBody.itemconfig(self.text6, text="???")
    		self.button2.configure(text='Detener Ayuda')
    	else:
    		self.canvasBody.itemconfig(self.text1, text=self.cola[1])
    		self.canvasBody.itemconfig(self.text2, text=self.cola[2])
    		self.canvasBody.itemconfig(self.text3, text=self.cola[3])
    		self.canvasBody.itemconfig(self.text4, text=self.cola[4])
    		self.canvasBody.itemconfig(self.text5, text='0')
    		self.canvasBody.itemconfig(self.text6, text='0')
    		self.button2.configure(text='Ayuda')
    	# Nuevo Nombre
    	

class Modulo3(tk.Frame):
    def __init__(self, parent, controller):
    	tk.Frame.__init__(self, parent)
    	self.controller = controller
    	# Canvas Head
    	canvasHead = tk.Canvas(self, width=500, height=50, bg="#154e72")
    	canvasHead.place(x=0, y=0, anchor = tk.CENTER)
    	canvasHead.create_text(250, 25, text="Módulo de Entrenamiento", fill="white", font=('Verdana 15 bold'))
    	canvasHead.pack()
    	# Canvas Body
    	self.canvasBody = tk.Canvas(self, width=500, height=259, bg="#154e72")
    	self.canvasBody.create_rectangle(0,0,310,259, outline='white', width=3)
    	self.canvasBody.pack()
    	# IMG Salir
    	self.salirPNG = Image.open("salir.png")
    	self.newSalirPNG = self.salirPNG.convert(mode="RGBA")
    	self.resized_image= self.newSalirPNG.resize((40,40), Image.Resampling.LANCZOS)
    	self.salirTK = ImageTk.PhotoImage(self.resized_image)
    	self.buttonSalir = tk.Button(self, image=self.salirTK, command=lambda: controller.show_frame("StartPage"), bg="#154e72", activebackground="#154e72", highlightthickness = 0, bd = 0)
    	self.buttonSalir_widget = canvasHead.create_window(470, 28, window=self.buttonSalir)
    	# Label Nombre de Usuario
    	labelUsername = tk.Label(self,text="Nombre de Usuario", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
    	LabelUsername_widget = self.canvasBody.create_window(400, 15, window=labelUsername)
    	# Label Password
    	labelPassword = tk.Label(self,text="Contraseña", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
    	LabelPassword_widget = self.canvasBody.create_window(400, 65, window=labelPassword)
    	# Entrada Nombre de Usuario
    	self.strName = tk.StringVar()
    	self.nameEntry = ttk.Entry(self, textvariable=self.strName, width=20)
    	self.strName_widget = self.canvasBody.create_window(400, 40, window=self.nameEntry)
    	# Entrada Contraseña
    	self.strPassword = tk.StringVar()
    	self.passwordEntry = ttk.Entry(self, show="*", textvariable=self.strPassword, width=20)
    	self.strPassword_widget = self.canvasBody.create_window(400, 90, window=self.passwordEntry)
    	# Botón Iniciar Entrenamiento
    	self.button1 = tk.Button(self, text='Iniciar Entrenamiento', anchor="center")
    	self.button1.configure(width = 16, height = 1, activebackground = "#33B5E5", relief = tk.FLAT)
    	self.button1_widget = self.canvasBody.create_window(400, 225, window=self.button1)
    	# Botón Establecer Equipo
    	button2 = tk.Button(self, text='Establecer Equipo', anchor="center")
    	button2.configure(width = 16, height = 1, activebackground = "#33B5E5", relief = tk.FLAT)
    	button2_widget = self.canvasBody.create_window(400, 185, window=button2)


if __name__ == "__main__":
    app = LoremIpsum()
    app.mainloop()