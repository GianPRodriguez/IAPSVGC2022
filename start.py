import multiprocessing
import math
import queue
import sys
import os 
import threading
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

		button1 = tk.Button(self, text='Módulo de Batallas', anchor="center", command=lambda: controller.show_frame("Modulo1"))
		button1.configure(width = 50, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button1_widget = canvasBody.create_window(70, 35, anchor=tk.NW, window=button1)

		button2 = tk.Button(self, text='Módulo de Análisis', anchor="center", command=lambda: controller.show_frame("Modulo2"))
		button2.configure(width = 50, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button2_widget = canvasBody.create_window(70, 105, anchor=tk.NW, window=button2)

		button3 = tk.Button(self, text='Módulo de Entrenamiento', anchor="center", command=lambda: controller.show_frame("Modulo3"))
		button3.configure(width = 50, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button3_widget = canvasBody.create_window(70, 175, anchor=tk.NW, window=button3)

class Modulo1(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.auxController = controller
		# Canvas Head
		self.canvasHead = tk.Canvas(self, width=500, height=50, bg="#283158")
		self.canvasHead.place(x=0, y=0, anchor = tk.CENTER)
		self.canvasHead.create_text(250, 25, text="Módulo de Batallas", fill="white", font=('Verdana 15 bold'))
		self.canvasHead.pack()
		# Canvas Body
		self.canvasBody = tk.Canvas(self, width=500, height=259, bg="#283158")
		self.canvasBody.pack()
		# Button Start Battle
		self.button1 = tk.Button(self, text='Iniciar Batalla', anchor="center")
		self.button1.configure(width = 23, height = 2, relief = tk.FLAT, command= self.IniciarBatalla)
		self.button1_widget = self.canvasBody.create_window(250, 180, window=self.button1)
		self.auxButton1 = self.button1
		# Label Username
		self.labelUsername = tk.Label(self,text="Nombre de Usuario", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelUsername_widget = self.canvasBody.create_window(250, 30, window=self.labelUsername)
		# Label AgentTeam
		self.labelTeam = tk.Label(self,text="Equipo del Agente", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelTeam_widget = self.canvasBody.create_window(250, 85, window=self.labelTeam)
		# Entry Username
		self.strName = tk.StringVar()
		self.nameEntry = ttk.Entry(self, textvariable=self.strName, width=28)
		self.strName_widget = self.canvasBody.create_window(250, 55, window=self.nameEntry)
		self.auxNameEntry = self.nameEntry
		# ListBox AgentTeam
		self.textVar = tk.StringVar(self)
		self.textVar.set('')
		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=["random", "classic", "alternative", "Eduardo Cunha"], width=25)
		self.listbox_widget = self.canvasBody.create_window(250, 110, window=self.cbbTeam)
		self.auxCbbTeam = self.cbbTeam
		# IMG Exit
		self.salirPNG = Image.open("salir.png")
		self.newSalirPNG = self.salirPNG.convert(mode="RGBA")
		self.resized_image= self.newSalirPNG.resize((40,40), Image.Resampling.LANCZOS)
		self.salirTK = ImageTk.PhotoImage(self.resized_image)
		self.buttonSalir = tk.Button(self, image=self.salirTK, command=lambda: controller.show_frame("StartPage"), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
		self.buttonSalir_widget = self.canvasHead.create_window(470, 28, window=self.buttonSalir)
		# Close Windows
		self.activeClose = 0
		if self.activeClose == 0:
			self.activeClose = 1
			self.auxController.protocol('WM_DELETE_WINDOW', self.closeWindows)
		else:
			self.auxController.destroy()

	def monitor(self, q):
		if self.p.is_alive():
			# check the thread every 100ms
			self.after(100, lambda: self.monitor(q))
			try:
				# Get Queue
				self.cola = q.get(0)
				if self.cola[0] == -1: 									# battle finished
					# Widgets are available
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					# Battle results
					self.isWinner = self.cola[1]
					self.nameOpponent = self.cola[2]
					self.endBattle()
				elif self.cola[0] == 0: 								# battle in progress
					self.button1.configure(text='Batalla en Curso')
				elif self.cola[0] == -2: 								# username incorrect
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					self.textIncorrect = self.canvasBody.create_text(170, 235, text="El oponente ha rechazado la solicitud de batalla", fill="orange", font=('Verdana 10'))
				elif self.cola[0] == -3: 								# username incorrect
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					self.textIncorrect = self.canvasBody.create_text(210, 235, text="¡El equipo del agente no cumple con las reglas del formato!", fill="orange", font=('Verdana 10'))
				elif self.cola[0] == -4: 								# username incorrect
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					self.textIncorrect = self.canvasBody.create_text(160, 235, text="¡El nombre de usuario no ha sido encontrado!", fill="orange", font=('Verdana 10'))
			except queue.Empty:
				# None
				...
		else:
			# None
			self.p.join()
			return

	def IniciarBatalla(self):
		if self.textVar.get() == '':
			if hasattr(self, "textIncorrect"):
				self.canvasBody.delete(self.textIncorrect)
			self.textIncorrect = self.canvasBody.create_text(120, 235, text="No se ha seleccionado un equipo", fill="orange", font=('Verdana 10'))
		else:
			if hasattr(self, "textIncorrect"):
				self.canvasBody.delete(self.textIncorrect)
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
			self.p = Thread(target=AsyncDownload, args=(q1,username,password,teamname,botmode,))
			self.p.start()
			# Monitor
			self.monitor(q1)

	def closeWindows(self):
		if self.button1.config('text')[-1] != "Batalla en Curso":
			self.exitWindows()
		else:
			self.isCloseWindows = True
			# Purple background
			self.rectangle1 = self.canvasBody.create_rectangle(0,0,500,259, fill='#4e2858')
			# Hide widgets
			self.canvasBody.delete(self.button1_widget)
			self.canvasBody.delete(self.listbox_widget)
			self.canvasBody.delete(self.LabelUsername_widget)
			self.canvasBody.delete(self.LabelTeam_widget)
			self.canvasBody.delete(self.strName_widget)
			# Emoji
			self.emoji = Image.open("close.png")
			self.emojiPNG = self.emoji.convert(mode="RGBA")
			self.emojiResized_image = self.emojiPNG.resize((80,80), Image.Resampling.LANCZOS)
			self.emojiTK = ImageTk.PhotoImage(self.emojiResized_image)
			self.emoji_widget = self.canvasBody.create_image(420,85,image=self.emojiTK)
			# Text
			self.textClose_widget1 = self.canvasBody.create_text(185, 75, text="¡La batalla está en curso!", fill="white", font=('Verdana 14'))
			self.textClose_widget2 = self.canvasBody.create_text(185, 100, text="¿Está seguro de salir?", fill="white", font=('Verdana 11'))
			# Button Sí
			self.buttonYes = tk.Button(self, text='Sí', anchor="center")
			self.buttonYes.configure(width = 23, height = 2, relief = tk.FLAT, command= self.exitWindows)
			self.buttonYes_widget = self.canvasBody.create_window(150, 180, window=self.buttonYes)
			# Button No
			self.buttonNo = tk.Button(self, text='No', anchor="center")
			self.buttonNo.configure(width = 23, height = 2, relief = tk.FLAT, command= self.screenNormalClose)
			self.buttonNo_widget = self.canvasBody.create_window(350, 180, window=self.buttonNo)
			self.auxController.protocol('WM_DELETE_WINDOW', self.exitWindows)

	def exitWindows(self):
		#self.p.join()
		os._exit(0)
		self.auxController.destroy()

	def endBattle(self):
		# Hide widgets
		self.canvasBody.delete(self.button1_widget)
		self.canvasBody.delete(self.listbox_widget)
		self.canvasBody.delete(self.LabelUsername_widget)
		self.canvasBody.delete(self.LabelTeam_widget)
		self.canvasBody.delete(self.strName_widget)
		# Emoji
		if self.isWinner:
			self.emoji = Image.open("winner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0,0,500,259, fill='#325928')
			# Text
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="¡Felicidades!", fill="white", font=('Verdana 14'))
			self.textWinner = "Has ganado la batalla"
			self.textWinner_widget2 = self.canvasBody.create_text(185, 100, text=self.textWinner, fill="white", font=('Verdana 11'))
		else:
			self.emoji = Image.open("noWinner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0,0,500,259, fill='#582831')
			# Text
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="Has perdido :(", fill="white", font=('Verdana 14'))
			self.textWinner_widget2 = self.canvasBody.create_text(185, 100, text="Suerte para la próxima", fill="white", font=('Verdana 11'))
		self.emojiPNG = self.emoji.convert(mode="RGBA")
		self.emojiResized_image = self.emojiPNG.resize((80,80), Image.Resampling.LANCZOS)
		self.emojiTK = ImageTk.PhotoImage(self.emojiResized_image)
		self.emoji_widget = self.canvasBody.create_image(420,85,image=self.emojiTK)
		# Button
		self.buttonHide = tk.Button(self, text='¡Seguir jugando!', anchor="center")
		self.buttonHide.configure(width = 23, height = 2, relief = tk.FLAT, command= self.screenNormalEnd)
		self.buttonHide_widget = self.canvasBody.create_window(250, 180, window=self.buttonHide)

	def screenNormalClose(self):
		self.canvasBody.delete(self.buttonYes_widget)
		self.canvasBody.delete(self.buttonNo_widget)
		self.canvasBody.delete(self.textClose_widget1)
		self.canvasBody.delete(self.textClose_widget2)
		self.canvasBody.delete(self.emoji_widget)
		self.canvasBody.delete(self.rectangle1)
		self.screenNormal()

	def screenNormalEnd(self):
		self.canvasBody.delete(self.buttonHide_widget)
		self.canvasBody.delete(self.textWinner_widget1)
		self.canvasBody.delete(self.textWinner_widget2)
		self.canvasBody.delete(self.emoji_widget)
		self.canvasBody.delete(self.rectangle1)
		self.screenNormal()

	def screenNormal(self):
		# Botón Iniciar Batalla
		self.button1_widget = self.canvasBody.create_window(250, 180, window=self.auxButton1)
		# Label Nombre de Usuario
		self.labelUsername = tk.Label(self,text="Nombre de Usuario", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelUsername_widget = self.canvasBody.create_window(250, 30, window=self.labelUsername)
		# Label Equipo del Agente
		self.labelTeam = tk.Label(self,text="Equipo del Agente", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelTeam_widget = self.canvasBody.create_window(250, 85, window=self.labelTeam)
		# Entrada Nombre de Usuario
		self.strName_widget = self.canvasBody.create_window(250, 55, window=self.auxNameEntry)
		# ListBox Equipo
		self.listbox_widget = self.canvasBody.create_window(250, 110, window=self.auxCbbTeam)
		# Close Windows
		self.activeClose = 0
		if self.activeClose == 0:
			self.activeClose = 1
			self.auxController.protocol('WM_DELETE_WINDOW', self.closeWindows)
		else:
			self.auxController.destroy()

class Modulo2(tk.Frame):
    def __init__(self, parent, controller):
    	tk.Frame.__init__(self, parent)
    	self.auxController = controller
    	# Canvas Head
    	self.canvasHead = tk.Canvas(self, width=500, height=50, bg="#283158")
    	self.canvasHead.place(x=0, y=0, anchor = tk.CENTER)
    	self.canvasHead.create_text(250, 25, text="Módulo de Análisis", fill="white", font=('Verdana 15 bold'))
    	self.canvasHead.pack()
    	# Canvas Body
    	self.canvasBody = tk.Canvas(self, width=500, height=259, bg="#283158")
    	self.canvasBody.create_rectangle(0,0,310,259, outline='white', width=3)
    	self.canvasBody.pack()
    	# Botón Iniciar Batalla
    	self.startBattle = tk.Button(self, text='Iniciar Batalla', anchor="center", command= self.IniciarBatalla)
    	self.startBattle.configure(width = 16, height = 1, relief = tk.FLAT)
    	self.startBattle_widget = self.canvasBody.create_window(400, 225, window=self.startBattle)
    	# Botón Ayuda
    	self.help = tk.Button(self, text='Ayuda', anchor="center", command= self.MostrarOcultarDatos)
    	self.help.configure(width = 16, height = 1, relief = tk.FLAT)
    	self.help_widget = self.canvasBody.create_window(400, 185, window=self.help)
    	self.help.config(state='disabled')
    	# Label Nombre de Usuario
    	self.labelUsername = tk.Label(self,text="Nombre de Usuario", bg = "#283158", font=('Verdana 8 bold'), fg="white")
    	self.LabelUsername_widget = self.canvasBody.create_window(400, 15, window=self.labelUsername)
    	# Label Password
    	self.labelPassword = tk.Label(self,text="Contraseña", bg = "#283158", font=('Verdana 8 bold'), fg="white")
    	self.LabelPassword_widget = self.canvasBody.create_window(400, 65, window=self.labelPassword)
    	# Label Equipo
    	self.labelTeam = tk.Label(self,text="Equipo", bg = "#283158", font=('Verdana 8 bold'), fg="white")
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
    	self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=["random", "classic", "alternative", "Eduardo Cunha"], width=17)
    	self.listbox_widget = self.canvasBody.create_window(400, 140, window=self.cbbTeam)
    	# IMG Exit
    	self.exitPNG = Image.open("IMGs/exit.png")
    	self.newExitPNG = self.exitPNG.convert(mode="RGBA")
    	self.exitResized = self.newExitPNG.resize((40,40), Image.Resampling.LANCZOS)
    	self.exitTK = ImageTk.PhotoImage(self.exitResized)
    	self.buttonExit = tk.Button(self, image=self.exitTK, command=lambda: controller.show_frame("StartPage"), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
    	self.buttonExit_widget = self.canvasHead.create_window(470, 28, window=self.buttonExit)
    	# IMG Bar Chart
    	self.barChartPNG = Image.open("IMGs/barChart.png")
    	self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
    	self.barChartResized = self.newBarChartPNG.resize((200,200), Image.Resampling.LANCZOS)
    	self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
    	self.barChart_widget = self.canvasBody.create_image(160,125,image=self.barChartTK)
    	# IMG Info
    	self.infoPNG = Image.open("IMGs/info.png")
    	self.newInfoPNG = self.infoPNG.convert(mode="RGBA")
    	self.infoResized = self.newInfoPNG.resize((38,38), Image.Resampling.LANCZOS)
    	self.infoTK = ImageTk.PhotoImage(self.infoResized)
    	self.buttonInfo = tk.Button(self, image=self.infoTK, command=lambda: os.system('"PDFs/Módulo de Análisis.pdf"'), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
    	self.buttonInfo_widget = self.canvasHead.create_window(28, 28, window=self.buttonInfo)

    def IniciarBatalla(self):
    	if self.textVar.get() == '':
    		if hasattr(self, "textIncorrect"):
    			self.canvasBody.delete(self.textIncorrect)
    		self.textIncorrect = self.canvasBody.create_text(120, 235, text="No se ha seleccionado un equipo", fill="red", font=('Verdana 10'))
    	else:
    		if hasattr(self, "textIncorrect"):
    			self.canvasBody.delete(self.textIncorrect)
    		# Parameters
	    	self.q = multiprocessing.Queue()
	    	self.username = self.nameEntry.get()
	    	self.password = self.passwordEntry.get()
	    	self.teamname = self.textVar.get()
	    	botmode = "SEARCH_LADDER"
	    	# Widgets are blocked
	    	self.nameEntry.config(state='disabled')
	    	self.passwordEntry.config(state='disabled')
	    	self.cbbTeam.config(state='disabled')
	    	# Start Battle
	    	self.startBattle['state'] = tk.DISABLED
	    	self.startBattle.configure(text='Buscando Oponente')
	    	self.p = Thread(target=AsyncDownload, args=(self.q, self.username, self.password, self.teamname, botmode, ))
	    	self.p.start()
	    	self.isBattle = False
	    	self.monitor()

    def monitor(self):
        if self.p.is_alive():
            # check the thread every 100ms
            self.after(100, lambda: self.monitor())
            try:
            	self.cola = self.q.get(0)
            	if self.cola[0] == 1:
            		self.namePokemon = self.cola
            		# Team Preview
            		self.text1 = self.namePokemon[1]
            		self.text2 = self.namePokemon[2]
            		self.text3 = self.namePokemon[3]
            		self.text4 = self.namePokemon[4]
            		# Help is enabled
            		self.help['state'] = tk.NORMAL
            		# Update 4 Pkmn
            		if self.help.config('text')[-1] == 'Ayuda':
            			self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text="???", fill="black", font=('Verdana 10'))
            			self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text="???", fill="black", font=('Verdana 10'))
            			self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text="???", fill="black", font=('Verdana 10'))
            			self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text="???", fill="black", font=('Verdana 10'))
            		else:
            			self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text=self.text1, fill="black", font=('Verdana 10'))
            			self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text=self.text2, fill="black", font=('Verdana 10'))
            			self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text=self.text3, fill="black", font=('Verdana 10'))
            			self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text=self.text4, fill="black", font=('Verdana 10'))
            		self.text5_widget = self.canvasBody.create_text(28+55, 180+20, text="", fill="black", font=('Verdana 10'))
            		self.text6_widget = self.canvasBody.create_text(160+55, 180+20, text="", fill="black", font=('Verdana 10'))
            	elif self.cola[0] == 0:
            		# New Battle
            		self.startBattle.configure(text='Batalla en Curso')
            		# Bye IMG Bar Chart
            		self.canvasBody.delete(self.barChart_widget)
            		# Label Team Preview
            		self.labelMove = tk.Label(self,text="Vista Previa del Equipo", bg = "#283158", font=('Verdana 12 bold'), fg="white")
            		self.LabelMove_widget = self.canvasBody.create_window(130, 30-5, window=self.labelMove)
            		# Team Preview Rectangles
            		self.color1 = '#ffffff'
            		self.color2 = '#ffffff'
            		self.color3 = '#ffffff'
            		self.color4 = '#ffffff'
            		self.color5 = '#283158'
            		self.color6 = '#283158'
            		self.color1_widget = self.canvasBody.create_rectangle(28,50-5,148,90-5, fill=self.color1, outline=self.color1)
            		self.color2_widget = self.canvasBody.create_rectangle(160,50-5,280,90-5, fill=self.color2, outline=self.color2)
            		self.color3_widget = self.canvasBody.create_rectangle(28,102-5,148,142-5, fill=self.color3, outline=self.color3)
            		self.color4_widget = self.canvasBody.create_rectangle(160,102-5,280,142-5, fill=self.color4, outline=self.color4)
            		self.color5_widget = self.canvasBody.create_rectangle(28,185-5,148,225-5, fill=self.color5, outline=self.color5)
            		self.color6_widget = self.canvasBody.create_rectangle(160,185-5,280,225-5, fill=self.color6, outline=self.color6)
            		# GIF Loading
            		...
            	elif self.cola[0] == 2:
            		self.isBattle = True
            		self.canvasBody.delete(self.LabelMove_widget)
            		self.labelMove = tk.Label(self,text="Movimientos", bg = "#283158", font=('Verdana 12 bold'), fg="white")
            		self.LabelMove_widget = self.canvasBody.create_window(80, 30-5, window=self.labelMove)
            		# New Turn
            		self.text1 = self.cola[1]
            		self.color1 = self.cola[2]
            		self.text2 = self.cola[3]
            		self.color2 = self.cola[4]
            		self.text3 = self.cola[5]
            		self.color3 = self.cola[6]
            		self.text4 = self.cola[7]
            		self.color4 = self.cola[8]
            		self.text5 = self.cola[9]
            		self.color5 = self.cola[10]
            		self.text6 = self.cola[11]
            		self.color6 = self.cola[12]
            		# Label Rendimiento
            		self.labelR = tk.Label(self,text="Rendimiento", bg = "#283158", font=('Verdana 12 bold'), fg="white")
            		self.LabelR_widget = self.canvasBody.create_window(80, 165-5, window=self.labelR)
            		self.labelR1 = tk.Label(self,text="Último turno", bg = "#283158", font=('Verdana 8 bold'), fg="white")
            		self.LabelR1_widget = self.canvasBody.create_window(68, 238-5, window=self.labelR1)
            		self.labelR2 = tk.Label(self,text="Histórico", bg = "#283158", font=('Verdana 8 bold'), fg="white")
            		self.LabelR2_widget = self.canvasBody.create_window(188, 238-5, window=self.labelR2)
            		# Update text & color
            		self.text1_widget = self.canvasBody.itemconfig(self.text1_widget, text=self.text1)
            		self.text2_widget = self.canvasBody.itemconfig(self.text2_widget, text=self.text2)
            		self.text3_widget = self.canvasBody.itemconfig(self.text3_widget, text=self.text3)
            		self.text4_widget = self.canvasBody.itemconfig(self.text4_widget, text=self.text4)
            		self.text5_widget = self.canvasBody.itemconfig(self.text5_widget, text=math.floor(self.text5))
            		self.text6_widget = self.canvasBody.itemconfig(self.text6_widget, text=math.floor(self.text6))
            		self.color1_widget = self.canvasBody.itemconfig(self.color1_widget, fill=self.color1, outline=self.color1)
            		self.color2_widget = self.canvasBody.itemconfig(self.color2_widget, fill=self.color2, outline=self.color2)
            		self.color3_widget = self.canvasBody.itemconfig(self.color3_widget, fill=self.color3, outline=self.color3)
            		self.color4_widget = self.canvasBody.itemconfig(self.color4_widget, fill=self.color4, outline=self.color4)
            		self.color5_widget = self.canvasBody.itemconfig(self.color5_widget, fill=self.color5, outline=self.color5)
            		self.color6_widget = self.canvasBody.itemconfig(self.color6_widget, fill=self.color6, outline=self.color6)
            	elif self.cola[0] == -1:
            		self.isBattle = False
            		# End Game
            		self.canvasBody.delete(self.text1_widget)
            		self.canvasBody.delete(self.text2_widget)
            		self.canvasBody.delete(self.text3_widget)
            		self.canvasBody.delete(self.text4_widget)
            		self.canvasBody.delete(self.text5_widget)
            		self.canvasBody.delete(self.text6_widget)
            		self.canvasBody.delete(self.color1_widget)
            		self.canvasBody.delete(self.color2_widget)
            		self.canvasBody.delete(self.color3_widget)
            		self.canvasBody.delete(self.color4_widget)
            		self.canvasBody.delete(self.color5_widget)
            		self.canvasBody.delete(self.color6_widget)
            		self.canvasBody.delete(self.LabelR_widget)
            		self.canvasBody.delete(self.LabelR1_widget)
            		self.canvasBody.delete(self.LabelR2_widget)
            		self.canvasBody.delete(self.labelMove)
            		# Buttons
            		self.help.configure(text='Ayuda')
            		self.startBattle['state'] = tk.NORMAL
            		self.startBattle.configure(text='Iniciar Batalla')
            		# IMG Bar Chart
            		self.barChartPNG = Image.open("IMGs/barChart.png")
            		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
            		self.barChartResized = self.newBarChartPNG.resize((200,200), Image.Resampling.LANCZOS)
            		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
            		self.barChart_widget = self.canvasBody.create_image(160,125,image=self.barChartTK)
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
            		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=["random", "classic", "alternative", "Eduardo Cunha"], width=17)
            		self.listbox_widget = self.canvasBody.create_window(400, 140, window=self.cbbTeam)
            	elif self.cola[0] == -3:
            		# Username or Password Incorrect
            		self.textIncorrect = self.canvasBody.create_text(120, 235, text="Usuario o contraseña incorrecta", fill="red", font=('Verdana 10'))
            		self.startBattle['state'] = tk.NORMAL
            		self.startBattle.configure(text='Iniciar Batalla')
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
            		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=["random", "classic", "alternative", "Eduardo Cunha"], width=17)
            		self.listbox_widget = self.canvasBody.create_window(400, 140, window=self.cbbTeam)
            	else:
            		...
            except queue.Empty:
            	...
        else:
        	...

    def MostrarOcultarDatos(self):
    	if self.help.config('text')[-1] == 'Ayuda':
    		self.canvasBody.itemconfig(self.text1_widget, text=self.text1)
    		self.canvasBody.itemconfig(self.text2_widget, text=self.text2)
    		self.canvasBody.itemconfig(self.text3_widget, text=self.text3)
    		self.canvasBody.itemconfig(self.text4_widget, text=self.text4)
    		self.canvasBody.itemconfig(self.color1_widget, fill=self.color1, outline=self.color1)
    		self.canvasBody.itemconfig(self.color2_widget, fill=self.color2, outline=self.color2)
    		self.canvasBody.itemconfig(self.color3_widget, fill=self.color3, outline=self.color3)
    		self.canvasBody.itemconfig(self.color4_widget, fill=self.color4, outline=self.color4)
    		if self.isBattle:
    			self.canvasBody.itemconfig(self.text5_widget, text=self.text5)
    			self.canvasBody.itemconfig(self.text6_widget, text=self.text6)
	    		self.canvasBody.itemconfig(self.color5_widget, fill=self.color5, outline=self.color5)
	    		self.canvasBody.itemconfig(self.color6_widget, fill=self.color6, outline=self.color6)
    		self.help.configure(text='Detener Ayuda')
    	else:
    		self.canvasBody.itemconfig(self.text1_widget, text="???")
    		self.canvasBody.itemconfig(self.text2_widget, text="???")
    		self.canvasBody.itemconfig(self.text3_widget, text="???")
    		self.canvasBody.itemconfig(self.text4_widget, text="???")
    		self.canvasBody.itemconfig(self.color1_widget, fill="#ffffff", outline="#ffffff")
    		self.canvasBody.itemconfig(self.color2_widget, fill="#ffffff", outline="#ffffff")
    		self.canvasBody.itemconfig(self.color3_widget, fill="#ffffff", outline="#ffffff")
    		self.canvasBody.itemconfig(self.color4_widget, fill="#ffffff", outline="#ffffff")
    		if self.isBattle:
    			self.canvasBody.itemconfig(self.text5_widget, text="???")
    			self.canvasBody.itemconfig(self.text6_widget, text="???")
	    		self.canvasBody.itemconfig(self.color5_widget, fill="#ffffff", outline="#ffffff")
	    		self.canvasBody.itemconfig(self.color6_widget, fill="#ffffff", outline="#ffffff")
    		self.help.configure(text='Ayuda')

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