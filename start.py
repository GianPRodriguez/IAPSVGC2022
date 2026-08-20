import multiprocessing
import time
import math
import queue
import sys
import os 
import threading
from threading import Thread
import tkinter as tk
from tkinter import ttk
from run import *
from API.PokeAPI import getPkmnImg
import subprocess
import asyncio
from PIL import ImageTk, Image
import requests
from io import BytesIO
from teams.team_converter import export_to_packed
from teams import load_team
from neuroevolution.start2 import inicio

async def stop(q, teamText):
	poller = asyncio.ensure_future(ValidarEquipo(q, teamText))
	while True:
		await asyncio.sleep(0.1)
		try:
			cola = q.get(0)
			if cola[0] == -1:
				break
		except queue.Empty:
			...
	poller.cancel()
	return cola[2], cola[1]

#########################################################################################
# OK - 
#########################################################################################
async def ValidarEquipo(q, teamText):
	team, listPkmn = export_to_packed(teamText)
	if team is None:
		q.put([-3, listPkmn, False])
		return
	ps_websocket_client = await PSWebsocketClient.create('', '', 'sim.smogon.com:8000')
	await ps_websocket_client.update_team(team)
	msg = await ps_websocket_client.receive_message()
	if msg.startswith("|popup|Your team"):
		print("[ERROR]    ¡El equipo del agente no cumple con las reglas del formato!")
		q.put([-3, listPkmn, False])
		return
	else:
		q.put([-3, listPkmn, True])
		return

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
		canvasHead = tk.Canvas(self, width=500, height=50, bg="#283158")
		canvasHead.place(x=0, y=0, anchor = tk.CENTER)
		canvasHead.create_text(250, 25, text="Lorem ipsum dolor sit amet", fill="white", font=('Verdana 15 bold'))
		canvasHead.pack()
		# Canvas Body
		canvasBody = tk.Canvas(self, width=500, height=259, bg="#283158")
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
		self.path = 'teams/teams/gen8/vgc2022'
		self.listBoxValues = os.listdir(self.path)
		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=self.listBoxValues, width=25)
		self.listbox_widget = self.canvasBody.create_window(250, 110, window=self.cbbTeam)
		self.auxCbbTeam = self.cbbTeam
		# IMG Exit
		self.salirPNG = Image.open("IMGs/exit.png")
		self.newSalirPNG = self.salirPNG.convert(mode="RGBA")
		self.resized_image= self.newSalirPNG.resize((40,40), Image.LANCZOS)
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
				if self.cola[0] == 0:
					self.button1.configure(text='Batalla en Curso')
				elif self.cola[0] == -1:
					# Widgets are available
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					# Battle results
					self.isWinner = self.cola[1]
					self.nameOpponent = self.cola[2]
					self.endBattle()
				elif self.cola[0] == -2:
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					self.textIncorrect = self.canvasBody.create_text(170, 235, text="¡El oponente ha rechazado la solicitud de batalla!", fill="orange", font=('Verdana 10'))
				elif self.cola[0] == -3:
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					self.textIncorrect = self.canvasBody.create_text(210, 235, text="¡El equipo del agente no cumple con las reglas del formato!", fill="orange", font=('Verdana 10'))
				elif self.cola[0] == -4:
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					self.textIncorrect = self.canvasBody.create_text(160, 235, text="¡El nombre de usuario no ha sido encontrado!", fill="orange", font=('Verdana 10'))
				elif self.cola[0] == -5:
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.button1['state'] = tk.NORMAL
					self.button1.configure(text='Iniciar Batalla')
					self.textIncorrect = self.canvasBody.create_text(160, 235, text="¡Usuario o contraseña incorrecta!", fill="orange", font=('Verdana 10'))
			except queue.Empty:
				# None
				...
		else:
			# None
			self.p.join()
			return

#################################################################################################################
# MAIN
#################################################################################################################
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
			# Thread 1
			self.p = Thread(target=AsyncDownload, args=(q1, username, password, teamname, botmode, 1,))
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
			self.emoji = Image.open("IMGs/close.png")
			self.emojiPNG = self.emoji.convert(mode="RGBA")
			self.emojiResized_image = self.emojiPNG.resize((80,80), Image.LANCZOS)
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
			self.emoji = Image.open("IMGs/winner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0,0,500,259, fill='#325928')
			# Text
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="¡Felicidades!", fill="white", font=('Verdana 14'))
			self.textWinner = "Has ganado la batalla"
			self.textWinner_widget2 = self.canvasBody.create_text(185, 100, text=self.textWinner, fill="white", font=('Verdana 11'))
		else:
			self.emoji = Image.open("IMGs/noWinner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0,0,500,259, fill='#582831')
			# Text
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="Has perdido :(", fill="white", font=('Verdana 14'))
			self.textWinner_widget2 = self.canvasBody.create_text(185, 100, text="Suerte para la próxima", fill="white", font=('Verdana 11'))
		self.emojiPNG = self.emoji.convert(mode="RGBA")
		self.emojiResized_image = self.emojiPNG.resize((80,80), Image.LANCZOS)
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
		self.auxStartBattle = self.startBattle
		# Botón Ayuda
		self.help = tk.Button(self, text='Ayuda', anchor="center", command= self.MostrarOcultarDatos)
		self.help.configure(width = 16, height = 1, relief = tk.FLAT)
		self.help_widget = self.canvasBody.create_window(400, 185, window=self.help)
		self.help.config(state='disabled')
		self.auxHelp = self.help
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
		self.auxNameEntry = self.nameEntry
		# Entrada Contraseña
		self.strPassword = tk.StringVar()
		self.passwordEntry = ttk.Entry(self, show="*", textvariable=self.strPassword, width=20)
		self.strPassword_widget = self.canvasBody.create_window(400, 90, window=self.passwordEntry)
		self.auxPasswordEntry = self.passwordEntry
		# ListBox Equipo
		self.textVar = tk.StringVar(self)
		self.textVar.set('')
		self.path = 'teams/teams/gen8/vgc2022'
		self.listBoxValues = os.listdir(self.path)
		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=self.listBoxValues, width=17)
		self.listbox_widget = self.canvasBody.create_window(400, 140, window=self.cbbTeam)
		self.auxCbbTeam = self.cbbTeam
		# IMG Exit
		self.exitPNG = Image.open("IMGs/exit.png")
		self.newExitPNG = self.exitPNG.convert(mode="RGBA")
		self.exitResized = self.newExitPNG.resize((40,40), Image.LANCZOS)
		self.exitTK = ImageTk.PhotoImage(self.exitResized)
		self.buttonExit = tk.Button(self, image=self.exitTK, command=lambda: self.auxController.show_frame("StartPage"), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
		self.buttonExit_widget = self.canvasHead.create_window(470, 28, window=self.buttonExit)
		# IMG Bar Chart
		self.barChartPNG = Image.open("IMGs/barChart.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((200,200), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(160,125,image=self.barChartTK)
		# IMG Info
		self.infoPNG = Image.open("IMGs/info.png")
		self.newInfoPNG = self.infoPNG.convert(mode="RGBA")
		self.infoResized = self.newInfoPNG.resize((38,38), Image.LANCZOS)
		self.infoTK = ImageTk.PhotoImage(self.infoResized)
		self.buttonInfo = tk.Button(self, image=self.infoTK, command=lambda: os.system('"PDFs/Módulo de Análisis.pdf"'), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
		self.buttonInfo_widget = self.canvasHead.create_window(28, 28, window=self.buttonInfo)
		# Close Windows
		self.activeClose = 0
		if self.activeClose == 0:
			self.activeClose = 1
			self.auxController.protocol('WM_DELETE_WINDOW', self.closeWindows)
		else:
			self.auxController.destroy()

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
			###########################################
			# Start Battle
			###########################################
			self.startBattle['state'] = tk.DISABLED
			self.startBattle.configure(text='Buscando Oponente')
			self.p = Thread(target=AsyncDownload, args=(self.q, self.username, self.password, self.teamname, botmode, 1,))
			self.p.start()
			self.isBattle = False
			self.monitor()

	def monitor(self):
		if self.p.is_alive():
            # check the thread every 100ms
			self.after(100, lambda: self.monitor())
			try:
				self.cola = self.q.get(0)
				if self.cola[0] == 2:
					#####################################################################################################
					# (2): Fase: Battle Turn - Se envía el identificador a la cola desde el Agente (main.py)
					#####################################################################################################
					self.isBattle = True
					self.labelMove.configure(text="Movimientos", bg = "#283158", font=('Verdana 12 bold'), fg="white")
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
					# Create Rectangle
					if self.help.config('text')[-1] == 'Ayuda':
						self.color1_widget = self.canvasBody.create_rectangle(28,50-5,148,90-5, fill="#ffffff", outline="#ffffff")
						self.color2_widget = self.canvasBody.create_rectangle(160,50-5,280,90-5, fill="#ffffff", outline="#ffffff")
						self.color3_widget = self.canvasBody.create_rectangle(28,102-5,148,142-5, fill="#ffffff", outline="#ffffff")
						self.color4_widget = self.canvasBody.create_rectangle(160,102-5,280,142-5, fill="#ffffff", outline="#ffffff")
						self.color5_widget = self.canvasBody.create_rectangle(28,185-5,148,225-5, fill="#ffffff", outline="#ffffff")
						self.color6_widget = self.canvasBody.create_rectangle(160,185-5,280,225-5, fill="#ffffff", outline="#ffffff")
						self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text="???", fill="black", font=('Verdana 10'))
						self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text="???", fill="black", font=('Verdana 10'))
						self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text="???", fill="black", font=('Verdana 10'))
						self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text="???", fill="black", font=('Verdana 10'))
						self.text5_widget = self.canvasBody.create_text(28+55, 180+20, text="???", fill="black", font=('Verdana 10'))
						self.text6_widget = self.canvasBody.create_text(160+55, 180+20, text="???", fill="black", font=('Verdana 10'))
					else:
						self.color1_widget = self.canvasBody.create_rectangle(28,50-5,148,90-5, fill=self.color1, outline=self.color1)
						self.color2_widget = self.canvasBody.create_rectangle(160,50-5,280,90-5, fill=self.color2, outline=self.color2)
						self.color3_widget = self.canvasBody.create_rectangle(28,102-5,148,142-5, fill=self.color3, outline=self.color3)
						self.color4_widget = self.canvasBody.create_rectangle(160,102-5,280,142-5, fill=self.color4, outline=self.color4)
						self.color5_widget = self.canvasBody.create_rectangle(28,185-5,148,225-5, fill=self.color5, outline=self.color5)
						self.color6_widget = self.canvasBody.create_rectangle(160,185-5,280,225-5, fill=self.color6, outline=self.color6)
						self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text=self.text1, fill="black", font=('Verdana 10'))
						self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text=self.text2, fill="black", font=('Verdana 10'))
						self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text=self.text3, fill="black", font=('Verdana 8'))
						self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text=self.text4, fill="black", font=('Verdana 8'))
						self.text5_widget = self.canvasBody.create_text(28+55, 180+20, text=math.floor(self.text5), fill="white", font=('Verdana 12'))
						self.text6_widget = self.canvasBody.create_text(160+55, 180+20, text=math.floor(self.text6), fill="white", font=('Verdana 12'))
				elif self.cola[0] == 1:
					#####################################################################################################
					# (1): Fase: Team Preview - Se envía el identificador a la cola desde el Agente (main.py)
					#####################################################################################################
					self.namePokemon = self.cola
					# Team Preview
					self.text1 = self.namePokemon[1]
					self.text2 = self.namePokemon[2]
					self.text3 = self.namePokemon[3]
					self.text4 = self.namePokemon[4]
					self.text5 = ""
					self.text6 = ""
					# Help is enabled
					self.help['state'] = tk.NORMAL
					self.auxHelp = self.help
					# Update 4 Pkmn
					if self.help.config('text')[-1] == 'Ayuda':
						self.canvasBody.itemconfig(self.text1_widget, text="???", font=('Verdana 10'))
						self.canvasBody.itemconfig(self.text2_widget, text="???", font=('Verdana 10'))
						self.canvasBody.itemconfig(self.text3_widget, text="???", font=('Verdana 10'))
						self.canvasBody.itemconfig(self.text4_widget, text="???", font=('Verdana 10'))
					else:
						self.canvasBody.itemconfig(self.text1_widget, text=self.text1, font=('Verdana 8'))
						self.canvasBody.itemconfig(self.text2_widget, text=self.text2, font=('Verdana 8'))
						self.canvasBody.itemconfig(self.text3_widget, text=self.text3, font=('Verdana 8'))
						self.canvasBody.itemconfig(self.text4_widget, text=self.text4, font=('Verdana 8'))
					self.text5_widget = self.canvasBody.create_text(28+55, 180+20, text=self.text5, fill="white", font=('Verdana 12'))
					self.text6_widget = self.canvasBody.create_text(160+55, 180+20, text=self.text6, fill="white", font=('Verdana 12'))
				elif self.cola[0] == 0:
					#####################################################################################################
					# (0): Fase: New Battle - Se envía el identificador a la cola desde la clase (run_battle.py)
					#####################################################################################################
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
					# Cargando...
					self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text="Calculando...", fill="black", font=('Verdana 10'))
					self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text="Calculando...", fill="black", font=('Verdana 10'))
					self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text="Calculando...", fill="black", font=('Verdana 10'))
					self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text="Calculando...", fill="black", font=('Verdana 10'))
				elif self.cola[0] == -1:
					#####################################################################################################
					# (-1): Fase: End Battle OK
					#####################################################################################################
					# Widgets are available
					self.nameEntry.config(state='normal')
					self.passwordEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.startBattle['state'] = tk.NORMAL
					self.isBattle = False
					self.activeClose = 0
					# Battle results
					self.isWinner = self.cola[1]
					self.nameOpponent = self.cola[2]
					self.endBattle()
				elif self.cola[0] == -3:
					#####################################################################################################
					# (-3): Fase: Invalid Team
					#####################################################################################################
					self.nameEntry.config(state='normal')
					self.cbbTeam.config(state='normal')
					self.startBattle['state'] = tk.NORMAL
					self.startBattle.configure(text='Iniciar Batalla')
					self.textIncorrect = self.canvasBody.create_text(155, 235, text="¡El equipo no cumple con las reglas del formato!", fill="orange", font=('Verdana 9'))
				elif self.cola[0] == -5:
					#####################################################################################################
					# (-5): Fase: Username or Password Incorrect
					#####################################################################################################
					# Username or Password Incorrect
					self.textIncorrect = self.canvasBody.create_text(120, 235, text="¡Usuario o contraseña incorrecta!", fill="orange", font=('Verdana 10'))
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
					self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values = self.listBoxValues, width=17)
					self.listbox_widget = self.canvasBody.create_window(400, 140, window=self.cbbTeam)
				elif self.cola[0] == -6:
					#####################################################################################################
					# (-6): Fase: System Error
					#####################################################################################################
					# SYSTEM ERROR
					self.textIncorrect = self.canvasBody.create_text(120, 235, text="¡Error del sistema!", fill="orange", font=('Verdana 10'))
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
					self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values = self.listBoxValues, width=17)
					self.listbox_widget = self.canvasBody.create_window(400, 140, window=self.cbbTeam)
			except queue.Empty:
				...
		else:
			...

	def MostrarOcultarDatos(self):
		if self.help.config('text')[-1] == 'Ayuda':
			self.canvasBody.itemconfig(self.text1_widget, text=self.text1)
			self.canvasBody.itemconfig(self.text2_widget, text=self.text2)
			self.canvasBody.itemconfig(self.text3_widget, text=self.text3, fill="black", font=('Verdana 8'))
			self.canvasBody.itemconfig(self.text4_widget, text=self.text4, fill="black", font=('Verdana 8'))
			self.canvasBody.itemconfig(self.color1_widget, fill=self.color1, outline=self.color1)
			self.canvasBody.itemconfig(self.color2_widget, fill=self.color2, outline=self.color2)
			self.canvasBody.itemconfig(self.color3_widget, fill=self.color3, outline=self.color3)
			self.canvasBody.itemconfig(self.color4_widget, fill=self.color4, outline=self.color4)
			if self.isBattle:
				self.canvasBody.itemconfig(self.text5_widget, text=self.text5, fill="white", font=('Verdana 12'))
				self.canvasBody.itemconfig(self.text6_widget, text=self.text6, fill="white", font=('Verdana 12'))
				self.canvasBody.itemconfig(self.color5_widget, fill=self.color5, outline=self.color5)
				self.canvasBody.itemconfig(self.color6_widget, fill=self.color6, outline=self.color6)
			self.help.configure(text='Detener Ayuda')
			self.auxHelp = self.help
		else:
			self.canvasBody.itemconfig(self.text1_widget, text="???", font=('Verdana 10'))
			self.canvasBody.itemconfig(self.text2_widget, text="???", font=('Verdana 10'))
			self.canvasBody.itemconfig(self.text3_widget, text="???", font=('Verdana 10'))
			self.canvasBody.itemconfig(self.text4_widget, text="???", font=('Verdana 10'))
			self.canvasBody.itemconfig(self.color1_widget, fill="#ffffff", outline="#ffffff")
			self.canvasBody.itemconfig(self.color2_widget, fill="#ffffff", outline="#ffffff")
			self.canvasBody.itemconfig(self.color3_widget, fill="#ffffff", outline="#ffffff")
			self.canvasBody.itemconfig(self.color4_widget, fill="#ffffff", outline="#ffffff")
			if self.isBattle:
				self.canvasBody.itemconfig(self.text5_widget, text="???", fill="black", font=('Verdana 10'))
				self.canvasBody.itemconfig(self.text6_widget, text="???", fill="black", font=('Verdana 10'))
				self.canvasBody.itemconfig(self.color5_widget, fill="#ffffff", outline="#ffffff")
				self.canvasBody.itemconfig(self.color6_widget, fill="#ffffff", outline="#ffffff")
			self.help.configure(text='Ayuda')
			self.auxHelp = self.help

	def closeWindows(self):
		if self.startBattle.config('text')[-1] != "Batalla en Curso":
			self.exitWindows()
		else:
			# Al cerrar la ventana...
			self.isCloseWindows = True
			# Clean Canvas
			self.canvasBody.delete("all")
			# Purple background
			self.rectangle1 = self.canvasBody.create_rectangle(0,0,500,259, fill='#4e2858')
			# Emoji
			self.emoji = Image.open("IMGs/close.png")
			self.emojiPNG = self.emoji.convert(mode="RGBA")
			self.emojiResized_image = self.emojiPNG.resize((80,80), Image.LANCZOS)
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

#################################################################################################################################
# Acciones a realizar al finalizar la batalla, NO CONSIDERA ERROR DEL SISTEMA
#################################################################################################################################
	def endBattle(self):
		# Clean Canvas
		self.canvasBody.delete("all")
		# Emoji
		if self.isWinner:
			self.emoji = Image.open("IMGs/winner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0,0,500,259, fill='#325928')
			# Text
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="¡Felicidades!", fill="white", font=('Verdana 14'))
			self.textWinner = "Has ganado la batalla contra " + self.nameOpponent
			self.textWinner_widget2 = self.canvasBody.create_text(185, 100, text=self.textWinner, fill="white", font=('Verdana 11'))
		else:
			self.emoji = Image.open("IMGs/noWinner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0,0,500,259, fill='#582831')
			# Text
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="Has perdido :(", fill="white", font=('Verdana 14'))
			self.textWinner_widget2 = self.canvasBody.create_text(185, 100, text="Suerte para la próxima", fill="white", font=('Verdana 11'))
		
		self.emojiPNG = self.emoji.convert(mode="RGBA")
		self.emojiResized_image = self.emojiPNG.resize((80,80), Image.LANCZOS)
		self.emojiTK = ImageTk.PhotoImage(self.emojiResized_image)
		self.emoji_widget = self.canvasBody.create_image(420,85,image=self.emojiTK)
		# Button
		self.buttonHide = tk.Button(self, text='¡Seguir jugando!', anchor="center")
		self.buttonHide.configure(width = 23, height = 2, relief = tk.FLAT, command= self.screenNormalEnd)
		self.buttonHide_widget = self.canvasBody.create_window(250, 180, window=self.buttonHide)

	def screenNormalClose(self):
		# Deseo continuar la partida, regresa totalmente a su estado anterior
		self.canvasBody.delete(self.buttonYes_widget)
		self.canvasBody.delete(self.buttonNo_widget)
		self.canvasBody.delete(self.textClose_widget1)
		self.canvasBody.delete(self.textClose_widget2)
		self.canvasBody.delete(self.emoji_widget)
		self.canvasBody.delete(self.rectangle1)
		# Label Turn
		if self.isBattle:
			self.labelMove = tk.Label(self,text="Movimientos", bg = "#283158", font=('Verdana 12 bold'), fg="white")
			self.LabelMove_widget = self.canvasBody.create_window(80, 30-5, window=self.labelMove)
			self.labelR = tk.Label(self,text="Rendimiento", bg = "#283158", font=('Verdana 12 bold'), fg="white")
			self.LabelR_widget = self.canvasBody.create_window(80, 165-5, window=self.labelR)
			self.labelR1 = tk.Label(self,text="Último turno", bg = "#283158", font=('Verdana 8 bold'), fg="white")
			self.LabelR1_widget = self.canvasBody.create_window(68, 238-5, window=self.labelR1)
			self.labelR2 = tk.Label(self,text="Histórico", bg = "#283158", font=('Verdana 8 bold'), fg="white")
			self.LabelR2_widget = self.canvasBody.create_window(188, 238-5, window=self.labelR2)
		else:
			self.labelMove = tk.Label(self,text="Vista Previa del Equipo", bg = "#283158", font=('Verdana 12 bold'), fg="white")
			self.LabelMove_widget = self.canvasBody.create_window(130, 30-5, window=self.labelMove)
		# Rectangles
		if self.isBattle:
			if self.help.config('text')[-1] == 'Ayuda':
				self.color1_widget = self.canvasBody.create_rectangle(28,50-5,148,90-5, fill="#ffffff", outline="#ffffff")
				self.color2_widget = self.canvasBody.create_rectangle(160,50-5,280,90-5, fill="#ffffff", outline="#ffffff")
				self.color3_widget = self.canvasBody.create_rectangle(28,102-5,148,142-5, fill="#ffffff", outline="#ffffff")
				self.color4_widget = self.canvasBody.create_rectangle(160,102-5,280,142-5, fill="#ffffff", outline="#ffffff")
				self.color5_widget = self.canvasBody.create_rectangle(28,185-5,148,225-5, fill="#ffffff", outline="#ffffff")
				self.color6_widget = self.canvasBody.create_rectangle(160,185-5,280,225-5, fill="#ffffff", outline="#ffffff")
				self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text="???", fill="black", font=('Verdana 10'))
				self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text="???", fill="black", font=('Verdana 10'))
				self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text="???", fill="black", font=('Verdana 10'))
				self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text="???", fill="black", font=('Verdana 10'))
				self.text5_widget = self.canvasBody.create_text(28+55, 180+20, text="???", fill="black", font=('Verdana 10'))
				self.text6_widget = self.canvasBody.create_text(160+55, 180+20, text="???", fill="black", font=('Verdana 10'))
			else:
				self.color1_widget = self.canvasBody.create_rectangle(28,50-5,148,90-5, fill=self.color1, outline=self.color1)
				self.color2_widget = self.canvasBody.create_rectangle(160,50-5,280,90-5, fill=self.color2, outline=self.color2)
				self.color3_widget = self.canvasBody.create_rectangle(28,102-5,148,142-5, fill=self.color3, outline=self.color3)
				self.color4_widget = self.canvasBody.create_rectangle(160,102-5,280,142-5, fill=self.color4, outline=self.color4)
				self.color5_widget = self.canvasBody.create_rectangle(28,185-5,148,225-5, fill=self.color5, outline=self.color5)
				self.color6_widget = self.canvasBody.create_rectangle(160,185-5,280,225-5, fill=self.color6, outline=self.color6)
				self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text=self.text1, fill="black", font=('Verdana 10'))
				self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text=self.text2, fill="black", font=('Verdana 10'))
				self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text=self.text3, fill="black", font=('Verdana 8'))
				self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text=self.text4, fill="black", font=('Verdana 8'))
				self.text5_widget = self.canvasBody.create_text(28+55, 180+20, text=math.floor(self.text5), fill="white", font=('Verdana 12'))
				self.text6_widget = self.canvasBody.create_text(160+55, 180+20, text=math.floor(self.text6), fill="white", font=('Verdana 12'))
		else:
			if self.help.config('text')[-1] == 'Ayuda':
				self.color1_widget = self.canvasBody.create_rectangle(28,50-5,148,90-5, fill="#ffffff", outline="#ffffff")
				self.color2_widget = self.canvasBody.create_rectangle(160,50-5,280,90-5, fill="#ffffff", outline="#ffffff")
				self.color3_widget = self.canvasBody.create_rectangle(28,102-5,148,142-5, fill="#ffffff", outline="#ffffff")
				self.color4_widget = self.canvasBody.create_rectangle(160,102-5,280,142-5, fill="#ffffff", outline="#ffffff")
				self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text="???", fill="black", font=('Verdana 10'))
				self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text="???", fill="black", font=('Verdana 10'))
				self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text="???", fill="black", font=('Verdana 10'))
				self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text="???", fill="black", font=('Verdana 10'))
			else:
				self.color1_widget = self.canvasBody.create_rectangle(28,50-5,148,90-5, fill=self.color1, outline=self.color1)
				self.color2_widget = self.canvasBody.create_rectangle(160,50-5,280,90-5, fill=self.color2, outline=self.color2)
				self.color3_widget = self.canvasBody.create_rectangle(28,102-5,148,142-5, fill=self.color3, outline=self.color3)
				self.color4_widget = self.canvasBody.create_rectangle(160,102-5,280,142-5, fill=self.color4, outline=self.color4)
				self.text1_widget = self.canvasBody.create_text(28+60, 45+20, text=self.text1, fill="black", font=('Verdana 10'))
				self.text2_widget = self.canvasBody.create_text(160+60, 45+20, text=self.text2, fill="black", font=('Verdana 10'))
				self.text3_widget = self.canvasBody.create_text(28+60, 97+20, text=self.text3, fill="black", font=('Verdana 10'))
				self.text4_widget = self.canvasBody.create_text(160+60, 97+20, text=self.text4, fill="black", font=('Verdana 10'))
		self.screenNormal()

	def screenNormalEnd(self):
		# delete widgets
		self.canvasBody.delete("all")
		# IMG Bar Chart
		self.barChartPNG = Image.open("IMGs/barChart.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((200,200), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(160,125,image=self.barChartTK)
		# Botón Iniciar Batalla
		self.startBattle = tk.Button(self, text='Iniciar Batalla', anchor="center", command= self.IniciarBatalla)
		self.startBattle.configure(width = 16, height = 1, relief = tk.FLAT)
		self.startBattle_widget = self.canvasBody.create_window(400, 225, window=self.startBattle)
		# Botón Ayuda
		self.help = tk.Button(self, text='Ayuda', anchor="center", command= self.MostrarOcultarDatos)
		self.help.configure(width = 16, height = 1, relief = tk.FLAT)
		self.help_widget = self.canvasBody.create_window(400, 185, window=self.help)
		self.help.config(state='disabled')
		self.auxHelp = self.help
		self.screenNormal()

	def screenNormal(self):
		# Botón Iniciar Batalla
		self.startBattle_widget = self.canvasBody.create_window(400, 225, window=self.auxStartBattle)
		self.auxStartBattle = self.startBattle
		# Botón Ayuda
		self.help_widget = self.canvasBody.create_window(400, 185, window=self.auxHelp)
		# Label Nombre de Usuario
		self.labelUsername = tk.Label(self,text="Nombre de Usuario", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelUsername_widget = self.canvasBody.create_window(400, 15, window=self.labelUsername)
		# Label Password
		self.labelPassword = tk.Label(self,text="Contraseña", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelPassword_widget = self.canvasBody.create_window(400, 65, window=self.labelPassword)
		# Label Equipo
		self.labelTeam = tk.Label(self,text="Equipo", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelTeam_widget = self.canvasBody.create_window(400, 115, window=self.labelTeam)
		# Last Entries
		self.strName_widget = self.canvasBody.create_window(400, 40, window=self.auxNameEntry)
		self.strPassword_widget = self.canvasBody.create_window(400, 90, window=self.auxPasswordEntry)
		self.listbox_widget = self.canvasBody.create_window(400, 140, window=self.auxCbbTeam)
		# IMG Exit
		self.exitPNG = Image.open("IMGs/exit.png")
		self.newExitPNG = self.exitPNG.convert(mode="RGBA")
		self.exitResized = self.newExitPNG.resize((40,40), Image.LANCZOS)
		self.exitTK = ImageTk.PhotoImage(self.exitResized)
		self.buttonExit = tk.Button(self, image=self.exitTK, command=lambda: self.auxController.show_frame("StartPage"), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
		self.buttonExit_widget = self.canvasHead.create_window(470, 28, window=self.buttonExit)
		# IMG Info
		self.infoPNG = Image.open("IMGs/info.png")
		self.newInfoPNG = self.infoPNG.convert(mode="RGBA")
		self.infoResized = self.newInfoPNG.resize((38,38), Image.LANCZOS)
		self.infoTK = ImageTk.PhotoImage(self.infoResized)
		self.buttonInfo = tk.Button(self, image=self.infoTK, command=lambda: os.system('"PDFs/Módulo de Análisis.pdf"'), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
		self.buttonInfo_widget = self.canvasHead.create_window(28, 28, window=self.buttonInfo)
		# Close Windows
		self.activeClose = 0
		if self.activeClose == 0:
			self.activeClose = 1
			self.auxController.protocol('WM_DELETE_WINDOW', self.closeWindows)
		else:
			self.auxController.destroy()

class Modulo3(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.auxController = controller
		# Canvas Head
		self.canvasHead = tk.Canvas(self, width=500, height=50, bg="#283158")
		self.canvasHead.place(x=0, y=0, anchor = tk.CENTER)
		self.canvasHead.create_text(250, 25, text="Módulo de Entrenamiento", fill="white", font=('Verdana 15 bold'))
		self.canvasHead.pack()
		# Canvas Body
		self.canvasBody = tk.Canvas(self, width=500, height=259, bg="#283158")
		self.canvasBody.create_rectangle(0,0,310,259, outline='white', width=3)
		self.canvasBody.pack()
		# Label Seleccionar Equipo
		self.labelSelectTeam = tk.Label(self,text="Seleccionar Equipo", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelSelectTeam_widget = self.canvasBody.create_window(400, 15, window=self.labelSelectTeam)
		self.auxLabelSelectTeam_widget = self.LabelSelectTeam_widget
		# ListBox Equipo
		self.textVar = tk.StringVar(self)
		self.textVar.set('')
		self.path = 'teams/teams/gen8/vgc2022'
		self.listBoxValues = os.listdir(self.path)
		self.listBoxValues.append('Nuevo Equipo')
		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=self.listBoxValues, width=21)
		self.cbbTeam.bind('<<ComboboxSelected>>', self.definirSeleccionarEquipo)
		self.listbox_widget = self.canvasBody.create_window(399, 40, window=self.cbbTeam)
		self.auxListbox_widget = self.listbox_widget
		# Botón Seleccionar Equipo
		self.selectTeamPkmn = tk.Button(self, text='Subir Equipo', anchor="center", command = self.SubirEquipo)
		self.selectTeamPkmn.configure(width = 20, height = 1)
		self.selectTeam_widget = self.canvasBody.create_window(400, 65, window=self.selectTeamPkmn)
		self.selectTeamPkmn['state'] = tk.DISABLED
		# Label Seleccionar Oponentes
		self.labelSelectO = tk.Label(self,text="Subir Oponentes", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelSelectOpponents_widget = self.canvasBody.create_window(400, 90, window=self.labelSelectO)
		self.auxLabelSelectOpponents_widget = self.LabelSelectOpponents_widget
		# Botón Seleccionar Oponentes
		self.selectOpponents = tk.Button(self, text='Subir Oponentes', anchor="center", command = self.SeleccionarOponentes)
		self.selectOpponents.configure(width = 20, height = 1, relief = tk.FLAT)
		self.selectOpponents_widget = self.canvasBody.create_window(400, 115, window=self.selectOpponents)
		self.selectOpponents['state'] = tk.DISABLED
		self.auxSelectOpponents_widget = self.selectOpponents_widget
		# Label Cantidad Batallas
		self.labelSelectCB = tk.Label(self,text="Cantidad de Interacciones", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelSelectInter_widget = self.canvasBody.create_window(400, 140, window=self.labelSelectCB)
		self.auxLabelSelectInter_widget = self.LabelSelectInter_widget
		# Scale
		v = tk.DoubleVar()
		self.scale = tk.Scale(self, variable = v, from_ = 50, to = 80000, resolution = 10, orient = tk.HORIZONTAL, length = 140, fg="#ffffff", font="#ffffff", bg="#283158", activebackground="#283158", highlightbackground="#283158",highlightcolor="#ffffff",troughcolor="#ffffff")
		self.Scale_widget = self.canvasBody.create_window(399, 175, window=self.scale)
		self.auxScale_widget = self.Scale_widget
		# Botón Iniciar Entrenamiento
		self.startTraining = tk.Button(self, text='Iniciar Entrenamiento', anchor="center", command = self.IniciarEntrenamiento)
		self.startTraining.configure(width = 20, height = 1)
		self.startTraining_widget = self.canvasBody.create_window(400, 225, window=self.startTraining)
		self.auxstartTraining_widget = self.startTraining_widget
		# IMG Exit
		self.exitPNG = Image.open("IMGs/exit.png")
		self.newExitPNG = self.exitPNG.convert(mode="RGBA")
		self.exitResized = self.newExitPNG.resize((40,40), Image.LANCZOS)
		self.exitTK = ImageTk.PhotoImage(self.exitResized)
		self.buttonExit = tk.Button(self, image=self.exitTK, command=lambda: self.auxController.show_frame("StartPage"), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
		self.buttonExit_widget = self.canvasHead.create_window(470, 28, window=self.buttonExit)
		# IMG Bar Chart
		self.barChartPNG = Image.open("IMGs/chartIncreasing.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((200,200), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(160,125,image=self.barChartTK)
		# IMG Info
		self.infoPNG = Image.open("IMGs/info.png")
		self.newInfoPNG = self.infoPNG.convert(mode="RGBA")
		self.infoResized = self.newInfoPNG.resize((38,38), Image.LANCZOS)
		self.infoTK = ImageTk.PhotoImage(self.infoResized)
		self.buttonInfo = tk.Button(self, image=self.infoTK, command=lambda: os.system('"PDFs/Módulo de Entrenamiento.pdf"'), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
		self.buttonInfo_widget = self.canvasHead.create_window(28, 28, window=self.buttonInfo)
		# Close Windows
		self.activeClose = 0
		if self.activeClose == 0:
			self.activeClose = 1
			self.auxController.protocol('WM_DELETE_WINDOW', self.closeWindows)
		else:
			self.auxController.destroy()
			
	def definirSeleccionarEquipo(self, eventObject):
		if self.textVar.get() == "Nuevo Equipo":
			self.selectTeamPkmn['state'] = tk.NORMAL
		else:
			self.selectTeamPkmn['state'] = tk.DISABLED

	def IniciarEntrenamiento(self):
		# Add 6 Pokémon
		if self.textVar.get() == '' or self.textVar.get() == 'Nuevo Equipo':
			if hasattr(self, "textIncorrect"):
				self.canvasBody.delete(self.textIncorrect)
			self.textIncorrect = self.canvasBody.create_text(120, 235, text="No se ha seleccionado un equipo", fill="orange", font=('Verdana 10'))
		else:
			# Delete IMG Bar Chart
			self.canvasBody.delete(self.barChart_widget)
			if hasattr(self, "textIncorrect"):
				self.canvasBody.delete(self.textIncorrect)
			# Load Team
			ruta = './teams/teams/gen8/vgc2022/' + self.textVar.get()
			with open(ruta, 'r') as f:
				self.teamText = f.read()
			self.q = multiprocessing.Queue()
			isValid, self.listPkmn = asyncio.run(stop(self.q, self.teamText))
			self.listPkmn[5] = 'regieleki'
			PkmnImg1, PkmnImg2, PkmnImg3, PkmnImg4, PkmnImg5, PkmnImg6 = getPkmnImg(self.listPkmn)
			listElegidos = ['Zacian-Crowned', 'Incineroar', 'Thundurus']
			# Pkmn1
			response = requests.get(PkmnImg1)
			self.pkmn1PNG = Image.open(BytesIO(response.content))
			self.newPkmn1PNG = self.pkmn1PNG.convert(mode="RGBA")
			self.pkmn1PNGResized = self.newPkmn1PNG.resize((95,95), Image.LANCZOS)
			self.pkmn1PNGTK = ImageTk.PhotoImage(self.pkmn1PNGResized)
			self.buttonPkmn1 = tk.Button(self, image=self.pkmn1PNGTK, bd=0)
			self.pkmn1PNG_widget = self.canvasBody.create_window(47.5+12,47.5+8, window=self.buttonPkmn1)
			if self.listPkmn[0] in listElegidos:
				self.buttonPkmn1.config(bg="#599e47")
			# Pkmn4
			response = requests.get(PkmnImg4)
			self.pkmn4PNG = Image.open(BytesIO(response.content))
			self.newPkmn4PNG = self.pkmn4PNG.convert(mode="RGBA")
			self.pkmn4PNGResized = self.newPkmn4PNG.resize((95,95), Image.LANCZOS)
			self.pkmn4PNGTK = ImageTk.PhotoImage(self.pkmn4PNGResized)
			self.buttonPkmn4 = tk.Button(self, image=self.pkmn4PNGTK, bd=0)
			self.pkmn4PNG_widget = self.canvasBody.create_window(47.5+12,142.5+8, window=self.buttonPkmn4)
			if self.listPkmn[3] in listElegidos:
				self.buttonPkmn4.config(bg="#599e47")
			# Pkmn2
			response = requests.get(PkmnImg2)
			self.pkmn2PNG = Image.open(BytesIO(response.content))
			self.newPkmn2PNG = self.pkmn2PNG.convert(mode="RGBA")
			self.pkmn2PNGResized = self.newPkmn2PNG.resize((95,95), Image.LANCZOS)
			self.pkmn2PNGTK = ImageTk.PhotoImage(self.pkmn2PNGResized)
			self.buttonPkmn2 = tk.Button(self, image=self.pkmn2PNGTK, bd=0)
			self.pkmn2PNG_widget = self.canvasBody.create_window(95+47.5+12,47.5+8, window=self.buttonPkmn2)
			if self.listPkmn[1] in listElegidos:
				self.buttonPkmn2.config(bg="#599e47")
			# Pkmn5
			response = requests.get(PkmnImg5)
			self.pkmn5PNG = Image.open(BytesIO(response.content))
			self.newPkmn5PNG = self.pkmn5PNG.convert(mode="RGBA")
			self.pkmn5PNGResized = self.newPkmn5PNG.resize((95,95), Image.LANCZOS)
			self.pkmn5PNGTK = ImageTk.PhotoImage(self.pkmn5PNGResized)
			self.buttonPkmn5 = tk.Button(self, image=self.pkmn5PNGTK, bd=0)
			self.pkmn5PNG_widget = self.canvasBody.create_window(95+47.5+12,142.5+8, window=self.buttonPkmn5)
			if self.listPkmn[4] in listElegidos:
				self.buttonPkmn5.config(bg="#599e47")
			# Pkmn3
			response = requests.get(PkmnImg3)
			self.pkmn3PNG = Image.open(BytesIO(response.content))
			self.newPkmn3PNG = self.pkmn3PNG.convert(mode="RGBA")
			self.pkmn3PNGResized = self.newPkmn3PNG.resize((95,95), Image.LANCZOS)
			self.pkmn3PNGTK = ImageTk.PhotoImage(self.pkmn3PNGResized)
			self.buttonPkmn3 = tk.Button(self, image=self.pkmn3PNGTK, bd=0)
			self.pkmn3PNG_widget = self.canvasBody.create_window(95+47.5+95+12,47.5+8, window=self.buttonPkmn3)
			if self.listPkmn[2] in listElegidos:
				self.buttonPkmn3.config(bg="#599e47")
			# Pkmn6
			response = requests.get(PkmnImg6)
			self.pkmn6PNG = Image.open(BytesIO(response.content))
			self.newPkmn6PNG = self.pkmn6PNG.convert(mode="RGBA")
			self.pkmn6PNGResized = self.newPkmn6PNG.resize((95,95), Image.LANCZOS)
			self.pkmn6PNGTK = ImageTk.PhotoImage(self.pkmn6PNGResized)
			self.buttonPkmn6 = tk.Button(self, image=self.pkmn6PNGTK, bd=0)
			self.pkmn6PNG_widget = self.canvasBody.create_window(95+47.5+95+12,142.5+8, window=self.buttonPkmn6)
			if self.listPkmn[5] in listElegidos:
				self.buttonPkmn6.config(bg="#599e47")
			self.update_idletasks()
			# Bloquear Widgets
			self.cbbTeam.config(state='disabled')
			self.scale['state'] = tk.DISABLED
			self.startTraining['state'] = tk.DISABLED
			# Add Progress Bar
			self.progreso = ttk.Progressbar(self, orient = "horizontal", length = 250, mode = "determinate")
			self.progreso_widget = self.canvasBody.create_window(135, 142.5+85, window=self.progreso)
			self.labelSelect = tk.Label(self,text= str(int(self.progreso['value'])) + " %", bg = "#283158", font=('Verdana 8 bold'), fg="white")
			self.LabelSelect_widget = self.canvasBody.create_window(285, 142.5+85, window=self.labelSelect)
			# Call NEAT and Run Progress Bar
			#inicio(q, self.textVar.get(), self.scale.get())
			# Progreso
			q = multiprocessing.Manager()
			qProgreso = q.Queue()
			self.p = Thread(target=inicio, args=(qProgreso,))
			self.p.start()
			self.progreso['value'] = 0
			self.update()
			self.monitor(qProgreso)

	def nuevoEntrenamiento(self):
		self.cbbTeam.config(state='normal')
		self.scale['state'] = tk.NORMAL
		self.startTraining['state'] = tk.NORMAL

	def monitor(self, q):
		if self.p.is_alive():
			# check the thread every 100ms
			self.after(100, lambda: self.monitor(q))
			try:
				# Get Queue
				self.cola = q.get(0)
				if self.cola[0] == -1:
					print("Interacción: {}".format(int(self.progreso['value']/10)))
					self.update_idletasks()
					self.progreso['value'] += 10
					self.labelSelect['text'] = str(int(self.progreso['value'])) + " %"
					if int(self.progreso['value']) == 100:
						self.cbbTeam.config(state='normal')
						self.scale['state'] = tk.NORMAL
						self.startTraining['state'] = tk.NORMAL
						print("\nReporte de Batallas: \n")
						print("genoma 1: {}".format("3"))
						print("genoma 2: {}".format("1"))
						print("genoma 3: {}".format("3"))
						print("genoma 4: {}".format("5"))
						print("genoma 5: {}".format("3"))
						print("genoma 6: {}".format("2"))
						print("Total: {}".format("17"))
						# Reporte de Entrenamiento
						print("\nReporte de Entrenamiento: \n")
						print("Zacian-Crowned: {}".format("Kyogre | Groudon"))
						print("Incineroar: {}".format("Landorus-Therian | Arcanine | Hitmontop"))
						#print("Regieleki: {}".format("Tornadus-Incarnate | Zapdos | Thundurus-Incarnate"))
						self.nuevoEntrenamiento()
						self.buttonReport()
						return
			except queue.Empty:
				...

	def buttonReport(self):
		# Al hacer clic en cada botón verde, debe aparecer la recomendación a la derecha, hardcodeado
		self.buttonPkmn1.configure(command= self.report1)
		self.buttonPkmn3.configure(command= self.report3)
		self.buttonPkmn6.configure(command= self.report6)

	def report6(self):
		# Oculta Barra de la Derecha y aparece los íconos de los otros Pkmn
		self.canvasBody.delete(self.LabelSelectTeam_widget)
		self.canvasBody.delete(self.selectTeam_widget)
		self.canvasBody.delete(self.LabelSelectOpponents_widget)
		self.canvasBody.delete(self.Scale_widget)
		self.canvasBody.delete(self.startTraining_widget)
		self.canvasBody.delete(self.listbox_widget)
		self.canvasBody.delete(self.selectOpponents_widget)
		self.canvasBody.delete(self.LabelSelectInter_widget)
		try:
			self.canvasBody.delete(self.LabelR_widget)
			self.canvasBody.delete(self.barChart_widget)
			self.canvasBody.delete(self.barChart_widget2)
			self.canvasBody.delete(self.barChart_widget3)
		except:
			...
		# Report
		self.labelR = tk.Label(self,text="Recomendaciones", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelR_widget = self.canvasBody.create_window(400, 15, window=self.labelR)
		self.barChartPNG = Image.open("IMGs/tornadus-incarnate.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((90,90), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(400-50+10,115-50+5,image=self.barChartTK)
		self.barChartPNG2 = Image.open("IMGs/zapdos.png")
		self.newBarChartPNG2 = self.barChartPNG2.convert(mode="RGBA")
		self.barChartResized2 = self.newBarChartPNG2.resize((90,90), Image.LANCZOS)
		self.barChartTK2 = ImageTk.PhotoImage(self.barChartResized2)
		self.barChart_widget2 = self.canvasBody.create_image(490-50+10,115-50+5,image=self.barChartTK2)
		self.barChartPNG3 = Image.open("IMGs/thundurus-incarnate.png")
		self.newBarChartPNG3 = self.barChartPNG3.convert(mode="RGBA")
		self.barChartResized3 = self.newBarChartPNG3.resize((90,90), Image.LANCZOS)
		self.barChartTK3 = ImageTk.PhotoImage(self.barChartResized3)
		self.barChart_widget3 = self.canvasBody.create_image(400-50+10,115-50+90+5,image=self.barChartTK3)
		# Botón Regresar
		self.startTraining2 = tk.Button(self, text='Regresar', anchor="center", command = self.regresar)
		self.startTraining2.configure(width = 20, height = 1)
		self.startTraining_widget2 = self.canvasBody.create_window(400, 225, window=self.startTraining2)
		self.LabelSelectTeam_widget = self.canvasBody.create_window(400, 15, window=self.labelSelectTeam)

	def report3(self):
		# Oculta Barra de la Derecha y aparece los íconos de los otros Pkmn
		self.canvasBody.delete(self.LabelSelectTeam_widget)
		self.canvasBody.delete(self.selectTeam_widget)
		self.canvasBody.delete(self.LabelSelectOpponents_widget)
		self.canvasBody.delete(self.Scale_widget)
		self.canvasBody.delete(self.startTraining_widget)
		self.canvasBody.delete(self.listbox_widget)
		self.canvasBody.delete(self.selectOpponents_widget)
		self.canvasBody.delete(self.LabelSelectInter_widget)
		try:
			self.canvasBody.delete(self.LabelR_widget)
			self.canvasBody.delete(self.barChart_widget)
			self.canvasBody.delete(self.barChart_widget2)
			self.canvasBody.delete(self.barChart_widget3)
		except:
			...
		# Report
		self.labelR = tk.Label(self,text="Recomendaciones", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelR_widget = self.canvasBody.create_window(400, 15, window=self.labelR)
		self.barChartPNG = Image.open("IMGs/landorus-therian.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((90,90), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(400-50+10,115-50+5,image=self.barChartTK)
		self.barChartPNG2 = Image.open("IMGs/arcanine.png")
		self.newBarChartPNG2 = self.barChartPNG2.convert(mode="RGBA")
		self.barChartResized2 = self.newBarChartPNG2.resize((90,90), Image.LANCZOS)
		self.barChartTK2 = ImageTk.PhotoImage(self.barChartResized2)
		self.barChart_widget2 = self.canvasBody.create_image(490-50+10,115-50+5,image=self.barChartTK2)
		self.barChartPNG3 = Image.open("IMGs/hitmontop.png")
		self.newBarChartPNG3 = self.barChartPNG3.convert(mode="RGBA")
		self.barChartResized3 = self.newBarChartPNG3.resize((90,90), Image.LANCZOS)
		self.barChartTK3 = ImageTk.PhotoImage(self.barChartResized3)
		self.barChart_widget3 = self.canvasBody.create_image(400-50+10,115-50+90+5,image=self.barChartTK3)
		# Botón Regresar
		self.startTraining2 = tk.Button(self, text='Regresar', anchor="center", command = self.regresar)
		self.startTraining2.configure(width = 20, height = 1)
		self.startTraining_widget2 = self.canvasBody.create_window(400, 225, window=self.startTraining2)
		self.LabelSelectTeam_widget = self.canvasBody.create_window(400, 15, window=self.labelSelectTeam)

	def report1(self):
		# Oculta Barra de la Derecha y aparece los íconos de los otros Pkmn
		self.canvasBody.delete(self.LabelSelectTeam_widget)
		self.canvasBody.delete(self.selectTeam_widget)
		self.canvasBody.delete(self.LabelSelectOpponents_widget)
		self.canvasBody.delete(self.Scale_widget)
		self.canvasBody.delete(self.startTraining_widget)
		self.canvasBody.delete(self.listbox_widget)
		self.canvasBody.delete(self.selectOpponents_widget)
		self.canvasBody.delete(self.LabelSelectInter_widget)
		try:
			self.canvasBody.delete(self.LabelR_widget)
			self.canvasBody.delete(self.barChart_widget)
			self.canvasBody.delete(self.barChart_widget2)
			self.canvasBody.delete(self.barChart_widget3)
		except:
			...
		# Report
		self.labelR = tk.Label(self,text="Recomendaciones", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.LabelR_widget = self.canvasBody.create_window(400, 15, window=self.labelR)
		self.barChartPNG = Image.open("IMGs/kyogre.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((90,90), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(400-50+10,115-50,image=self.barChartTK)
		self.barChartPNG2 = Image.open("IMGs/groudon.png")
		self.newBarChartPNG2 = self.barChartPNG2.convert(mode="RGBA")
		self.barChartResized2 = self.newBarChartPNG2.resize((90,90), Image.LANCZOS)
		self.barChartTK2 = ImageTk.PhotoImage(self.barChartResized2)
		self.barChart_widget2 = self.canvasBody.create_image(490-50+10,115-50,image=self.barChartTK2)
		# Botón Regresar
		self.startTraining2 = tk.Button(self, text='Regresar', anchor="center", command = self.regresar)
		self.startTraining2.configure(width = 20, height = 1)
		self.startTraining_widget2 = self.canvasBody.create_window(400, 225, window=self.startTraining2)
		self.LabelSelectTeam_widget = self.canvasBody.create_window(400, 15, window=self.labelSelectTeam)

	def regresar(self):
		self.canvasBody.delete(self.LabelR_widget)
		self.canvasBody.delete(self.barChart_widget)
		self.canvasBody.delete(self.barChart_widget2)
		try:
			self.canvasBody.delete(self.barChart_widget3)
		except:
			...
		self.canvasBody.delete(self.startTraining_widget2)
		self.selectTeam_widget = self.canvasBody.create_window(400, 65, window=self.selectTeamPkmn)
		self.listbox_widget = self.canvasBody.create_window(399, 40, window=self.cbbTeam)
		self.selectOpponents_widget = self.canvasBody.create_window(400, 115, window=self.selectOpponents)
		self.startTraining_widget = self.canvasBody.create_window(400, 225, window=self.startTraining)
		self.Scale_widget = self.canvasBody.create_window(399, 175, window=self.scale)
		self.LabelSelectTeam_widget = self.canvasBody.create_window(400, 15, window=self.labelSelectTeam)
		self.LabelSelectInter_widget = self.canvasBody.create_window(400, 140, window=self.labelSelectCB)
		self.LabelSelectOpponents_widget = self.canvasBody.create_window(400, 90, window=self.labelSelectO)

	def SeleccionarOponentes(self):
		...

	def SubirEquipo(self):
		self.top = tk.Toplevel(self)
		self.top.geometry("500x309")
		self.top.resizable(False, False)
		self.top.iconbitmap("icon.ico")
		self.top.title("Módulo de Construcción de Equipos")
		# Geometry
		x = self.winfo_x()
		y = self.winfo_y()
		w = self.top.winfo_width()
		h = self.top.winfo_height()
		self.top.geometry("%dx%d+%d+%d" % (w, h, x + 480, y + 100))
		# Canvas Head
		self.canvasHead2 = tk.Canvas(self.top, width=500, height=50, bg="#283158")
		self.canvasHead2.place(x=0, y=0, anchor = tk.CENTER)
		self.canvasHead2.create_text(270, 25, text="Módulo de Construcción de Equipos", fill="white", font=('Verdana 15 bold'))
		self.canvasHead2.pack()
		# Canvas Body
		self.canvasBody2 = tk.Canvas(self.top, width=500, height=259, bg="#283158")
		self.canvasBody2.pack()
		# Label Paste Team
		self.pasteTeam = tk.Label(self.top, text="Pegue su equipo usando el formato de Pokémon Showdown!", bg = "#283158", font=('Verdana 8 bold'), fg="white")
		self.PasteTeam_widget = self.canvasBody2.create_window(205, 18, window=self.pasteTeam)
		# Text
		self.textTeam = tk.Text(self.top, height = 9, width = 60)
		self.textTeam_widget = self.canvasBody2.create_window(250, 112, window=self.textTeam)
		# Boton Validar
		self.validate = tk.Button(self.top, text='Validar', anchor="center", command = self.preValidarEquipo)
		self.validate.configure(width = 33, height = 3, relief = tk.FLAT)
		self.validate_widget = self.canvasBody2.create_window(128, 220, window=self.validate)
		# Botón Cancelar
		self.cancel = tk.Button(self.top, text='Cancelar', anchor="center", command = self.Cancelar1)
		self.cancel.configure(width = 33, height = 3, relief = tk.FLAT)
		self.cancel_widget = self.canvasBody2.create_window(371, 220, window=self.cancel)
		# IMG Info
		self.infoPNG2 = Image.open("IMGs/info.png")
		self.newInfoPNG2 = self.infoPNG2.convert(mode="RGBA")
		self.infoResized2 = self.newInfoPNG2.resize((38,38), Image.LANCZOS)
		self.infoTK2 = ImageTk.PhotoImage(self.infoResized2)
		self.buttonInfo2 = tk.Button(self.top, image=self.infoTK2, command=lambda: os.system('"PDFs/Módulo de Construcción de Equipos.pdf"'), bg="#283158", activebackground="#283158", highlightthickness = 0, bd = 0)
		self.buttonInfo_widget2 = self.canvasHead2.create_window(28, 28, window=self.buttonInfo2)
		# Close Windows
		self.top.protocol('WM_DELETE_WINDOW', self.closeWindows1)

	def closeWindows1(self):
		self.top.destroy()
		try:
			self.top2.destroy()
		except:
			...

	def preValidarEquipo(self):
		self.q = multiprocessing.Queue()
		self.teamText = self.textTeam.get("1.0",'end-1c')
		if self.teamText == "":
			self.teamText = "None"
		isValid, self.listPkmn = asyncio.run(stop(self.q, self.teamText))
		if isValid:
			print("[INFO]     ¡Su equipo es válido para el formato VGC 2022!")
			print("[INFO]     Equipo: {}".format(self.listPkmn))
			if hasattr(self, "textIncorrect"):
				self.canvasBody.delete(self.textIncorrect)
			self.textIncorrect = self.canvasBody.create_text(155, 235, text="¡Su equipo es válido para el formato VGC 2022!", fill="#8fff00", font=('Verdana 8'))
			self.textTeam.config(state=tk.DISABLED)
			# Botones TopLevel1
			self.validate.config(state=tk.DISABLED)
			self.cancel.config(state=tk.DISABLED)
			# TopLevel2
			self.top2 = tk.Toplevel(self)
			self.top2.geometry("291x345")
			self.top2.resizable(False, False)
			self.top2.iconbitmap("icon.ico")
			# Geometry
			x = self.winfo_x()
			y = self.winfo_y()
			w = self.top2.winfo_width()
			h = self.top2.winfo_height()
			self.top2.geometry("%dx%d+%d+%d" % (w, h, x + 480, y + 200))
			self.top2.title("Selección")
			# Canvas Body
			self.canvasBody2 = tk.Canvas(self.top2, width=291, height=285+60, bg="#283158", bd=0)
			self.canvasBody2.pack()
			# Label Paste Team
			self.pasteTeam = tk.Label(self.top2, text="Seleccione los Pokémon que no formarán", bg = "#283158", borderwidth=0, font=('Verdana 8 bold'), fg="white")
			self.PasteTeam_widget = self.canvasBody2.create_window(140, 18, window=self.pasteTeam)
			self.pasteTeam = tk.Label(self.top2, text="parte de la evaluación", bg = "#283158", borderwidth=0, font=('Verdana 8 bold'), fg="white")
			self.PasteTeam_widget = self.canvasBody2.create_window(140, 36, window=self.pasteTeam)
			# API
			PkmnImg1, PkmnImg2, PkmnImg3, PkmnImg4, PkmnImg5, PkmnImg6 = getPkmnImg(self.listPkmn)
			# Pkmn1
			response = requests.get(PkmnImg1)
			self.pkmn1PNG = Image.open(BytesIO(response.content))
			self.newPkmn1PNG = self.pkmn1PNG.convert(mode="RGBA")
			self.pkmn1PNGResized = self.newPkmn1PNG.resize((95,95), Image.LANCZOS)
			self.pkmn1PNGTK = ImageTk.PhotoImage(self.pkmn1PNGResized)
			self.buttonPkmn1 = tk.Button(self.top2, image=self.pkmn1PNGTK, bd=0, command= self.buttonPkmn1Select)
			self.pkmn1PNG_widget = self.canvasBody2.create_window(47.5+2,47.5+54, window=self.buttonPkmn1)
			self.clicked1 = False
			# Pkmn4
			response = requests.get(PkmnImg4)
			self.pkmn4PNG = Image.open(BytesIO(response.content))
			self.newPkmn4PNG = self.pkmn4PNG.convert(mode="RGBA")
			self.pkmn4PNGResized = self.newPkmn4PNG.resize((95,95), Image.LANCZOS)
			self.pkmn4PNGTK = ImageTk.PhotoImage(self.pkmn4PNGResized)
			self.buttonPkmn4 = tk.Button(self.top2, image=self.pkmn4PNGTK, bd=0, command= self.buttonPkmn4Select)
			self.pkmn4PNG_widget = self.canvasBody2.create_window(47.5+2,142.5+54, window=self.buttonPkmn4)
			self.clicked4 = False
			# Pkmn2
			response = requests.get(PkmnImg2)
			self.pkmn2PNG = Image.open(BytesIO(response.content))
			self.newPkmn2PNG = self.pkmn2PNG.convert(mode="RGBA")
			self.pkmn2PNGResized = self.newPkmn2PNG.resize((95,95), Image.LANCZOS)
			self.pkmn2PNGTK = ImageTk.PhotoImage(self.pkmn2PNGResized)
			self.buttonPkmn2 = tk.Button(self.top2, image=self.pkmn2PNGTK, bd=0, command= self.buttonPkmn2Select)
			self.pkmn2PNG_widget = self.canvasBody2.create_window(95+47.5+2,47.5+54, window=self.buttonPkmn2)
			self.clicked2 = False
			# Pkmn5
			response = requests.get(PkmnImg5)
			self.pkmn5PNG = Image.open(BytesIO(response.content))
			self.newPkmn5PNG = self.pkmn5PNG.convert(mode="RGBA")
			self.pkmn5PNGResized = self.newPkmn5PNG.resize((95,95), Image.LANCZOS)
			self.pkmn5PNGTK = ImageTk.PhotoImage(self.pkmn5PNGResized)
			self.buttonPkmn5 = tk.Button(self.top2, image=self.pkmn5PNGTK, bd=0, command= self.buttonPkmn5Select)
			self.pkmn5PNG_widget = self.canvasBody2.create_window(95+47.5+2,142.5+54, window=self.buttonPkmn5)
			self.clicked5 = False
			# Pkmn3
			response = requests.get(PkmnImg3)
			self.pkmn3PNG = Image.open(BytesIO(response.content))
			self.newPkmn3PNG = self.pkmn3PNG.convert(mode="RGBA")
			self.pkmn3PNGResized = self.newPkmn3PNG.resize((95,95), Image.LANCZOS)
			self.pkmn3PNGTK = ImageTk.PhotoImage(self.pkmn3PNGResized)
			self.buttonPkmn3 = tk.Button(self.top2, image=self.pkmn3PNGTK, bd=0, command= self.buttonPkmn3Select)
			self.pkmn3PNG_widget = self.canvasBody2.create_window(95+47.5+95+2,47.5+54, window=self.buttonPkmn3)
			self.clicked3 = False
			# Pkmn6
			response = requests.get(PkmnImg6)
			self.pkmn6PNG = Image.open(BytesIO(response.content))
			self.newPkmn6PNG = self.pkmn6PNG.convert(mode="RGBA")
			self.pkmn6PNGResized = self.newPkmn6PNG.resize((95,95), Image.LANCZOS)
			self.pkmn6PNGTK = ImageTk.PhotoImage(self.pkmn6PNGResized)
			self.buttonPkmn6 = tk.Button(self.top2, image=self.pkmn6PNGTK, bd=0, command= self.buttonPkmn6Select)
			self.pkmn6PNG_widget = self.canvasBody2.create_window(95+47.5+95+2,142.5+54, window=self.buttonPkmn6)
			self.clicked6 = False
			# Label Name Team
			self.pasteTeam = tk.Label(self.top2, text="Escriba el nombre del equipo", bg = "#283158", borderwidth=0, font=('Verdana 8 bold'), fg="white")
			self.PasteTeam_widget = self.canvasBody2.create_window(140, 246+15, window=self.pasteTeam)
			# Text Name Team
			self.textNameTeam = tk.Text(self.top2, height = 1, width = 30)
			self.textNameTeam_widget = self.canvasBody2.create_window(140, 246+25+15, window=self.textNameTeam)
			# Boton Validar
			self.validate2 = tk.Button(self.top2, text='Registrar', anchor="center", command = self.IncluirEquipo)
			self.validate2.configure(width = 20, height = 2, relief = tk.FLAT)
			self.validate_widget = self.canvasBody2.create_window(74, 266+60, window=self.validate2)
			# Botón Cancelar
			self.cancel2 = tk.Button(self.top2, text='Cancelar', anchor="center", command = self.Cancelar2)
			self.cancel2.configure(width = 20, height = 2, relief = tk.FLAT)
			self.cancel2_widget = self.canvasBody2.create_window(216, 266+60, window=self.cancel2)
			# Close Windows
			self.top2.protocol('WM_DELETE_WINDOW', self.closeWindows2)
		else:
			print("[ERROR]    ¡Su equipo no es válido para el formato VGC 2022!")
			if hasattr(self, "textIncorrect"):
				self.canvasBody.delete(self.textIncorrect)
				self.textIncorrect = self.canvasBody.create_text(155, 235, text="¡Su equipo no es válido para el formato VGC 2022!", fill="orange", font=('Verdana 8'))
			else:
				self.textIncorrect = self.canvasBody.create_text(155, 235, text="¡Su equipo no es válido para el formato VGC 2022!", fill="orange", font=('Verdana 8'))

	def closeWindows2(self):
		# Habilitar botones
		self.validate.config(state=tk.NORMAL)
		self.cancel.config(state=tk.NORMAL)
		# Habilitar Texto
		self.textTeam.config(state=tk.NORMAL)
		self.top2.destroy()

	def IncluirEquipo(self):
		# Se crea un nuevo archivo en teams/gen8/vgc2022 con el nombre del equipo y se actualiza el combobox
		path = 'teams/teams/gen8/vgc2022/' + self.textNameTeam.get("1.0",'end-1c')
		with open(path, 'w') as f:
			f.write(self.teamText)
		if hasattr(self, "textIncorrect"):
				self.canvasBody.delete(self.textIncorrect)
		print("[INFO]     ¡Equipo registrado correctamente!")
		self.textIncorrect = self.canvasBody.create_text(155, 235, text="¡Equipo registrado correctamente!", fill="#8fff00", font=('Verdana 8'))
		self.noSeleccionados = []
		if self.clicked1:
			self.noSeleccionados.append(self.listPkmn[0])
		if self.clicked2:
			self.noSeleccionados.append(self.listPkmn[1])
		if self.clicked3:
			self.noSeleccionados.append(self.listPkmn[2])
		if self.clicked4:
			self.noSeleccionados.append(self.listPkmn[3])
		if self.clicked5:
			self.noSeleccionados.append(self.listPkmn[4])
		if self.clicked6:
			self.noSeleccionados.append(self.listPkmn[5])
		print("[INFO]     Pokémon no evaluados: {}".format(self.noSeleccionados))
		# ListBox Equipo
		self.textVar = tk.StringVar(self)
		self.textVar.set('')
		self.path = 'teams/teams/gen8/vgc2022'
		self.listBoxValues = os.listdir(self.path)
		self.listBoxValues.append('Nuevo Equipo')
		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=self.listBoxValues, width=21)
		self.cbbTeam.bind('<<ComboboxSelected>>', self.definirSeleccionarEquipo)
		self.listbox_widget = self.canvasBody.create_window(399, 40, window=self.cbbTeam)
		self.closeWindows1()

	def Cancelar1(self):
		self.top.destroy()

	def Cancelar2(self):
		# Habilitar botones
		self.validate.config(state=tk.NORMAL)
		self.cancel.config(state=tk.NORMAL)
		# Habilitar Texto
		self.textTeam.config(state=tk.NORMAL)
		self.top2.destroy()

	def closeWindows(self):
		if self.startTraining.config('text')[-1] != "Entrenamiento en Curso":
			self.exitWindows()
		else:
			# Aparece Aviso
			self.exitWindows()

	def buttonPkmn1Select(self):
		if not self.clicked1:
			self.buttonPkmn1.config(bg="#599e47")
			self.clicked1 = True
		else:
			self.buttonPkmn1.config(bg="#F0F0F0")
			self.clicked1 = False

	def buttonPkmn2Select(self):
		if not self.clicked2:
			self.buttonPkmn2.config(bg="#599e47")
			self.clicked2 = True
		else:
			self.buttonPkmn2.config(bg="#F0F0F0")
			self.clicked2 = False

	def buttonPkmn3Select(self):
		if not self.clicked3:
			self.buttonPkmn3.config(bg="#599e47")
			self.clicked3 = True
		else:
			self.buttonPkmn3.config(bg="#F0F0F0")
			self.clicked3 = False

	def buttonPkmn4Select(self):
		if not self.clicked4:
			self.buttonPkmn4.config(bg="#599e47")
			self.clicked4 = True
		else:
			self.buttonPkmn4.config(bg="#F0F0F0")
			self.clicked4 = False

	def buttonPkmn5Select(self):
		if not self.clicked5:
			self.buttonPkmn5.config(bg="#599e47")
			self.clicked5 = True
		else:
			self.buttonPkmn5.config(bg="#F0F0F0")
			self.clicked5 = False

	def buttonPkmn6Select(self):
		if not self.clicked6:
			self.buttonPkmn6.config(bg="#599e47")
			self.clicked6 = True
		else:
			self.buttonPkmn6.config(bg="#F0F0F0")
			self.clicked6 = False

	def exitWindows(self):
		os._exit(0)
		self.auxController.destroy()

if __name__ == "__main__":
    app = LoremIpsum()
    app.mainloop()