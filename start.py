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
from old.neuroevolution.start2 import inicio
from environs import Env

BG = "#000000"
FG = "#ffffff"
ACCENT = "#33B5E5"
APP_TITLE = "IAPSVGC — Agente VGC"


def load_ui_config():
	"""Lee ajustes de .env usados por la UI (formato, carpeta de equipos, bot)."""
	env = Env()
	env.read_env()
	return {
		"pokemon_mode": env("POKEMON_MODE", "gen9championsvgc2026regmcbo3"),
		"team_dir": env("TEAM_DIR", "gen8/vgc2022").strip().strip("/"),
		"battle_bot": env("BATTLE_BOT", "random").strip().lower() or "random",
		"ps_username": env("PS_USERNAME", ""),
	}

def apply_dark_theme(root):
	root.configure(bg=BG)
	root.tk_setPalette(
		background=BG,
		foreground=FG,
		activeBackground="#222222",
		activeForeground=FG,
		highlightColor=FG,
		highlightBackground=BG,
		insertBackground=FG,
		selectBackground="#333333",
		selectForeground=FG,
		troughColor="#222222",
	)
	style = ttk.Style(root)
	try:
		style.theme_use("clam")
	except tk.TclError:
		...
	style.configure("TFrame", background=BG)
	style.configure("TLabel", background=BG, foreground=FG)
	style.configure("TEntry", fieldbackground="#111111", foreground=FG, background=BG, insertcolor=FG)
	style.configure("TCombobox", fieldbackground="#111111", foreground=FG, background=BG, arrowcolor=FG)
	style.map(
		"TCombobox",
		fieldbackground=[("readonly", "#111111"), ("disabled", "#111111")],
		foreground=[("readonly", FG), ("disabled", "#888888")],
		background=[("readonly", BG)],
	)
	style.configure("TProgressbar", background="#33B5E5", troughcolor="#222222")
	root.option_add("*TCombobox*Listbox.background", "#111111")
	root.option_add("*TCombobox*Listbox.foreground", FG)
	root.option_add("*TCombobox*Listbox.selectBackground", "#333333")
	root.option_add("*TCombobox*Listbox.selectForeground", FG)

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

class VGCApp(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		apply_dark_theme(self)
		self.ui_config = load_ui_config()
		# Container
		container = tk.Frame(self, bg=BG)
		container.pack(side="top", fill="both", expand=True)
		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)
		# Configuración Ventana
		self.title(APP_TITLE)
		self.geometry("500x360")
		self.resizable(False, False)
		try:
			self.iconbitmap("icon.ico")
		except tk.TclError:
			pass
		self.protocol("WM_DELETE_WINDOW", self._on_close_request)
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
		self._top_frame = frame
		frame.tkraise()

	def _on_close_request(self):
		"""Delega el cierre a la pantalla activa si define closeWindows; si no, sale."""
		top = getattr(self, "_top_frame", None)
		if top is not None and hasattr(top, "closeWindows"):
			top.closeWindows()
			return
		sys.exit(0)

class StartPage(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent, bg=BG)
		self.controller = controller
		cfg = controller.ui_config
		# Canvas Head
		canvasHead = tk.Canvas(self, width=500, height=50, bg=BG, highlightthickness=0)
		canvasHead.place(x=0, y=0, anchor = tk.CENTER)
		canvasHead.create_text(250, 25, text=APP_TITLE, fill="white", font=('Verdana 14 bold'))
		canvasHead.pack()
		# Canvas Body
		canvasBody = tk.Canvas(self, width=500, height=310, bg=BG, highlightthickness=0)
		canvasBody.place(x=0, y=50, anchor = tk.CENTER)
		canvasBody.pack()

		button1 = tk.Button(self, text='Módulo de Batallas', anchor="center", command=lambda: controller.show_frame("Modulo1"))
		button1.configure(width = 50, height = 2, bg="#1a1a1a", fg=FG, activebackground = ACCENT, relief = tk.FLAT)
		canvasBody.create_window(70, 25, anchor=tk.NW, window=button1)

		button2 = tk.Button(self, text='Módulo de Análisis', anchor="center", command=lambda: controller.show_frame("Modulo2"))
		button2.configure(width = 50, height = 2, bg="#1a1a1a", fg=FG, activebackground = ACCENT, relief = tk.FLAT)
		canvasBody.create_window(70, 95, anchor=tk.NW, window=button2)

		button3 = tk.Button(self, text='Módulo de Entrenamiento', anchor="center", command=lambda: controller.show_frame("Modulo3"))
		button3.configure(width = 50, height = 2, bg="#1a1a1a", fg=FG, activebackground = ACCENT, relief = tk.FLAT)
		canvasBody.create_window(70, 165, anchor=tk.NW, window=button3)

		canvasBody.create_text(
			250, 250,
			text=f"Formato: {cfg['pokemon_mode']}  ·  Bot: {cfg['battle_bot']}",
			fill="#888888",
			font=('Verdana 8'),
		)
		if cfg.get("ps_username"):
			canvasBody.create_text(
				250, 275,
				text=f"Cuenta PS: {cfg['ps_username']}",
				fill="#666666",
				font=('Verdana 8'),
			)

class Modulo1(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent, bg=BG)
		self.auxController = controller
		self.cfg = controller.ui_config
		self.p = None
		self.isWinner = False
		self.nameOpponent = ""
		self.canvasHead = tk.Canvas(self, width=500, height=50, bg=BG, highlightthickness=0)
		self.canvasHead.create_text(250, 25, text="Módulo de Batallas", fill="white", font=('Verdana 15 bold'))
		self.canvasHead.pack()
		self.canvasBody = tk.Canvas(self, width=500, height=310, bg=BG, highlightthickness=0)
		self.canvasBody.pack()
		self.button1 = tk.Button(self, text='Iniciar Batalla', anchor="center")
		self.button1.configure(width=23, height=2, relief=tk.FLAT, bg="#1a1a1a", fg=FG, activebackground=ACCENT, command=self.IniciarBatalla)
		self.button1_widget = self.canvasBody.create_window(250, 200, window=self.button1)
		self.auxButton1 = self.button1
		self.labelUsername = tk.Label(self, text="Oponente (usuario PS)", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelUsername_widget = self.canvasBody.create_window(250, 22, window=self.labelUsername)
		self.labelTeam = tk.Label(self, text="Equipo del agente", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelTeam_widget = self.canvasBody.create_window(250, 82, window=self.labelTeam)
		self.strName = tk.StringVar()
		self.nameEntry = ttk.Entry(self, textvariable=self.strName, width=28)
		self.strName_widget = self.canvasBody.create_window(250, 48, window=self.nameEntry)
		self.auxNameEntry = self.nameEntry
		self.textVar = tk.StringVar(self)
		self.textVar.set('')
		self.path = os.path.join('teams', 'teams', self.cfg['team_dir'])
		try:
			self.listBoxValues = sorted(os.listdir(self.path))
		except FileNotFoundError:
			self.listBoxValues = []
		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=self.listBoxValues, width=25)
		self.listbox_widget = self.canvasBody.create_window(250, 108, window=self.cbbTeam)
		self.auxCbbTeam = self.cbbTeam
		self.meta_text = (
			f"Formato: {self.cfg['pokemon_mode']}\n"
			f"Bot: {self.cfg['battle_bot']}  ·  Cuenta: {self.cfg.get('ps_username') or '—'}"
		)
		self.meta_widget = self.canvasBody.create_text(250, 148, text=self.meta_text, fill="#888888", font=('Verdana 8'), justify=tk.CENTER)
		self.salirPNG = Image.open("assets/images/exit.png")
		self.newSalirPNG = self.salirPNG.convert(mode="RGBA")
		self.resized_image = self.newSalirPNG.resize((40, 40), Image.LANCZOS)
		self.salirTK = ImageTk.PhotoImage(self.resized_image)
		self.buttonSalir = tk.Button(self, image=self.salirTK, command=lambda: controller.show_frame("StartPage"), bg=BG, activebackground=BG, highlightthickness=0, bd=0)
		self.canvasHead.create_window(470, 28, window=self.buttonSalir)

	def _show_error(self, text, x=250):
		if hasattr(self, "textIncorrect"):
			self.canvasBody.delete(self.textIncorrect)
		self.textIncorrect = self.canvasBody.create_text(x, 265, text=text, fill="orange", font=('Verdana 9'))

	def _unlock_form(self):
		self.nameEntry.config(state='normal')
		self.cbbTeam.config(state='readonly')
		self.button1['state'] = tk.NORMAL
		self.button1.configure(text='Iniciar Batalla')

	def monitor(self, q):
		if self.p is not None and self.p.is_alive():
			self.after(100, lambda: self.monitor(q))
			try:
				self.cola = q.get(0)
				code = self.cola[0]
				if code == 0:
					self.button1.configure(text='Batalla en Curso')
				elif code == -1:
					self._unlock_form()
					print("[INFO]     Cierre Interfaz Ganador...")
					self.isWinner = self.cola[1]
					# cola: [-1, isWinner, user_name, opponent_name]
					self.nameOpponent = self.cola[3] if len(self.cola) > 3 else self.cola[2]
					self.endBattle()
				elif code == -2:
					self._unlock_form()
					self._show_error("¡El oponente rechazó la solicitud de batalla!")
				elif code == -3:
					self._unlock_form()
					self._show_error("¡El equipo no cumple las reglas del formato!")
				elif code == -4:
					self._unlock_form()
					self._show_error("¡Usuario oponente no encontrado!")
				elif code == -5:
					self._unlock_form()
					self._show_error("¡Usuario o contraseña PS incorrectos (.env)!")
				elif code == -6:
					self._unlock_form()
					self._show_error("¡Error del sistema al iniciar/jugar la batalla!")
			except queue.Empty:
				pass
		elif self.p is not None:
			self.p.join(timeout=0.1)

	def IniciarBatalla(self):
		if hasattr(self, "textIncorrect"):
			self.canvasBody.delete(self.textIncorrect)
		username = self.nameEntry.get().strip()
		teamname = self.textVar.get()
		if not username:
			self._show_error("Indica el usuario oponente a retar")
			return
		if not teamname:
			self._show_error("No se ha seleccionado un equipo")
			return
		if not self.listBoxValues:
			self._show_error(f"No hay equipos en {self.path}")
			return
		q1 = multiprocessing.Queue()
		botmode = "CHALLENGE_USER"
		self.nameEntry.config(state='disabled')
		self.cbbTeam.config(state='disabled')
		self.button1['state'] = tk.DISABLED
		self.button1.configure(text='Esperando Confirmación')
		self.p = AsyncDownload(q1, username, None, teamname, botmode, 1)
		self.p.start()
		self.monitor(q1)

	def closeWindows(self):
		status = self.button1.config('text')[-1]
		if status not in ("Batalla en Curso", "Esperando Confirmación"):
			self.exitWindows()
			return
		self.isCloseWindows = True
		self.rectangle1 = self.canvasBody.create_rectangle(0, 0, 500, 310, fill='#4e2858')
		for widget_id in (
			self.button1_widget, self.listbox_widget, self.LabelUsername_widget,
			self.LabelTeam_widget, self.strName_widget, self.meta_widget,
		):
			self.canvasBody.delete(widget_id)
		self.emoji = Image.open("assets/images/close.png")
		self.emojiPNG = self.emoji.convert(mode="RGBA")
		self.emojiResized_image = self.emojiPNG.resize((80, 80), Image.LANCZOS)
		self.emojiTK = ImageTk.PhotoImage(self.emojiResized_image)
		self.emoji_widget = self.canvasBody.create_image(420, 85, image=self.emojiTK)
		self.textClose_widget1 = self.canvasBody.create_text(185, 75, text="¡La batalla está en curso!", fill="white", font=('Verdana 14'))
		self.textClose_widget2 = self.canvasBody.create_text(185, 100, text="¿Está seguro de salir?", fill="white", font=('Verdana 11'))
		self.buttonYes = tk.Button(self, text='Sí', anchor="center")
		self.buttonYes.configure(width=23, height=2, relief=tk.FLAT, command=self.exitWindows)
		self.buttonYes_widget = self.canvasBody.create_window(150, 200, window=self.buttonYes)
		self.buttonNo = tk.Button(self, text='No', anchor="center")
		self.buttonNo.configure(width=23, height=2, relief=tk.FLAT, command=self.screenNormalClose)
		self.buttonNo_widget = self.canvasBody.create_window(350, 200, window=self.buttonNo)

	def exitWindows(self):
		sys.exit(0)

	def endBattle(self):
		for widget_id in (
			self.button1_widget, self.listbox_widget, self.LabelUsername_widget,
			self.LabelTeam_widget, self.strName_widget, self.meta_widget,
		):
			self.canvasBody.delete(widget_id)
		opponent = self.nameOpponent or "oponente"
		if self.isWinner:
			self.emoji = Image.open("assets/images/winner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0, 0, 500, 310, fill='#325928')
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="¡Felicidades!", fill="white", font=('Verdana 14'))
			self.textWinner_widget2 = self.canvasBody.create_text(185, 105, text=f"Victoria vs {opponent}", fill="white", font=('Verdana 11'))
		else:
			self.emoji = Image.open("assets/images/noWinner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0, 0, 500, 310, fill='#582831')
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="Has perdido", fill="white", font=('Verdana 14'))
			self.textWinner_widget2 = self.canvasBody.create_text(185, 105, text=f"Derrota vs {opponent}", fill="white", font=('Verdana 11'))
		self.emojiPNG = self.emoji.convert(mode="RGBA")
		self.emojiResized_image = self.emojiPNG.resize((80, 80), Image.LANCZOS)
		self.emojiTK = ImageTk.PhotoImage(self.emojiResized_image)
		self.emoji_widget = self.canvasBody.create_image(420, 85, image=self.emojiTK)
		self.buttonHide = tk.Button(self, text='¡Seguir jugando!', anchor="center")
		self.buttonHide.configure(width=23, height=2, relief=tk.FLAT, command=self.screenNormalEnd)
		self.buttonHide_widget = self.canvasBody.create_window(250, 200, window=self.buttonHide)

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
		self.button1_widget = self.canvasBody.create_window(250, 200, window=self.auxButton1)
		self.labelUsername = tk.Label(self, text="Oponente (usuario PS)", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelUsername_widget = self.canvasBody.create_window(250, 22, window=self.labelUsername)
		self.labelTeam = tk.Label(self, text="Equipo del agente", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelTeam_widget = self.canvasBody.create_window(250, 82, window=self.labelTeam)
		self.strName_widget = self.canvasBody.create_window(250, 48, window=self.auxNameEntry)
		self.listbox_widget = self.canvasBody.create_window(250, 108, window=self.auxCbbTeam)
		self.meta_widget = self.canvasBody.create_text(250, 148, text=self.meta_text, fill="#888888", font=('Verdana 8'), justify=tk.CENTER)

class Modulo2(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent, bg=BG)
		self.auxController = controller
		self.cfg = getattr(controller, 'ui_config', {}) or {}
		self.p = None
		self.q = None
		self.isBattle = False
		self.isWinner = False
		self.nameOpponent = ""
		self.canvasHead = tk.Canvas(self, width=500, height=50, bg=BG, highlightthickness=0)
		self.canvasHead.create_text(250, 25, text="Módulo de Análisis", fill="white", font=('Verdana 15 bold'))
		self.canvasHead.pack()
		self.canvasBody = tk.Canvas(self, width=500, height=310, bg=BG, highlightthickness=0)
		self.canvasBody.create_rectangle(0, 0, 310, 259, outline='white', width=3)
		self.canvasBody.pack()
		self.startBattle = tk.Button(self, text='Buscar Ladder', anchor="center", command=self.IniciarBatalla)
		self.startBattle.configure(width=16, height=1, relief=tk.FLAT, bg="#1a1a1a", fg=FG, activebackground=ACCENT)
		self.startBattle_widget = self.canvasBody.create_window(400, 215, window=self.startBattle)
		self.auxStartBattle = self.startBattle
		self.help = tk.Button(self, text='Ayuda', anchor="center", command=self.MostrarOcultarDatos)
		self.help.configure(width=16, height=1, relief=tk.FLAT, bg="#1a1a1a", fg=FG, activebackground=ACCENT)
		self.help_widget = self.canvasBody.create_window(400, 175, window=self.help)
		self.help.config(state='disabled')
		self.auxHelp = self.help
		self.labelUsername = tk.Label(self, text="Usuario PS", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelUsername_widget = self.canvasBody.create_window(400, 12, window=self.labelUsername)
		self.labelPassword = tk.Label(self, text="Contraseña", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelPassword_widget = self.canvasBody.create_window(400, 58, window=self.labelPassword)
		self.labelTeam = tk.Label(self, text="Equipo", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelTeam_widget = self.canvasBody.create_window(400, 104, window=self.labelTeam)
		self.strName = tk.StringVar(value=self.cfg.get('ps_username', '') or '')
		self.nameEntry = ttk.Entry(self, textvariable=self.strName, width=20)
		self.strName_widget = self.canvasBody.create_window(400, 35, window=self.nameEntry)
		self.auxNameEntry = self.nameEntry
		self.strPassword = tk.StringVar()
		self.passwordEntry = ttk.Entry(self, show="*", textvariable=self.strPassword, width=20)
		self.strPassword_widget = self.canvasBody.create_window(400, 81, window=self.passwordEntry)
		self.auxPasswordEntry = self.passwordEntry
		self.textVar = tk.StringVar(self)
		self.textVar.set('')
		self.path = os.path.join('teams', 'teams', self.cfg.get('team_dir', 'gen8/vgc2022'))
		try:
			self.listBoxValues = sorted(os.listdir(self.path))
		except FileNotFoundError:
			self.listBoxValues = []
		self.cbbTeam = ttk.Combobox(self, textvariable=self.textVar, state="readonly", values=self.listBoxValues, width=17)
		self.listbox_widget = self.canvasBody.create_window(400, 127, window=self.cbbTeam)
		self.auxCbbTeam = self.cbbTeam
		self.meta_text = f"{self.cfg.get('pokemon_mode', '?')} · bot:{self.cfg.get('battle_bot', 'random')}"
		self.meta_widget = self.canvasBody.create_text(400, 152, text=self.meta_text, fill="#888888", font=('Verdana 7'), width=180)
		self.status_widget = self.canvasBody.create_text(400, 250, text="Ladder · SEARCH_LADDER", fill="#666666", font=('Verdana 7'))
		self.exitPNG = Image.open("assets/images/exit.png")
		self.newExitPNG = self.exitPNG.convert(mode="RGBA")
		self.exitResized = self.newExitPNG.resize((40, 40), Image.LANCZOS)
		self.exitTK = ImageTk.PhotoImage(self.exitResized)
		self.buttonExit = tk.Button(self, image=self.exitTK, command=lambda: self.auxController.show_frame("StartPage"), bg=BG, activebackground=BG, highlightthickness=0, bd=0)
		self.canvasHead.create_window(470, 28, window=self.buttonExit)
		self.barChartPNG = Image.open("assets/images/barChart.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((200, 200), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(160, 125, image=self.barChartTK)
		self.infoPNG = Image.open("assets/images/info.png")
		self.newInfoPNG = self.infoPNG.convert(mode="RGBA")
		self.infoResized = self.newInfoPNG.resize((38, 38), Image.LANCZOS)
		self.infoTK = ImageTk.PhotoImage(self.infoResized)
		self.buttonInfo = tk.Button(self, image=self.infoTK, command=lambda: os.system('"docs/pdfs/Módulo de Análisis.pdf"'), bg=BG, activebackground=BG, highlightthickness=0, bd=0)
		self.canvasHead.create_window(28, 28, window=self.buttonInfo)

	def _show_error(self, text):
		if hasattr(self, "textIncorrect"):
			try:
				self.canvasBody.delete(self.textIncorrect)
			except tk.TclError:
				pass
		self.textIncorrect = self.canvasBody.create_text(155, 280, text=text, fill="orange", font=('Verdana 8'))

	def _unlock_form(self):
		self.nameEntry.config(state='normal')
		self.passwordEntry.config(state='normal')
		self.cbbTeam.config(state='readonly')
		self.startBattle['state'] = tk.NORMAL
		self.startBattle.configure(text='Buscar Ladder')
		self.isBattle = False
		try:
			self.canvasBody.itemconfig(self.status_widget, text="Ladder · SEARCH_LADDER")
		except tk.TclError:
			pass

	def IniciarBatalla(self):
		if hasattr(self, "textIncorrect"):
			try:
				self.canvasBody.delete(self.textIncorrect)
			except tk.TclError:
				pass
		username = self.nameEntry.get().strip()
		password = self.passwordEntry.get()
		teamname = self.textVar.get()
		if not username:
			self._show_error("Indica usuario de Pokémon Showdown")
			return
		if not password:
			self._show_error("Indica la contraseña de la cuenta")
			return
		if not teamname:
			self._show_error("No se ha seleccionado un equipo")
			return
		if not self.listBoxValues:
			self._show_error(f"Sin equipos en {self.path}")
			return
		self.q = multiprocessing.Queue()
		self.username = username
		self.password = password
		self.teamname = teamname
		botmode = "SEARCH_LADDER"
		self.nameEntry.config(state='disabled')
		self.passwordEntry.config(state='disabled')
		self.cbbTeam.config(state='disabled')
		self.startBattle['state'] = tk.DISABLED
		self.startBattle.configure(text='Buscando Oponente')
		try:
			self.canvasBody.itemconfig(self.status_widget, text="Buscando en ladder...")
		except tk.TclError:
			pass
		self.p = AsyncDownload(self.q, self.username, self.password, self.teamname, botmode, 1)
		self.p.start()
		self.isBattle = False
		self.monitor()

	def monitor(self):
		if self.p is not None and self.p.is_alive():
			self.after(100, lambda: self.monitor())
			try:
				self.cola = self.q.get(0)
				code = self.cola[0]
				if code == 2:
					self.isBattle = True
					try:
						self.canvasBody.itemconfig(self.status_widget, text="Turno en curso")
					except tk.TclError:
						pass
					self.labelMove.configure(text="Movimientos", bg=BG, font=('Verdana 12 bold'), fg="white")
					self.LabelMove_widget = self.canvasBody.create_window(80, 25, window=self.labelMove)
					self.text1 = self.cola[1]
					self.color1 = self.cola[2]
					self.text2 = self.cola[3]
					self.color2 = self.cola[4]
					self.text3 = self.cola[5]
					self.color3 = self.cola[6]
					self.text4 = self.cola[7]
					self.color4 = self.cola[8]
					self.text5 = self.cola[9] if len(self.cola) > 9 else 0
					self.color5 = self.cola[10] if len(self.cola) > 10 else "#565828"
					self.text6 = self.cola[11] if len(self.cola) > 11 else 0
					self.color6 = self.cola[12] if len(self.cola) > 12 else "#582831"
					self.labelR = tk.Label(self, text="Rendimiento", bg=BG, font=('Verdana 12 bold'), fg="white")
					self.LabelR_widget = self.canvasBody.create_window(80, 160, window=self.labelR)
					self.labelR1 = tk.Label(self, text="Último turno", bg=BG, font=('Verdana 8 bold'), fg="white")
					self.LabelR1_widget = self.canvasBody.create_window(68, 233, window=self.labelR1)
					self.labelR2 = tk.Label(self, text="Histórico", bg=BG, font=('Verdana 8 bold'), fg="white")
					self.LabelR2_widget = self.canvasBody.create_window(188, 233, window=self.labelR2)
					if self.help.config('text')[-1] == 'Ayuda':
						self.color1_widget = self.canvasBody.create_rectangle(28, 45, 148, 85, fill="#ffffff", outline="#ffffff")
						self.color2_widget = self.canvasBody.create_rectangle(160, 45, 280, 85, fill="#ffffff", outline="#ffffff")
						self.color3_widget = self.canvasBody.create_rectangle(28, 97, 148, 137, fill="#ffffff", outline="#ffffff")
						self.color4_widget = self.canvasBody.create_rectangle(160, 97, 280, 137, fill="#ffffff", outline="#ffffff")
						self.color5_widget = self.canvasBody.create_rectangle(28, 180, 148, 220, fill="#ffffff", outline="#ffffff")
						self.color6_widget = self.canvasBody.create_rectangle(160, 180, 280, 220, fill="#ffffff", outline="#ffffff")
						self.text1_widget = self.canvasBody.create_text(88, 65, text="???", fill="black", font=('Verdana 10'))
						self.text2_widget = self.canvasBody.create_text(220, 65, text="???", fill="black", font=('Verdana 10'))
						self.text3_widget = self.canvasBody.create_text(88, 117, text="???", fill="black", font=('Verdana 10'))
						self.text4_widget = self.canvasBody.create_text(220, 117, text="???", fill="black", font=('Verdana 10'))
						self.text5_widget = self.canvasBody.create_text(83, 200, text="???", fill="black", font=('Verdana 10'))
						self.text6_widget = self.canvasBody.create_text(215, 200, text="???", fill="black", font=('Verdana 10'))
					else:
						self.color1_widget = self.canvasBody.create_rectangle(28, 45, 148, 85, fill=self.color1, outline=self.color1)
						self.color2_widget = self.canvasBody.create_rectangle(160, 45, 280, 85, fill=self.color2, outline=self.color2)
						self.color3_widget = self.canvasBody.create_rectangle(28, 97, 148, 137, fill=self.color3, outline=self.color3)
						self.color4_widget = self.canvasBody.create_rectangle(160, 97, 280, 137, fill=self.color4, outline=self.color4)
						self.color5_widget = self.canvasBody.create_rectangle(28, 180, 148, 220, fill=self.color5, outline=self.color5)
						self.color6_widget = self.canvasBody.create_rectangle(160, 180, 280, 220, fill=self.color6, outline=self.color6)
						self.text1_widget = self.canvasBody.create_text(88, 65, text=self.text1, fill="black", font=('Verdana 10'))
						self.text2_widget = self.canvasBody.create_text(220, 65, text=self.text2, fill="black", font=('Verdana 10'))
						self.text3_widget = self.canvasBody.create_text(88, 117, text=self.text3, fill="black", font=('Verdana 8'))
						self.text4_widget = self.canvasBody.create_text(220, 117, text=self.text4, fill="black", font=('Verdana 8'))
						self.text5_widget = self.canvasBody.create_text(83, 200, text=math.floor(self.text5), fill="white", font=('Verdana 12'))
						self.text6_widget = self.canvasBody.create_text(215, 200, text=math.floor(self.text6), fill="white", font=('Verdana 12'))
				elif code == 1:
					self.namePokemon = self.cola
					self.text1 = self.namePokemon[1]
					self.text2 = self.namePokemon[2]
					self.text3 = self.namePokemon[3]
					self.text4 = self.namePokemon[4]
					self.text5 = ""
					self.text6 = ""
					self.help['state'] = tk.NORMAL
					self.auxHelp = self.help
					try:
						self.canvasBody.itemconfig(self.status_widget, text="Team Preview")
					except tk.TclError:
						pass
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
					self.text5_widget = self.canvasBody.create_text(83, 200, text=self.text5, fill="white", font=('Verdana 12'))
					self.text6_widget = self.canvasBody.create_text(215, 200, text=self.text6, fill="white", font=('Verdana 12'))
				elif code == 0:
					self.startBattle.configure(text='Batalla en Curso')
					try:
						self.canvasBody.itemconfig(self.status_widget, text="Batalla en curso")
					except tk.TclError:
						pass
					self.canvasBody.delete(self.barChart_widget)
					self.labelMove = tk.Label(self, text="Vista Previa del Equipo", bg=BG, font=('Verdana 12 bold'), fg="white")
					self.LabelMove_widget = self.canvasBody.create_window(130, 25, window=self.labelMove)
					self.color1 = '#ffffff'
					self.color2 = '#ffffff'
					self.color3 = '#ffffff'
					self.color4 = '#ffffff'
					self.color5 = '#000000'
					self.color6 = '#000000'
					self.color1_widget = self.canvasBody.create_rectangle(28, 45, 148, 85, fill=self.color1, outline=self.color1)
					self.color2_widget = self.canvasBody.create_rectangle(160, 45, 280, 85, fill=self.color2, outline=self.color2)
					self.color3_widget = self.canvasBody.create_rectangle(28, 97, 148, 137, fill=self.color3, outline=self.color3)
					self.color4_widget = self.canvasBody.create_rectangle(160, 97, 280, 137, fill=self.color4, outline=self.color4)
					self.color5_widget = self.canvasBody.create_rectangle(28, 180, 148, 220, fill=self.color5, outline=self.color5)
					self.color6_widget = self.canvasBody.create_rectangle(160, 180, 280, 220, fill=self.color6, outline=self.color6)
					self.text1_widget = self.canvasBody.create_text(88, 65, text="Calculando...", fill="black", font=('Verdana 10'))
					self.text2_widget = self.canvasBody.create_text(220, 65, text="Calculando...", fill="black", font=('Verdana 10'))
					self.text3_widget = self.canvasBody.create_text(88, 117, text="Calculando...", fill="black", font=('Verdana 10'))
					self.text4_widget = self.canvasBody.create_text(220, 117, text="Calculando...", fill="black", font=('Verdana 10'))
				elif code == -1:
					self._unlock_form()
					self.isWinner = self.cola[1]
					self.nameOpponent = self.cola[3] if len(self.cola) > 3 else self.cola[2]
					self.endBattle()
				elif code == -2:
					self._unlock_form()
					self._show_error("¡El oponente rechazó / canceló la búsqueda!")
				elif code == -3:
					self._unlock_form()
					self._show_error("¡El equipo no cumple las reglas del formato!")
				elif code == -4:
					self._unlock_form()
					self._show_error("¡Usuario no encontrado!")
				elif code == -5:
					self._unlock_form()
					self._show_error("¡Usuario o contraseña incorrectos!")
				elif code == -6:
					self._unlock_form()
					self._show_error("¡Error del sistema en la batalla ladder!")
			except queue.Empty:
				pass
			except Exception as e:
				print(f"[WARN]     Modulo2.monitor: {e}")
		elif self.p is not None:
			self.p.join(timeout=0.1)

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
		status = self.startBattle.config('text')[-1]
		if status not in ("Batalla en Curso", "Buscando Oponente"):
			self.exitWindows()
			return
		self.isCloseWindows = True
		self.canvasBody.delete("all")
		self.rectangle1 = self.canvasBody.create_rectangle(0, 0, 500, 310, fill='#4e2858')
		self.emoji = Image.open("assets/images/close.png")
		self.emojiPNG = self.emoji.convert(mode="RGBA")
		self.emojiResized_image = self.emojiPNG.resize((80, 80), Image.LANCZOS)
		self.emojiTK = ImageTk.PhotoImage(self.emojiResized_image)
		self.emoji_widget = self.canvasBody.create_image(420, 85, image=self.emojiTK)
		self.textClose_widget1 = self.canvasBody.create_text(185, 75, text="¡La batalla está en curso!", fill="white", font=('Verdana 14'))
		self.textClose_widget2 = self.canvasBody.create_text(185, 100, text="¿Está seguro de salir?", fill="white", font=('Verdana 11'))
		self.buttonYes = tk.Button(self, text='Sí', anchor="center")
		self.buttonYes.configure(width=23, height=2, relief=tk.FLAT, command=self.exitWindows)
		self.buttonYes_widget = self.canvasBody.create_window(150, 200, window=self.buttonYes)
		self.buttonNo = tk.Button(self, text='No', anchor="center")
		self.buttonNo.configure(width=23, height=2, relief=tk.FLAT, command=self.screenNormalClose)
		self.buttonNo_widget = self.canvasBody.create_window(350, 200, window=self.buttonNo)

	def exitWindows(self):
		sys.exit(0)

	def endBattle(self):
		self.canvasBody.delete("all")
		opponent = self.nameOpponent or "oponente"
		if self.isWinner:
			self.emoji = Image.open("assets/images/winner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0, 0, 500, 310, fill='#325928')
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="¡Felicidades!", fill="white", font=('Verdana 14'))
			self.textWinner_widget2 = self.canvasBody.create_text(185, 105, text=f"Victoria vs {opponent}", fill="white", font=('Verdana 11'))
		else:
			self.emoji = Image.open("assets/images/noWinner.png")
			self.rectangle1 = self.canvasBody.create_rectangle(0, 0, 500, 310, fill='#582831')
			self.textWinner_widget1 = self.canvasBody.create_text(185, 75, text="Has perdido", fill="white", font=('Verdana 14'))
			self.textWinner_widget2 = self.canvasBody.create_text(185, 105, text=f"Derrota vs {opponent}", fill="white", font=('Verdana 11'))
		self.emojiPNG = self.emoji.convert(mode="RGBA")
		self.emojiResized_image = self.emojiPNG.resize((80, 80), Image.LANCZOS)
		self.emojiTK = ImageTk.PhotoImage(self.emojiResized_image)
		self.emoji_widget = self.canvasBody.create_image(420, 85, image=self.emojiTK)
		self.buttonHide = tk.Button(self, text='¡Seguir analizando!', anchor="center")
		self.buttonHide.configure(width=23, height=2, relief=tk.FLAT, command=self.screenNormalEnd)
		self.buttonHide_widget = self.canvasBody.create_window(250, 200, window=self.buttonHide)

	def screenNormalClose(self):
		self.canvasBody.delete(self.buttonYes_widget)
		self.canvasBody.delete(self.buttonNo_widget)
		self.canvasBody.delete(self.textClose_widget1)
		self.canvasBody.delete(self.textClose_widget2)
		self.canvasBody.delete(self.emoji_widget)
		self.canvasBody.delete(self.rectangle1)
		if self.isBattle:
			self.labelMove = tk.Label(self, text="Movimientos", bg=BG, font=('Verdana 12 bold'), fg="white")
			self.LabelMove_widget = self.canvasBody.create_window(80, 25, window=self.labelMove)
			self.labelR = tk.Label(self, text="Rendimiento", bg=BG, font=('Verdana 12 bold'), fg="white")
			self.LabelR_widget = self.canvasBody.create_window(80, 160, window=self.labelR)
			self.labelR1 = tk.Label(self, text="Último turno", bg=BG, font=('Verdana 8 bold'), fg="white")
			self.LabelR1_widget = self.canvasBody.create_window(68, 233, window=self.labelR1)
			self.labelR2 = tk.Label(self, text="Histórico", bg=BG, font=('Verdana 8 bold'), fg="white")
			self.LabelR2_widget = self.canvasBody.create_window(188, 233, window=self.labelR2)
		else:
			self.labelMove = tk.Label(self, text="Vista Previa del Equipo", bg=BG, font=('Verdana 12 bold'), fg="white")
			self.LabelMove_widget = self.canvasBody.create_window(130, 25, window=self.labelMove)
		if self.isBattle:
			if self.help.config('text')[-1] == 'Ayuda':
				self.color1_widget = self.canvasBody.create_rectangle(28, 45, 148, 85, fill="#ffffff", outline="#ffffff")
				self.color2_widget = self.canvasBody.create_rectangle(160, 45, 280, 85, fill="#ffffff", outline="#ffffff")
				self.color3_widget = self.canvasBody.create_rectangle(28, 97, 148, 137, fill="#ffffff", outline="#ffffff")
				self.color4_widget = self.canvasBody.create_rectangle(160, 97, 280, 137, fill="#ffffff", outline="#ffffff")
				self.color5_widget = self.canvasBody.create_rectangle(28, 180, 148, 220, fill="#ffffff", outline="#ffffff")
				self.color6_widget = self.canvasBody.create_rectangle(160, 180, 280, 220, fill="#ffffff", outline="#ffffff")
				self.text1_widget = self.canvasBody.create_text(88, 65, text="???", fill="black", font=('Verdana 10'))
				self.text2_widget = self.canvasBody.create_text(220, 65, text="???", fill="black", font=('Verdana 10'))
				self.text3_widget = self.canvasBody.create_text(88, 117, text="???", fill="black", font=('Verdana 10'))
				self.text4_widget = self.canvasBody.create_text(220, 117, text="???", fill="black", font=('Verdana 10'))
				self.text5_widget = self.canvasBody.create_text(83, 200, text="???", fill="black", font=('Verdana 10'))
				self.text6_widget = self.canvasBody.create_text(215, 200, text="???", fill="black", font=('Verdana 10'))
			else:
				self.color1_widget = self.canvasBody.create_rectangle(28, 45, 148, 85, fill=self.color1, outline=self.color1)
				self.color2_widget = self.canvasBody.create_rectangle(160, 45, 280, 85, fill=self.color2, outline=self.color2)
				self.color3_widget = self.canvasBody.create_rectangle(28, 97, 148, 137, fill=self.color3, outline=self.color3)
				self.color4_widget = self.canvasBody.create_rectangle(160, 97, 280, 137, fill=self.color4, outline=self.color4)
				self.color5_widget = self.canvasBody.create_rectangle(28, 180, 148, 220, fill=self.color5, outline=self.color5)
				self.color6_widget = self.canvasBody.create_rectangle(160, 180, 280, 220, fill=self.color6, outline=self.color6)
				self.text1_widget = self.canvasBody.create_text(88, 65, text=self.text1, fill="black", font=('Verdana 10'))
				self.text2_widget = self.canvasBody.create_text(220, 65, text=self.text2, fill="black", font=('Verdana 10'))
				self.text3_widget = self.canvasBody.create_text(88, 117, text=self.text3, fill="black", font=('Verdana 8'))
				self.text4_widget = self.canvasBody.create_text(220, 117, text=self.text4, fill="black", font=('Verdana 8'))
				self.text5_widget = self.canvasBody.create_text(83, 200, text=math.floor(self.text5), fill="white", font=('Verdana 12'))
				self.text6_widget = self.canvasBody.create_text(215, 200, text=math.floor(self.text6), fill="white", font=('Verdana 12'))
		else:
			if self.help.config('text')[-1] == 'Ayuda':
				self.color1_widget = self.canvasBody.create_rectangle(28, 45, 148, 85, fill="#ffffff", outline="#ffffff")
				self.color2_widget = self.canvasBody.create_rectangle(160, 45, 280, 85, fill="#ffffff", outline="#ffffff")
				self.color3_widget = self.canvasBody.create_rectangle(28, 97, 148, 137, fill="#ffffff", outline="#ffffff")
				self.color4_widget = self.canvasBody.create_rectangle(160, 97, 280, 137, fill="#ffffff", outline="#ffffff")
				self.text1_widget = self.canvasBody.create_text(88, 65, text="???", fill="black", font=('Verdana 10'))
				self.text2_widget = self.canvasBody.create_text(220, 65, text="???", fill="black", font=('Verdana 10'))
				self.text3_widget = self.canvasBody.create_text(88, 117, text="???", fill="black", font=('Verdana 10'))
				self.text4_widget = self.canvasBody.create_text(220, 117, text="???", fill="black", font=('Verdana 10'))
			else:
				self.color1_widget = self.canvasBody.create_rectangle(28, 45, 148, 85, fill=self.color1, outline=self.color1)
				self.color2_widget = self.canvasBody.create_rectangle(160, 45, 280, 85, fill=self.color2, outline=self.color2)
				self.color3_widget = self.canvasBody.create_rectangle(28, 97, 148, 137, fill=self.color3, outline=self.color3)
				self.color4_widget = self.canvasBody.create_rectangle(160, 97, 280, 137, fill=self.color4, outline=self.color4)
				self.text1_widget = self.canvasBody.create_text(88, 65, text=self.text1, fill="black", font=('Verdana 10'))
				self.text2_widget = self.canvasBody.create_text(220, 65, text=self.text2, fill="black", font=('Verdana 10'))
				self.text3_widget = self.canvasBody.create_text(88, 117, text=self.text3, fill="black", font=('Verdana 10'))
				self.text4_widget = self.canvasBody.create_text(220, 117, text=self.text4, fill="black", font=('Verdana 10'))
		self.screenNormal()

	def screenNormalEnd(self):
		self.canvasBody.delete("all")
		self.barChartPNG = Image.open("assets/images/barChart.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((200, 200), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(160, 125, image=self.barChartTK)
		self.startBattle = tk.Button(self, text='Buscar Ladder', anchor="center", command=self.IniciarBatalla)
		self.startBattle.configure(width=16, height=1, relief=tk.FLAT, bg="#1a1a1a", fg=FG, activebackground=ACCENT)
		self.startBattle_widget = self.canvasBody.create_window(400, 215, window=self.startBattle)
		self.auxStartBattle = self.startBattle
		self.help = tk.Button(self, text='Ayuda', anchor="center", command=self.MostrarOcultarDatos)
		self.help.configure(width=16, height=1, relief=tk.FLAT, bg="#1a1a1a", fg=FG, activebackground=ACCENT)
		self.help_widget = self.canvasBody.create_window(400, 175, window=self.help)
		self.help.config(state='disabled')
		self.auxHelp = self.help
		self.screenNormal()

	def screenNormal(self):
		self.canvasBody.create_rectangle(0, 0, 310, 259, outline='white', width=3)
		self.startBattle_widget = self.canvasBody.create_window(400, 215, window=self.auxStartBattle)
		self.auxStartBattle = self.startBattle
		self.help_widget = self.canvasBody.create_window(400, 175, window=self.auxHelp)
		self.labelUsername = tk.Label(self, text="Usuario PS", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelUsername_widget = self.canvasBody.create_window(400, 12, window=self.labelUsername)
		self.labelPassword = tk.Label(self, text="Contraseña", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelPassword_widget = self.canvasBody.create_window(400, 58, window=self.labelPassword)
		self.labelTeam = tk.Label(self, text="Equipo", bg=BG, font=('Verdana 8 bold'), fg="white")
		self.LabelTeam_widget = self.canvasBody.create_window(400, 104, window=self.labelTeam)
		self.strName_widget = self.canvasBody.create_window(400, 35, window=self.auxNameEntry)
		self.strPassword_widget = self.canvasBody.create_window(400, 81, window=self.auxPasswordEntry)
		self.listbox_widget = self.canvasBody.create_window(400, 127, window=self.auxCbbTeam)
		self.meta_widget = self.canvasBody.create_text(400, 152, text=self.meta_text, fill="#888888", font=('Verdana 7'), width=180)
		self.status_widget = self.canvasBody.create_text(400, 250, text="Ladder · SEARCH_LADDER", fill="#666666", font=('Verdana 7'))
		self.exitPNG = Image.open("assets/images/exit.png")
		self.newExitPNG = self.exitPNG.convert(mode="RGBA")
		self.exitResized = self.newExitPNG.resize((40, 40), Image.LANCZOS)
		self.exitTK = ImageTk.PhotoImage(self.exitResized)
		self.buttonExit = tk.Button(self, image=self.exitTK, command=lambda: self.auxController.show_frame("StartPage"), bg=BG, activebackground=BG, highlightthickness=0, bd=0)
		self.canvasHead.create_window(470, 28, window=self.buttonExit)
		self.infoPNG = Image.open("assets/images/info.png")
		self.newInfoPNG = self.infoPNG.convert(mode="RGBA")
		self.infoResized = self.newInfoPNG.resize((38, 38), Image.LANCZOS)
		self.infoTK = ImageTk.PhotoImage(self.infoResized)
		self.buttonInfo = tk.Button(self, image=self.infoTK, command=lambda: os.system('"docs/pdfs/Módulo de Análisis.pdf"'), bg=BG, activebackground=BG, highlightthickness=0, bd=0)
		self.canvasHead.create_window(28, 28, window=self.buttonInfo)

class Modulo3(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent, bg=BG)
		self.auxController = controller
		# Canvas Head
		self.canvasHead = tk.Canvas(self, width=500, height=50, bg=BG, highlightthickness=0)
		self.canvasHead.place(x=0, y=0, anchor = tk.CENTER)
		self.canvasHead.create_text(250, 25, text="Módulo de Entrenamiento", fill="white", font=('Verdana 15 bold'))
		self.canvasHead.pack()
		# Canvas Body
		self.canvasBody = tk.Canvas(self, width=500, height=259, bg=BG, highlightthickness=0)
		self.canvasBody.create_rectangle(0,0,310,259, outline='white', width=3)
		self.canvasBody.pack()
		# Label Seleccionar Equipo
		self.labelSelectTeam = tk.Label(self,text="Seleccionar Equipo", bg = "#000000", font=('Verdana 8 bold'), fg="white")
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
		self.labelSelectO = tk.Label(self,text="Subir Oponentes", bg = "#000000", font=('Verdana 8 bold'), fg="white")
		self.LabelSelectOpponents_widget = self.canvasBody.create_window(400, 90, window=self.labelSelectO)
		self.auxLabelSelectOpponents_widget = self.LabelSelectOpponents_widget
		# Botón Seleccionar Oponentes
		self.selectOpponents = tk.Button(self, text='Subir Oponentes', anchor="center", command = self.SeleccionarOponentes)
		self.selectOpponents.configure(width = 20, height = 1, relief = tk.FLAT)
		self.selectOpponents_widget = self.canvasBody.create_window(400, 115, window=self.selectOpponents)
		self.selectOpponents['state'] = tk.DISABLED
		self.auxSelectOpponents_widget = self.selectOpponents_widget
		# Label Cantidad Batallas
		self.labelSelectCB = tk.Label(self,text="Cantidad de Interacciones", bg = "#000000", font=('Verdana 8 bold'), fg="white")
		self.LabelSelectInter_widget = self.canvasBody.create_window(400, 140, window=self.labelSelectCB)
		self.auxLabelSelectInter_widget = self.LabelSelectInter_widget
		# Scale
		v = tk.DoubleVar()
		self.scale = tk.Scale(self, variable = v, from_ = 50, to = 80000, resolution = 10, orient = tk.HORIZONTAL, length = 140, fg="#ffffff", font="#ffffff", bg="#000000", activebackground="#000000", highlightbackground="#000000",highlightcolor="#ffffff",troughcolor="#333333")
		self.Scale_widget = self.canvasBody.create_window(399, 175, window=self.scale)
		self.auxScale_widget = self.Scale_widget
		# Botón Iniciar Entrenamiento
		self.startTraining = tk.Button(self, text='Iniciar Entrenamiento', anchor="center", command = self.IniciarEntrenamiento)
		self.startTraining.configure(width = 20, height = 1)
		self.startTraining_widget = self.canvasBody.create_window(400, 225, window=self.startTraining)
		self.auxstartTraining_widget = self.startTraining_widget
		# IMG Exit
		self.exitPNG = Image.open("assets/images/exit.png")
		self.newExitPNG = self.exitPNG.convert(mode="RGBA")
		self.exitResized = self.newExitPNG.resize((40,40), Image.LANCZOS)
		self.exitTK = ImageTk.PhotoImage(self.exitResized)
		self.buttonExit = tk.Button(self, image=self.exitTK, command=lambda: self.auxController.show_frame("StartPage"), bg="#000000", activebackground="#000000", highlightthickness = 0, bd = 0)
		self.buttonExit_widget = self.canvasHead.create_window(470, 28, window=self.buttonExit)
		# IMG Bar Chart
		self.barChartPNG = Image.open("assets/images/chartIncreasing.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((200,200), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(160,125,image=self.barChartTK)
		# IMG Info
		self.infoPNG = Image.open("assets/images/info.png")
		self.newInfoPNG = self.infoPNG.convert(mode="RGBA")
		self.infoResized = self.newInfoPNG.resize((38,38), Image.LANCZOS)
		self.infoTK = ImageTk.PhotoImage(self.infoResized)
		self.buttonInfo = tk.Button(self, image=self.infoTK, command=lambda: os.system('"docs/pdfs/Módulo de Entrenamiento.pdf"'), bg="#000000", activebackground="#000000", highlightthickness = 0, bd = 0)
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
			self.labelSelect = tk.Label(self,text= str(int(self.progreso['value'])) + " %", bg = "#000000", font=('Verdana 8 bold'), fg="white")
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
		self.labelR = tk.Label(self,text="Recomendaciones", bg = "#000000", font=('Verdana 8 bold'), fg="white")
		self.LabelR_widget = self.canvasBody.create_window(400, 15, window=self.labelR)
		self.barChartPNG = Image.open("assets/images/tornadus-incarnate.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((90,90), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(400-50+10,115-50+5,image=self.barChartTK)
		self.barChartPNG2 = Image.open("assets/images/zapdos.png")
		self.newBarChartPNG2 = self.barChartPNG2.convert(mode="RGBA")
		self.barChartResized2 = self.newBarChartPNG2.resize((90,90), Image.LANCZOS)
		self.barChartTK2 = ImageTk.PhotoImage(self.barChartResized2)
		self.barChart_widget2 = self.canvasBody.create_image(490-50+10,115-50+5,image=self.barChartTK2)
		self.barChartPNG3 = Image.open("assets/images/thundurus-incarnate.png")
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
		self.labelR = tk.Label(self,text="Recomendaciones", bg = "#000000", font=('Verdana 8 bold'), fg="white")
		self.LabelR_widget = self.canvasBody.create_window(400, 15, window=self.labelR)
		self.barChartPNG = Image.open("assets/images/landorus-therian.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((90,90), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(400-50+10,115-50+5,image=self.barChartTK)
		self.barChartPNG2 = Image.open("assets/images/arcanine.png")
		self.newBarChartPNG2 = self.barChartPNG2.convert(mode="RGBA")
		self.barChartResized2 = self.newBarChartPNG2.resize((90,90), Image.LANCZOS)
		self.barChartTK2 = ImageTk.PhotoImage(self.barChartResized2)
		self.barChart_widget2 = self.canvasBody.create_image(490-50+10,115-50+5,image=self.barChartTK2)
		self.barChartPNG3 = Image.open("assets/images/hitmontop.png")
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
		self.labelR = tk.Label(self,text="Recomendaciones", bg = "#000000", font=('Verdana 8 bold'), fg="white")
		self.LabelR_widget = self.canvasBody.create_window(400, 15, window=self.labelR)
		self.barChartPNG = Image.open("assets/images/kyogre.png")
		self.newBarChartPNG = self.barChartPNG.convert(mode="RGBA")
		self.barChartResized = self.newBarChartPNG.resize((90,90), Image.LANCZOS)
		self.barChartTK = ImageTk.PhotoImage(self.barChartResized)
		self.barChart_widget = self.canvasBody.create_image(400-50+10,115-50,image=self.barChartTK)
		self.barChartPNG2 = Image.open("assets/images/groudon.png")
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
		self.top.configure(bg=BG)
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
		self.canvasHead2 = tk.Canvas(self.top, width=500, height=50, bg=BG, highlightthickness=0)
		self.canvasHead2.place(x=0, y=0, anchor = tk.CENTER)
		self.canvasHead2.create_text(270, 25, text="Módulo de Construcción de Equipos", fill="white", font=('Verdana 15 bold'))
		self.canvasHead2.pack()
		# Canvas Body
		self.canvasBody2 = tk.Canvas(self.top, width=500, height=259, bg=BG, highlightthickness=0)
		self.canvasBody2.pack()
		# Label Paste Team
		self.pasteTeam = tk.Label(self.top, text="Pegue su equipo usando el formato de Pokémon Showdown!", bg = "#000000", font=('Verdana 8 bold'), fg="white")
		self.PasteTeam_widget = self.canvasBody2.create_window(205, 18, window=self.pasteTeam)
		# Text
		self.textTeam = tk.Text(self.top, height = 9, width = 60, bg="#111111", fg=FG, insertbackground=FG)
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
		self.infoPNG2 = Image.open("assets/images/info.png")
		self.newInfoPNG2 = self.infoPNG2.convert(mode="RGBA")
		self.infoResized2 = self.newInfoPNG2.resize((38,38), Image.LANCZOS)
		self.infoTK2 = ImageTk.PhotoImage(self.infoResized2)
		self.buttonInfo2 = tk.Button(self.top, image=self.infoTK2, command=lambda: os.system('"docs/pdfs/Módulo de Construcción de Equipos.pdf"'), bg="#000000", activebackground="#000000", highlightthickness = 0, bd = 0)
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
			self.top2.configure(bg=BG)
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
			self.canvasBody2 = tk.Canvas(self.top2, width=291, height=285+60, bg=BG, bd=0, highlightthickness=0)
			self.canvasBody2.pack()
			# Label Paste Team
			self.pasteTeam = tk.Label(self.top2, text="Seleccione los Pokémon que no formarán", bg = "#000000", borderwidth=0, font=('Verdana 8 bold'), fg="white")
			self.PasteTeam_widget = self.canvasBody2.create_window(140, 18, window=self.pasteTeam)
			self.pasteTeam = tk.Label(self.top2, text="parte de la evaluación", bg = "#000000", borderwidth=0, font=('Verdana 8 bold'), fg="white")
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
			self.pasteTeam = tk.Label(self.top2, text="Escriba el nombre del equipo", bg = "#000000", borderwidth=0, font=('Verdana 8 bold'), fg="white")
			self.PasteTeam_widget = self.canvasBody2.create_window(140, 246+15, window=self.pasteTeam)
			# Text Name Team
			self.textNameTeam = tk.Text(self.top2, height = 1, width = 30, bg="#111111", fg=FG, insertbackground=FG)
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
			self.buttonPkmn1.config(bg=BG)
			self.clicked1 = False

	def buttonPkmn2Select(self):
		if not self.clicked2:
			self.buttonPkmn2.config(bg="#599e47")
			self.clicked2 = True
		else:
			self.buttonPkmn2.config(bg=BG)
			self.clicked2 = False

	def buttonPkmn3Select(self):
		if not self.clicked3:
			self.buttonPkmn3.config(bg="#599e47")
			self.clicked3 = True
		else:
			self.buttonPkmn3.config(bg=BG)
			self.clicked3 = False

	def buttonPkmn4Select(self):
		if not self.clicked4:
			self.buttonPkmn4.config(bg="#599e47")
			self.clicked4 = True
		else:
			self.buttonPkmn4.config(bg=BG)
			self.clicked4 = False

	def buttonPkmn5Select(self):
		if not self.clicked5:
			self.buttonPkmn5.config(bg="#599e47")
			self.clicked5 = True
		else:
			self.buttonPkmn5.config(bg=BG)
			self.clicked5 = False

	def buttonPkmn6Select(self):
		if not self.clicked6:
			self.buttonPkmn6.config(bg="#599e47")
			self.clicked6 = True
		else:
			self.buttonPkmn6.config(bg=BG)
			self.clicked6 = False

	def exitWindows(self):
		os._exit(0)
		self.auxController.destroy()

if __name__ == "__main__":
    app = VGCApp()
    app.mainloop()