import tkinter as tk
import run
import subprocess

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
		# Frames
		self.frames = {}
		for F in (StartPage, Modulo1, Modulo2):
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

		button3 = tk.Button(self, text='Módulo de Retroalimentación de Equipos', anchor="center")
		button3.configure(width = 50, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button3_widget = canvasBody.create_window(70, 175, anchor=tk.NW, window=button3)

class Modulo1(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		# Canvas Head
		canvasHead = tk.Canvas(self, width=500, height=50, bg="#154e72")
		canvasHead.place(x=0, y=0, anchor = tk.CENTER)
		canvasHead.create_text(250, 25, text="Módulo de Batallas contra Agentes", fill="white", font=('Verdana 15 bold'))
		canvasHead.pack()
		# Canvas Body
		canvasBody = tk.Canvas(self, width=500, height=259, bg="#154e72")
		canvasBody.pack()

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
		canvasBody = tk.Canvas(self, width=500, height=259, bg="#154e72")
		canvasBody.create_rectangle(0,0,310,259, outline='white', width=3)
		canvasBody.pack()
		# Botón Iniciar Batalla
		button1 = tk.Button(self, text='Iniciar Batalla', anchor="center")
		button1.configure(width = 15, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button1_widget = canvasBody.create_window(405, 160, window=button1)
		# Botón Ayuda
		button2 = tk.Button(self, text='Ayuda', anchor="center")
		button2.configure(width = 15, height = 2, activebackground = "#33B5E5", relief = tk.FLAT)
		button2_widget = canvasBody.create_window(405, 80, window=button2)
		# Label Movimientos
		labelMove = tk.Label(self,text="Movimientos", bg = "#154e72", font=('Verdana 12 bold'), fg="white")
		LabelMove_widget = canvasBody.create_window(80, 30-5, window=labelMove)
		# Label Rendimiento
		labelR = tk.Label(self,text="Rendimiento", bg = "#154e72", font=('Verdana 12 bold'), fg="white")
		LabelR_widget = canvasBody.create_window(80, 165-5, window=labelR)
		labelR1 = tk.Label(self,text="Último turno", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
		LabelR1_widget = canvasBody.create_window(68, 238-5, window=labelR1)
		labelR2 = tk.Label(self,text="Histórico", bg = "#154e72", font=('Verdana 8 bold'), fg="white")
		LabelR2_widget = canvasBody.create_window(188, 238-5, window=labelR2)
		# Rectangulo Move
		canvasBody.create_rectangle(28,50-5,148,90-5, fill='white', outline='white')
		canvasBody.create_rectangle(160,50-5,280,90-5, fill='white', outline='white')
		canvasBody.create_rectangle(28,102-5,148,142-5, fill='white', outline='white')
		canvasBody.create_rectangle(160,102-5,280,142-5, fill='white', outline='white')
		# Rectangulo Rendimiento
		canvasBody.create_rectangle(28,185-5,148,225-5, fill='white', outline='white')
		canvasBody.create_rectangle(160,185-5,280,225-5, fill='white', outline='white')

	def IniciarModulo():
		subprocess.call(["python", "run.py"])


if __name__ == "__main__":
    app = LoremIpsum()
    app.mainloop()