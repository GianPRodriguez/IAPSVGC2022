class Team():
	def __init__(self):
		self.idTeam = -1
		self.nameTeam = None
		self.concludedBattles = 0
		self.battlesWon1000 = 0

	async def CreateTeam(self, idTeam, nameTeam, concludedBattles, battlesWon1000):
		self.idTeam = idTeam
		self.nameTeam = nameTeam
		self.concludedBattles = concludedBattles
		self.battlesWon1000 = battlesWon1000