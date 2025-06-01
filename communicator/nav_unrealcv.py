from simworld.communicator.unrealcv import UnrealCV

class NavUnrealCV(UnrealCV):
    def __init__(self, port = 9000, ip = '127.0.0.1', resolution = (320, 240)):
        super().__init__(port, ip, resolution)


