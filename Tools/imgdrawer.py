import cv2
import numpy as np

data = {"x1": 1065.999755859375, "y1": 415.8536071777344, "x2": 1198.3338623046875, "y2": 822.6866455078125}

# 1. Setup coordinate (conversione in int necessaria)
pt1 = (int(data["x1"]), int(data["y1"]))
pt2 = (int(data["x2"]), int(data["y2"]))

# 2. Creazione Canvas
# Nota: in OpenCV le dimensioni sono (altezza, larghezza, canali)
# Creiamo uno sfondo bianco (255)
canvas = np.ones((1000, 1500, 3), dtype="uint8") * 255

# 3. Disegno
# Sintassi: immagine, punto1, punto2, colore (BGR), spessore (-1 = riempimento)
# Nota: Il nero in BGR è (0, 0, 0)
cv2.rectangle(canvas, pt1, pt2, (0, 0, 0), -1)

# 4. Salvataggio
cv2.imwrite("output_opencv.png", canvas)
