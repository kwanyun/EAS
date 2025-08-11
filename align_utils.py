import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import PIL
import numpy as np
style_prompts = {
    'disney': 'A head of a Disney character with rounded features, large eyes, in a magical style.',
    'green_orc': 'A head of a green orc with rough, heavy-set features, tusks, and strong jawline.',
    'pixar_child': 'A head of a Pixar child character with soft, rounded features, large eyes, and expressive, innocent look.',
    'baby' : 'A head of a baby',
    'lego_head': 'A simple head of a Lego minifigure with minimalistic features, yellow cylinder lego figure head.',
    'statue' : 'A head of a greek gypsum statue',
    'yoda': 'A head of a yoda from Starwars, green skin, wrinkled skin, elongated ears.',
    'neanderthal' : 'A head of a neanderthal',
    'woody': 'A head of a Toy Story Woody character with a long face in a toy-like aesthetic.',
    'vampire': 'A head of a vampire with pale skin, sharp fangs, and hauntingly beautiful, cold eyes.',
    'moana': 'A head of Moana, with warm, youthful features.',
    'russell': 'A head of Russell from *Up*, Pixar, with round, circlular face, chubby cheeks.',
    'mulan': 'A head of Mulan, with graceful, determined features and a warrior’s poise.',
}




simple_neg = "grainy, messy, out of frame, shiny, greasy, clothed, t-shirts, background, frown"

def get_patch(landmarks, color='lime', closed=False):
    contour = landmarks
    ops = [Path.MOVETO] + [Path.LINETO]*(len(contour)-1)
    headcolor = (0, 0, 0, 0)      # Transparent fill color, if open
    if closed:
        contour.append(contour[0])
        ops.append(Path.CLOSEPOLY)
        headcolor = color
    path = Path(contour, ops)
    return patches.PathPatch(path, headcolor=headcolor, edgecolor=color, lw=4)

def conditioning_from_landmarks(landmarks, size=512):
    # Precisely control output image size
    dpi = 72
    fig, ax = plt.subplots(1, figsize=[size/dpi, size/dpi], tight_layout={'pad':0})
    fig.set_dpi(dpi)

    black = np.zeros((size, size, 3))
    ax.imshow(black)
    
    x, y  =landmarks[96]
    square96 = [
    [x - 1, y - 1],
    [x - 1, y + 1],
    [x + 1, y + 1],
    [x + 1, y - 1]
    ]
    x, y  =landmarks[97]
    square97 = [
    [x - 1, y - 1],
    [x - 1, y + 1],
    [x + 1, y + 1],
    [x + 1, y - 1]
    ]

    
    nose_v = get_patch([landmarks[53],landmarks[54],landmarks[57]], color='orange')
    l_eye = get_patch(square96, color='magenta', closed=True)
    r_eye = get_patch(square97, color='magenta', closed=True)
    outer_lips = get_patch([landmarks[76],landmarks[82]], color='cyan', closed=True)
    inner_lips = get_patch([landmarks[90],landmarks[94]], color='red', closed=True)

    ax.add_patch(nose_v)
    ax.add_patch(l_eye)
    ax.add_patch(r_eye)
    ax.add_patch(outer_lips)
    ax.add_patch(inner_lips)

    plt.axis('off')
    
    fig.canvas.draw()
    buffer, (width, height) = fig.canvas.print_to_buffer()
    assert width == height
    assert width == size
    
    buffer = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
    buffer = buffer[:, :, 0:3]
    plt.close(fig)
    return PIL.Image.fromarray(buffer)
