# A class for displaying the GUI, along with various states of the
# chess game. It (should) contain the main menu, a player versus AI
# board, multiplayer board (not networked, of course!), and a simple
# tool used for reviewing games. We use Python's default graphical
# library, tkinter, to create the GUI.

# IMPORTS
from tkinter import *
from tkinter import ttk
from tkinter import font

# FUNCTION DEFINITIONS
def raise_frame(frame):
    frame.tkraise()

# GUI CREATION:
# Create a root window of size 600x600px.
root = Tk()
root.geometry("600x600")

# Disable resizing.
root.resizable(False, False)

# FRAMES:
# Main menu frame:
# This contains an image of a white king at the top center, a title
# (simplechessAI), and three buttons, all of which are described above.
main_menu = Frame(root, width=600, height=600)
main_menu.place(x=0, y=0)

# Singleplayer frame:
ai_frame = Frame(root, width=600, height=600, bg="black")
ai_frame.place(x=0, y=0)

# Multiplayer frame:
mp_frame = Frame(root)

# Game viewer frame:
viewer_frame = Frame(root)


# Populating main menu:

# Title image:
img = PhotoImage(file="images\white_king.png")
img_container = Label(main_menu, image=img)
img_container.image = img
img_container.place(relx=0.5, y=60, anchor=CENTER)

# Title:
consolas = font.Font(family="Consolas", size=18, weight="bold")
button_font = font.Font(family="Helvetica", size=14, weight="bold")

title = Label(main_menu, text="simplechessAI", fg="black", font=consolas)
title.place(relx=0.5, y=100, anchor=CENTER)

# Buttons
ai_button = Button(main_menu, text="AI", height=2, width=20, font=button_font, command=lambda:raise_frame(ai_frame))
ai_button.place(x=300, y=150, anchor=CENTER)
mp_button = Button(main_menu, text="Multiplayer", height=2, width=20, font=button_font, command=lambda:raise_frame(mp_frame))
mp_button.place(x=300, y=220, anchor=CENTER)
viewer_button = Button(main_menu, text="Load Game", height=2, width=20, font=button_font, command=lambda:raise_frame(viewer_frame))
viewer_button.place(x=300, y=290, anchor=CENTER)

raise_frame(main_menu)
root.mainloop()
