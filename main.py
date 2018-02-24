# A class for displaying the GUI, along with various states of the
# chess game. It (should) contain the main menu, a player versus AI
# board, multiplayer board (not networked, of course!), and a simple
# tool used for reviewing games. We use Python's default graphical
# library, tkinter, to create the GUI.

# IMPORTS
from tkinter import *
from tkinter import ttk
from tkinter import font

# GAME STATE:
# 0 = main menu
# 1 = playing against AI
# 2 = playing against another human
# 3 = viewing game

# The above settings should essentially determine how much control the
# user has over the board. In state 1, they are only allowed to move
# their pieces (TODO: implement random sides). In state 2, they can
# move any piece, because there are assumed to be two players sitting
# at the same computer. In state 3, the player has no control over the
# board, and can only view the game state as it progresses.
global_game_state = 0

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
ai_frame = Frame(root, width=600, height=600)
ai_frame.place(x=0, y=0)

# Multiplayer frame:
mp_frame = Frame(root, width=600, height=600)
mp_frame.place(x=0, y=0)

# Game viewer frame:
viewer_frame = Frame(root)


####################################################################################################################################
# Populating main menu:

# Title image:
img = PhotoImage(file="images//white_king.png")
img_container = Label(main_menu, image=img)
img_container.image = img
img_container.place(relx=0.5, y=60, anchor=CENTER)

# Title:
consolas = font.Font(family="Consolas", size=18, weight="bold")
button_font = font.Font(family="Helvetica", size=14, weight="bold")

title = Label(main_menu, text="simplechessAI", fg="black", font=consolas)
title.place(relx=0.5, y=100, anchor=CENTER)

# Buttons
ai_button = Button(main_menu,
                   text="AI",
                   height=2,
                   width=20,
                   font=button_font,
                   command=lambda:raise_frame(ai_frame))
ai_button.place(x=300, y=150, anchor=CENTER)

mp_button = Button(main_menu,
                   text="Multiplayer",
                   height=2,
                   width=20,
                   font=button_font,
                   command=lambda:raise_frame(mp_frame))
mp_button.place(x=300, y=220, anchor=CENTER)

viewer_button = Button(main_menu,
                       text="Load Game",
                       height=2,
                       width=20,
                       font=button_font,
                       command=lambda:raise_frame(viewer_frame))
viewer_button.place(x=300, y=290, anchor=CENTER)
####################################################################################################################################
# Populating AI frame:
# We should assume that whenever the AI button is clicked, a new game is started. Thus, the board state should be reset, and
# the visual representation should be as well. After each move, we see what the player (or the AI) wants to change about
# the board, change the state, and then update the GUI.

# Board background:
board_img = PhotoImage(file="images//board.png")
ai_board_container = Label(ai_frame, image=board_img)
ai_board_container.image = board_img
ai_board_container.place(relx=0.5, rely=0.5, anchor=CENTER)

# Back button
ai_back_button = Button(ai_frame,
                     text="<--",
                     height=1,
                     width=3,
                     font=font.Font(family="Consolas", size=12, weight="bold"),
                     command=lambda:raise_frame(main_menu))
ai_back_button.place(rely=1, x=10, y=-30, anchor=W)
####################################################################################################################################
# Populating MP frame:
# Same rules as above. Whenever this frame is switched to, we reset the board state.

# Board background:
mp_board_container = Label(mp_frame, image=board_img)
mp_board_container.image = board_img
mp_board_container.place(relx=0.5, rely=0.5, anchor=CENTER)

# Back button
mp_back_button = Button(mp_frame,
                     text="<--",
                     height=1,
                     width=3,
                     font=font.Font(family="Consolas", size=12, weight="bold"),
                     command=lambda:raise_frame(main_menu))
mp_back_button.place(rely=1, x=10, y=-30, anchor=W)

raise_frame(main_menu)
root.mainloop()
