# A class for displaying the GUI, along with various states of the
# chess game. It (should) contain the main menu, a player versus AI
# board, multiplayer board (not networked, of course!), and a simple
# tool used for reviewing games. We use Python's default graphical
# library, tkinter, to create the GUI.

# IMPORTS
from tkinter import *
from tkinter import ttk
from tkinter import font
from board import board

# Image references:
img_ref = {"6": "images//white_king.png",
           "5": "images//white_queen.png",
           "4": "images//white_rook.png",
           "3": "images//white_bishop.png",
           "2": "images//white_knight.png",
           "1": "images//white_pawn.png",
           "-1": "images//black_pawn.png",
           "-2": "images//black_knight.png",
           "-3": "images//black_bishop.png",
           "-4": "images//black_rook.png",
           "-5": "images//black_queen.png",
           "-6": "images//black_king.png"}

# FUNCTION DEFINITIONS
def raise_frame(frame):
    frame.tkraise()

b = board()

def refresh():
    state = b.get_state()
    x_coord = 60
    y_coord = 60
    
    for row in range(8):
        for col in range(8):
            if (state[row][col] != 0):

                piece_img = PhotoImage(file="images//board.png")#file=img_ref[str(state[row][col])])
                test_canvas.create_image(x_coord, y_coord, anchor=NW, image=piece_img)
                
            x_coord += 60
        y_coord += 60
        x_coord = 60
    

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
test_canvas = Canvas(ai_frame, width=600, height=600)
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
test_canvas.place(x=0, y=0, anchor=NW)
test_canvas.create_image(0, 0, image=board_img, anchor=NW)

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
refresh()

root.mainloop()
