"""
- Si un ou plusieurs coups gagnants sont trouvés (option "Collect all wins"), on les colore en vert.
- Sinon, on ouvre une fenêtre listant mes coups ; cliquer sur une ligne
  met en évidence le coup envisagé (jaune) et les réponses adverses "gagnantes" (orange).
"""
import tkinter as tk
from tkinter import messagebox
from functools import lru_cache
from collections import deque
import threading

# ---------------- game parameters ----------------
NROWS = NCOLS = 7
N = NROWS * NCOLS
DIRS = [(1,0),(0,1),(-1,0),(0,-1)]

def idx(i,j): return i*NCOLS + j

# ----------------- solver (single-pass) -----------------
# Symmetry helpers (D4)
def board_to_mat(board):
    return [list(board[r*NCOLS:(r+1)*NCOLS]) for r in range(NROWS)]

def mat_to_board(mat):
    out=[]
    for row in mat: out.extend(row)
    return tuple(out)

def rotations_and_reflections(board):
    mat = board_to_mat(board)
    mats=[]
    cur = mat
    for _ in range(4):
        # rotate 90 deg clockwise
        cur = [[cur[r][c] for r in range(len(cur)-1,-1,-1)] for c in range(len(cur[0]))]
        mats.append(cur)
        mats.append([list(reversed(row)) for row in cur])  # reflect of current rotation
    return [mat_to_board(m) for m in mats]

def canonical(board):
    return min(rotations_and_reflections(board))

# rules helpers
def neighbors_coords(i,j):
    for di,dj in DIRS:
        ni, nj = i+di, j+dj
        if 0 <= ni < NROWS and 0 <= nj < NCOLS:
            yield (ni,nj)

def legal_moves_indices(board):
    moves=[]
    for ind,val in enumerate(board):
        if val != 0: continue
        i,j = divmod(ind, NCOLS)
        red = blue = 0
        for ni,nj in neighbors_coords(i,j):
            v = board[idx(ni,nj)]
            if v == 1: red += 1
            elif v == 2: blue += 1
        if red == blue:
            moves.append(ind)
    return moves

# cached solver using canonicalization
@lru_cache(maxsize=None)
def _is_winning_cached(canonical_board, turn):
    moves = legal_moves_indices(canonical_board)
    if not moves:
        return False
    player_color = 1 if turn==0 else 2
    for m in moves:
        lst = list(canonical_board)
        lst[m] = player_color
        child_can = canonical(tuple(lst))
        if not _is_winning_cached(child_can, 1-turn):
            return True
    return False

def is_winning(board, turn):
    # canonicalize the board before looking up
    return _is_winning_cached(canonical(tuple(board)), turn)

# single-pass analysis (modified)
def analyze_position_singlepass(board, turn, collect_all_wins=True, collect_all_replies=True):
    """
    Returns:
      ('terminal', None)                      -> no legal moves (player loses)
      ('wins', list_of_winning_moves)        -> list possibly of length 1 or more (according to collect_all_wins)
      ('no_win', outcomes) where outcomes: dict my_move->list of opponent winning replies
    Notes:
      - collect_all_wins: if True, find all my moves that are winning (else stop at first)
      - collect_all_replies: if True, collect all opponent winning replies for each my move; else stop at first reply
    """
    player = 1 if turn==0 else 2
    moves = legal_moves_indices(board)
    if not moves:
        return ('terminal', None)

    winning_moves = []
    outcomes = {}
    for m in moves:
        # play my move
        state_after_m = list(board)
        state_after_m[m] = player
        opp_moves = legal_moves_indices(tuple(state_after_m))
        # if opponent has no replies -> my move m is winning immediately
        if not opp_moves:
            winning_moves.append(m)
            if not collect_all_wins:
                return ('wins', winning_moves)
            else:
                continue
        winning_replies = []
        opp_color = 1 if (1-turn)==0 else 2
        for r in opp_moves:
            state_after_r = list(state_after_m)
            state_after_r[r] = opp_color
            # if I cannot force a win from state_after_r, r is a winning reply for opponent
            if not is_winning(tuple(state_after_r), turn):
                winning_replies.append(r)
                if not collect_all_replies:
                    break
        outcomes[m] = winning_replies
        if not winning_replies:
            # opponent has no winning reply => my move is winning
            winning_moves.append(m)
            if not collect_all_wins:
                return ('wins', winning_moves)
            else:
                # continue to collect other winning moves
                continue
    if winning_moves:
        return ('wins', winning_moves)
    return ('no_win', outcomes)

# ----------------- GUI -----------------
CELL_SIZE = 64
PAD = 6
RED_COLOR = "#ff6666"
BLUE_COLOR = "#6666ff"
WIN_COLOR = "#66ff66"
SELECT_COLOR = "#ffff66"    # yellow for selected my-move
REPLY_COLOR = "#ffcc66"     # orange for opponent replies
DEFAULT_BG = None  # will be detected

class HadronGUI:
    def __init__(self, master):
        global DEFAULT_BG
        self.master = master
        master.title(f"Hadron {NROWS}x{NCOLS} - GUI + Solver")

        # game state
        self.board = [0]*N  # 0 empty, 1 red, 2 blue
        self.turn = 0  # 0 red to play, 1 blue to play
        self.history = []  # stack of (index, color)
        self.buttons = [None]*N
        self.highlighted = []  # indices highlighted (store tuples (ind, orig_bg))

        # top frame: grid
        grid_frame = tk.Frame(master)
        grid_frame.pack(padx=10, pady=10)

        # create buttons
        for i in range(NROWS):
            for j in range(NCOLS):
                ind = idx(i,j)
                b = tk.Button(grid_frame, text=" ", width=4, height=2,
                              command=lambda k=ind: self.on_cell_click(k))
                b.grid(row=i, column=j, padx=3, pady=3, ipadx=4, ipady=8)
                self.buttons[ind] = b
                if DEFAULT_BG is None:
                    DEFAULT_BG = b.cget("background")

        # status area
        self.status_var = tk.StringVar()
        self.status_var.set("Turn: Red")
        status_label = tk.Label(master, textvariable=self.status_var, font=("Arial", 12))
        status_label.pack(pady=(0,6))

        # control buttons
        ctrl_frame = tk.Frame(master)
        ctrl_frame.pack(pady=(0,10))

        undo_btn = tk.Button(ctrl_frame, text="Undo", command=self.undo_move)
        undo_btn.grid(row=0, column=0, padx=6)

        reset_btn = tk.Button(ctrl_frame, text="Reset", command=self.reset_board)
        reset_btn.grid(row=0, column=1, padx=6)

        self.analyze_btn = tk.Button(ctrl_frame, text="Analyze", command=self.on_analyze)
        self.analyze_btn.grid(row=0, column=2, padx=6)

        # checkbox controlling whether to collect all winning moves (and also whether to collect all replies)
        self.collect_wins_var = tk.BooleanVar(value=True)
        collect_wins_chk = tk.Checkbutton(ctrl_frame, text="Collect all winning moves", variable=self.collect_wins_var)
        collect_wins_chk.grid(row=0, column=3, padx=6)

        self.collect_replies_var = tk.BooleanVar(value=True)
        collect_replies_chk = tk.Checkbutton(ctrl_frame, text="Collect all opponent replies", variable=self.collect_replies_var)
        collect_replies_chk.grid(row=0, column=4, padx=6)

        # output text for explanations
        self.output = tk.Text(master, height=10, width=70, state=tk.DISABLED)
        self.output.pack(padx=10, pady=(0,10))

        # help hint
        hint = tk.Label(master, text="Click a cell to play (Red starts). Then Analyze: green = winning moves; else click a move to see opponent replies.")
        hint.pack(pady=(0,8))

        # store window for replies list (so we can close / reuse)
        self.replies_window = None

    def on_cell_click(self, ind):
        # ignore if already occupied
        if self.board[ind] != 0:
            return
        color = 1 if self.turn==0 else 2
        self.board[ind] = color
        self.history.append((ind, color))
        btn = self.buttons[ind]
        if color == 1:
            btn.configure(bg=RED_COLOR, activebackground=RED_COLOR, state=tk.DISABLED)
        else:
            btn.configure(bg=BLUE_COLOR, activebackground=BLUE_COLOR, state=tk.DISABLED)
        # toggle turn
        self.turn = 1 - self.turn
        self.status_var.set("Turn: " + ("Red" if self.turn==0 else "Blue"))
        # clear prior highlights / output / replies window
        self.clear_highlights()
        self.clear_output()
        self.close_replies_window()

    def undo_move(self):
        if not self.history:
            return
        ind, color = self.history.pop()
        self.board[ind] = 0
        btn = self.buttons[ind]
        btn.configure(state=tk.NORMAL, bg=DEFAULT_BG, relief=tk.RAISED)
        # restore turn to the player who played that move
        self.turn = 0 if color==1 else 1
        self.status_var.set("Turn: " + ("Red" if self.turn==0 else "Blue"))
        # clear highlights and output
        self.clear_highlights()
        self.clear_output()
        self.close_replies_window()

    def reset_board(self):
        self.board = [0]*N
        self.turn = 0
        self.history.clear()
        for b in self.buttons:
            b.configure(state=tk.NORMAL, bg=DEFAULT_BG, relief=tk.RAISED)
        self.status_var.set("Turn: Red")
        self.clear_highlights()
        self.clear_output()
        self.close_replies_window()
        # also clear the solver cache to save memory if needed
        try:
            _is_winning_cached.cache_clear()
        except Exception:
            pass

    def on_analyze(self):
        # disable analyze button while computing
        self.analyze_btn.configure(state=tk.DISABLED)
        self.clear_highlights()
        self.clear_output()
        self.close_replies_window()
        collect_all_wins = self.collect_wins_var.get()
        collect_all_replies = self.collect_replies_var.get()
        # run analysis in a thread to keep GUI responsive
        t = threading.Thread(target=self._run_analysis_thread, args=(tuple(self.board), self.turn, collect_all_wins, collect_all_replies), daemon=True)
        t.start()

    def _run_analysis_thread(self, board, turn, collect_all_wins, collect_all_replies):
        try:
            res = analyze_position_singlepass(board, turn, collect_all_wins, collect_all_replies)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {e}"))
            self.master.after(0, lambda: self.analyze_btn.configure(state=tk.NORMAL))
            return
        self.master.after(0, lambda: self._display_analysis_result(res))

    def _display_analysis_result(self, res):
        kind, data = res
        if kind == 'terminal':
            self.append_output("No legal moves: current player loses.\n")
            # no messagebox per request, but inform in text
            self.analyze_btn.configure(state=tk.NORMAL)
            return
        if kind == 'wins':
            wins = data  # list of winning moves indices
            if not wins:
                # shouldn't happen but handle
                self.append_output("No winning moves found (unexpected empty list).\n")
                self.analyze_btn.configure(state=tk.NORMAL)
                return
            self.append_output(f"Found {len(wins)} winning move(s). Highlighting in green.\n")
            for m in wins:
                self.highlight_index(m, WIN_COLOR)
            # don't show message boxes, just enable button again
            self.analyze_btn.configure(state=tk.NORMAL)
            return
        # kind == 'no_win'
        outcomes = data
        self.append_output("No immediate winning move. Opponent winning replies per my move:\n")
        # prepare a small replies window with clickable list
        self.replies_window = tk.Toplevel(self.master)
        self.replies_window.title("Opponent replies per my move (click a move)")
        listbox = tk.Listbox(self.replies_window, width=60, height=12)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(self.replies_window, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)

        # map list index -> (my_move_index, replies_list)
        map_items = []
        for m, replies in sorted(outcomes.items()):
            i,j = divmod(m, NCOLS)
            if not replies:
                desc = f"My move ({i},{j}) [index {m}]: opponent has NO immediate winning reply."
            else:
                coords = [divmod(r, NCOLS) for r in replies]
                coords_str = ', '.join(f"({x},{y})" for x,y in coords)
                desc = f"My move ({i},{j}) [index {m}]: opponent winning replies: {coords_str}"
            listbox.insert(tk.END, desc)
            map_items.append((m, replies))
            # also append to text output
            self.append_output(desc + "\n")

        def on_select(evt):
            # clear previous highlights
            self.clear_highlights()
            w = evt.widget
            if not w.curselection():
                return
            idx_sel = int(w.curselection()[0])
            m, replies = map_items[idx_sel]
            # highlight my move (yellow)
            self.highlight_index(m, SELECT_COLOR)
            # highlight replies (orange)
            for r in replies:
                self.highlight_index(r, REPLY_COLOR)
            # scroll to the move in the main output too
            i,j = divmod(m, NCOLS)
            self.append_output(f"\nSelected my move ({i},{j}) -> opponent replies: {', '.join(str(divmod(r,NCOLS)) for r in replies)}\n")

        listbox.bind('<<ListboxSelect>>', on_select)

        self.analyze_btn.configure(state=tk.NORMAL)

    def append_output(self, text):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, text)
        self.output.configure(state=tk.DISABLED)
        self.output.see(tk.END)

    def clear_output(self):
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.configure(state=tk.DISABLED)

    def highlight_index(self, ind, color):
        """Highlight a button (store original bg so we can restore)."""
        b = self.buttons[ind]
        orig_bg = b.cget("background")
        # store original only once per index
        self.highlighted.append((ind, orig_bg))
        b.configure(relief=tk.SUNKEN, bd=3)
        # if the cell is empty, color it; if already filled, change border color
        if self.board[ind] == 0:
            try:
                b.configure(bg=color)
            except Exception:
                pass
        else:
            # already played cell: set highlight via highlightbackground if platform supports it
            try:
                b.configure(highlightbackground=color, highlightcolor=color)
            except Exception:
                pass

    def clear_highlights(self):
        for ind, orig_bg in self.highlighted:
            b = self.buttons[ind]
            b.configure(relief=tk.RAISED, bd=2)
            # restore background if empty
            if self.board[ind] == 0:
                b.configure(bg=orig_bg)
            else:
                # played cell: remove highlight settings if possible
                try:
                    b.configure(highlightbackground=orig_bg, highlightcolor=orig_bg)
                except Exception:
                    pass
        self.highlighted.clear()

    def close_replies_window(self):
        if self.replies_window is not None:
            try:
                self.replies_window.destroy()
            except Exception:
                pass
            self.replies_window = None

# ----------------- main -----------------
def main():
    root = tk.Tk()
    app = HadronGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
