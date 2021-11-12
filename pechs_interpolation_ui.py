import os
import sys

sys.path.append(os.path.dirname(__file__))

import tkinter as tk

from pi_main_window import MainWindow


def main():
    root = tk.Tk()
    root.title('Pechs interpolation')
    root.geometry('400x500')

    if len(sys.argv) > 1:
        ini = sys.argv[1]
    else:
        try:
            print(f'Проект {projectfilename}')
            ini = os.path.normpath(projectfilename)
        except Exception:
            ini = None

    main_win = MainWindow(root, proj=ini)
    root.mainloop()


if __name__ == '__main__':
    main()
