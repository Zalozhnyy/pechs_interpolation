import os
import sys

import tkinter as tk

from pi_main_window import MainWindow


def main():
    root = tk.Tk()
    root.title('Pechs interpolation')
    root.geometry('400x500')

    try:
        if len(sys.argv) > 1:
            projectfilename = sys.argv[1]
        else:
            print(f'Проект {projectfilename}')

        ini = os.path.normpath(projectfilename)
    except Exception:
        ini = None

    main_win = MainWindow(root, proj=ini)
    root.mainloop()


if __name__ == '__main__':
    main()
