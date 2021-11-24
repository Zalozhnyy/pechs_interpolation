import os

import tkinter as tk
import traceback
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter import filedialog as fd
from tkinter import simpledialog

from typing import List, Union, Optional, Dict

from pi_datatypes import UiConfiguration, InterpolationMethods, ToManyActiveDetectors, CantFindPillarAxe
from read_PECHS import read_dets, read_tmpl
from pi_controller import Calculations

INTERPOLATION_METHODS = {
    'По ближайшему': InterpolationMethods.nearest,
    'По n ближайшим': InterpolationMethods.n_nearest,
    'Столб': InterpolationMethods.pillar,
}

SUPPORTED_CALCS = {'energy', 'current', 'flux'}


class MainWindow(tk.Frame):
    def __init__(self, parent, proj):
        super().__init__(parent)

        self._conf = UiConfiguration()

        self.path_frame = tk.LabelFrame(self, text='Проекты')
        self.configurations_frame = tk.LabelFrame(self, text='Настройки')

        self.path_frame.grid(row=0, column=0, rowspan=5, columnspan=5, sticky='NW')
        self.configurations_frame.grid(row=6, column=0, rowspan=4, columnspan=5, sticky='NW')

        self.families_frame: ttk.LabelFrame = None

        self.grid(sticky='NWSE', padx=5, pady=5)

        self.remp_path_label = tk.Label
        self.pechs_path_label = tk.Label
        self.interpolation_methods_combobox: ttk.Combobox

        self._pechs_values_checkboxes: List[ttk.Combobox]
        self._pechs_values_activations: List[tk.BooleanVar]
        self._pechs_values_progressbar: List[ttk.Progressbar]

        self._choice_pechs_button: tk.Button

        self._count_detectors: tk.Entry
        self._count_detectors_value: tk.StringVar = tk.StringVar(value='8')

        self._init_path_frame()
        self._init_configuration_frame()

        self.calc_button = tk.Button(self, text='Расчитать', width=14, command=self._calculate,
                                     state='disabled')
        self.calc_button.grid(row=10, column=0, padx=20, pady=20)

        if proj:
            self._conf.remp_path = os.path.dirname(proj)
            self.remp_path_label['text'] = os.path.split(self._conf.remp_path)[1]
            self._choice_pechs_button['state'] = 'active'

    def _init_path_frame(self):
        prj_label = tk.Label(self.path_frame, text='Проект РЭМП:  ', justify='left')
        prj_label.grid(row=0, column=0)

        self.remp_path_label = tk.Label(self.path_frame, text='Проект не выбран', justify='left')
        self.remp_path_label.grid(row=0, column=1, columnspan=1, sticky='WE')

        choice_project_button = tk.Button(self.path_frame,
                                          text='Выбрать проект',
                                          width=14,
                                          command=self._set_remp_project)
        choice_project_button.grid(row=0, column=5, columnspan=1, padx=15, pady=10)

        pechs_label = tk.Label(self.path_frame, text='Проект PECHS: ', justify='left')
        pechs_label.grid(row=1, column=0)

        self.pechs_path_label = tk.Label(self.path_frame, text='Проект не выбран', justify='left')
        self.pechs_path_label.grid(row=1, column=1, columnspan=1, sticky='WE')

        self._choice_pechs_button = tk.Button(self.path_frame,
                                              text='Выбрать проект',
                                              width=14,
                                              command=self._set_pechs_project)
        self._choice_pechs_button.grid(row=1, column=5, columnspan=1, padx=15, pady=10)

    def _init_configuration_frame(self):
        calc_method_label = tk.Label(self.configurations_frame, text='Метод расчёта:  ', justify='left')
        calc_method_label.grid(row=0, column=0)

        self.interpolation_methods_combobox = ttk.Combobox(self.configurations_frame,
                                                           value=[v for v in INTERPOLATION_METHODS.keys()],
                                                           width=25,
                                                           state='readonly')

        self.interpolation_methods_combobox.grid(row=0, column=1, padx=15, pady=10)
        self.bind_class(self.interpolation_methods_combobox, "<<ComboboxSelected>>", self._edit_calc_method)

        self.interpolation_methods_combobox.set(list(INTERPOLATION_METHODS.keys())[0])

        self._count_detectors = tk.Entry(self.configurations_frame, textvariable=self._count_detectors_value,
                                         state='disabled',
                                         width=5, justify='center')
        self._count_detectors.grid(row=0, column=6, padx=5, pady=10)

        self._edit_calc_method(None)

        if not self._conf.remp_path:
            self._choice_pechs_button['state'] = 'disabled'

    def _edit_calc_method(self, event):
        self._conf.calculation_method = INTERPOLATION_METHODS[self.interpolation_methods_combobox.get()]

        if self._conf.calculation_method == InterpolationMethods.n_nearest:
            self._count_detectors['state'] = 'normal'
        else:
            self._count_detectors['state'] = 'disabled'

    def _set_remp_project(self):
        proj_path = fd.askopenfilename(title='Выберите файл проекта РЭМП', filetypes=[('PRJ files', '.PRJ')])

        if proj_path == '':
            return

        self._conf.remp_path = os.path.dirname(proj_path)
        self._conf.pechs_path = None

        self.remp_path_label['text'] = os.path.split(self._conf.remp_path)[1]
        self.pechs_path_label['text'] = 'проект не выбран'

        self._choice_pechs_button['state'] = 'active'
        self.calc_button['state'] = 'disabled'
        self._destroy_checkbox_frame()

    def _set_pechs_project(self):
        proj_path = fd.askopenfilename(title='Выберите файл проекта PECHS',
                                       filetypes=[('configuration', 'configuration')],
                                       initialdir=self._conf.remp_path)

        if proj_path == '':
            return

        self._destroy_checkbox_frame()

        self._conf.pechs_path = os.path.dirname(proj_path)
        self.pechs_path_label['text'] = os.path.split(self._conf.pechs_path)[1]

        self._pechs_processing()

    def _init_checkboxes(self):
        if self.families_frame:
            self.families_frame.destroy()

        self.families_frame = tk.LabelFrame(self, text='Семейства детекторов')
        self.families_frame.grid(row=10, column=0, rowspan=10, columnspan=5, sticky='NW')

        checkboxes, boolean_values, pb = [], [], []

        for i, (name, det) in enumerate(self._conf.pechs_detectors_families.items()):
            boolean_values.append(tk.BooleanVar(value=1))

            checkboxes.append(ttk.Checkbutton(self.families_frame,
                                              text=name,
                                              variable=boolean_values[i],
                                              onvalue=1, offvalue=0
                                              ))
            pb.append(ttk.Progressbar(self.families_frame, orient='horizontal',
                                      length=200, mode='determinate'))

            checkboxes[i].grid(row=i, column=1, padx=5, pady=10)
            pb[i].grid(row=i, column=2, padx=5, pady=10)

        self._pechs_values_checkboxes, self._pechs_values_activations = checkboxes, boolean_values
        self._pechs_values_progressbar = pb

        self.calc_button.grid_configure(row=len(self._conf.pechs_detectors_families) + 21)

    def _pechs_processing(self):

        detectors_path = os.path.join(self._conf.pechs_path, 'initials', 'detectors')

        if not os.path.exists(detectors_path):
            raise Exception(f'Не найдена директория {detectors_path}')

        self._conf.pechs_detectors_families = {det['name']: {key: value for key, value in det.items() if key != 'name'}
                                               for det
                                               in read_dets(detectors_path) if
                                               det['name'].split('_')[0] in SUPPORTED_CALCS}

        if self._conf.pechs_detectors_families:
            self._init_checkboxes()
            self.calc_button['state'] = 'active'

    def _destroy_checkbox_frame(self):
        if self.families_frame:
            self.families_frame.destroy()
            self._pechs_values_checkboxes, self._pechs_values_activations = None, None

    def _calculate(self):

        data = self._conf

        for i, (name, det) in enumerate(data.pechs_detectors_families.items()):
            if not self._pechs_values_activations[i].get():
                continue

            try:
                n = int(self._count_detectors_value.get()) if data.calculation_method == InterpolationMethods.n_nearest else 0
                self._pechs_values_progressbar[i]['value'] = 0

                d = {
                    'remp_dir': data.remp_path,
                    'res': os.path.join(data.pechs_path, 'results', name + '.res'),
                    'lst': os.path.join(data.pechs_path, 'initials', det['list']),
                    'type': det['type'],
                    'progress_bar': self._pechs_values_progressbar[i],
                    'method': data.calculation_method,
                    'n': n,
                    'widget': self,
                    'name': name,
                }

                if det['type'] == 'FLUX' or 'flux' in name:
                    d['method'] = InterpolationMethods.flux_translation
                    d['measure'] = det['measure']
                    d['template'] = read_tmpl(os.path.join(data.pechs_path, 'initials', det['templ']))

            except ValueError:
                mb.showerror('Ошибка', 'Неверное значение в количестве детекторов')
                return
            except Exception('unexpected exception') as e:
                print(e)
                return

            self.calc_button['state'] = 'disabled'

            try:
                Calculations().calculate(**d)

            except ToManyActiveDetectors as e:
                mb.showerror('Ошибка', e)

            except CantFindPillarAxe as e:
                mb.showerror('Ошибка', e)

            except Exception as e:
                print(traceback.print_tb(e))
                mb.showerror('Ошибка', e)

            self.calc_button['state'] = 'normal'
