import numpy as np
import os
import collections


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_vid(self, vidf, save):
    datfname = vidf.readline().strip()  # first line
    m_sTime, m_sTimeSize, n_f = vidf.readline().strip().split()
    n_f = int(n_f)
    fields = []
    for i_field in range(n_f):
        field = dict()
        name, size, n_axis = vidf.readline().strip().split()
        # print name, size,n_axis
        field['name'] = name  # string
        field['size'] = size  # string
        n_axis1 = int(n_axis)  # int
        axes = []
        npoints = 1
        for i_axis in range(n_axis1):
            axis = dict()
            line = vidf.readline().strip().split()
            name, _, npoints = line[:3]
            axis['name'] = name
            axis['npoints'] = abs(int(npoints))
            axis['points'] = list(map(float, line[3:]))
            axes.append(axis)
            npoints *= axis.npoints
        field['axes'] = axes
        fields.append(field)

    vidf.readline()  # DATA
    vidf.readline()  # <Number of arrays>
    n_fields = int(vidf.readline())
    for i_field in range(n_fields):
        vidf.readline()  # <Number of points on Time-axe>
        vidf.readline()  # 2001
        vidf.readline()  # <Coordinates (time)>
        vidf.readline()  # <Array number>
        num, name, _ = vidf.readline().strip().split()  # 2001 f1 Sgs
        vidf.readline()  # <1 - Si, 0 - Sgs>
        vidf.readline()  #
        vidf.readline()  # <0-point, 1-array>
        array_flg = int(vidf.readline())  #
        vidf.readline()  # <Nonformatted, Formatted, Display (1-Yes, 0 - No)>
        vidf.readline()
        coords = []
        for i_axis in range(3):
            vidf.readline()  # <Number of points on X-axe>
            _ = int(vidf.readline())
            vidf.readline()  # <Coordinates>
            coords.append(list(map(int, vidf.readline().strip().split())))
        fields[i_field]['array'] = array_flg
        fields[i_field]['name'] = name
        fields[i_field]['num'] = num
        if not array_flg:
            axes = []
            for i, i_axis in enumerate(coords):
                axis = dict()
                axis['name'] = 'xyz'[i]
                axis['npoints'] = 1
                axis['ipoints'] = coords[i]
                axes.append(axis)
            fields[i_field]['axes'] = axes + fields[i_field]['axes']

    return datfname, fields


def read_start(startFN):
    d = {}

    with open(startFN) as startf:
        d["prjf"] = startf.readline().strip()
        # d["npx"] = int(startf.readline().strip())
    return d


def read_prj(prjFN):
    d = {}
    with open(prjFN) as prjf:

        line = prjf.readline().strip()
        while line:
            line = prjf.readline().strip()
            if "Dxf name" in line:
                d["dxff"] = prjf.readline().strip()
            elif "Grd name" in line:
                d["grdf"] = prjf.readline().strip()
            elif "Cell name" in line:
                d["celf"] = prjf.readline().strip()
            elif "Particles-Layers name" in line:
                d["plf"] = prjf.readline().strip()
            elif "Particles name" in line:
                d["parf"] = prjf.readline().strip()
            elif "Layers name" in line:
                d["layf"] = prjf.readline().strip()
            elif "Output" in line:
                d["vidf"] = []
                n_outp = int(prjf.readline().strip())
                for i in range(n_outp):
                    d["vidf"].append(prjf.readline().strip())
                    prjf.readline()
    return d


def read_grid(grdFN):
    d = {}
    with open(grdFN) as grdf:
        grdf.readline()  # <FULL>
        grdf.readline()  # 1
        grdf.readline()  # <EQUAL (0-not equal 1-equal)>
        grdf.readline()  # 0
        grdf.readline()  # X
        d["nx"] = int(grdf.readline())
        d["x"] = list(map(float, grdf.readline().strip().split()))

        grdf.readline()
        d["ny"] = int(grdf.readline())
        d["y"] = list(map(float, grdf.readline().strip().split()))

        grdf.readline()
        d["nz"] = int(grdf.readline())
        d["z"] = list(map(float, grdf.readline().strip().split()))

        grdf.readline()
        d["nt"] = int(grdf.readline())
        d["t"] = list(map(float, grdf.readline().strip().split()))

    return d


def process_grid(xi):
    d = {}

    xi05 = [0.5 * (x1 + x2) for x1, x2 in zip(xi[1:], xi[:-1])]

    dxi05 = [x2 - x1 for x2, x1 in zip(xi[1:], xi[:-1])]
    dxi05.insert(0, dxi05[0])
    dxi05.append(dxi05[-1])

    xi05.insert(0, xi[0] - 0.5 * dxi05[0])
    xi05.append(xi[-1] + 0.5 * dxi05[-1])
    d['i'] = xi
    d['i05'] = xi05
    d['di05'] = dxi05

    return d


def read_space(celFN, shape):
    cells = np.loadtxt(celFN, np.dtype(int))
    size = (shape[0] - 2) * (shape[1] - 2) * (shape[2] - 2)
    space1 = np.array([-1] * size)
    space = np.full(shape, -1)

    assert np.sum(cells[1::2]) == size

    filled = 0
    for i in range(cells.size // 2):
        fill = cells[2 * i + 1]
        space1[filled:filled + fill] = cells[2 * i]
        filled += fill
    space1.resize((shape[0] - 2), (shape[1] - 2), (shape[2] - 2))
    space[1:-1, 1:-1, 1:-1] = space1
    return space


def read_par(parFN):
    d = dict()
    parfile = open(parFN)
    parfile.readline()
    parfile.readline()
    par_count = int(parfile.readline())
    parfile.readline()
    for i_par in range(par_count):
        parfile.readline()
        type = int(parfile.readline().split()[0].strip())
        parfile.readline()
        num = int(parfile.readline().split()[0])  # <Number,
        d[num] = type
        parfile.readline()  # <Kinetic ?nergy Barrier(MeV)>
        parfile.readline()
        parfile.readline()  # <Number of processes>
        count = int(parfile.readline())
        for _ in range(count):
            parfile.readline()
            parfile.readline()
        parfile.readline()
    parfile.close()
    return d


def read_lay(layFN):
    d = collections.OrderedDict()
    layfile = open(layFN)
    layfile.readline()
    layfile.readline()
    lay_count = int(layfile.readline())
    layfile.readline()
    for i_lay in range(lay_count):
        layfile.readline()  # <Number, layer name>
        num = int(layfile.readline().split()[0].strip())
        layfile.readline()  # <p/p N, conduct. N, ext. J (0-no,1-yes),
        _, _, curr, en, extra = map(int, layfile.readline().strip().split())  # <Number,
        d[num] = (curr, en)

        layfile.readline()
        layfile.readline()
        if extra:
            layfile.readline()
            layfile.readline()
        layfile.readline()

    layfile.close()
    return d


def read_pl(plFN):
    d = dict()
    plfile = open(plFN)
    plfile.readline()
    plfile.readline()
    par_count = int(plfile.readline())
    plfile.readline()
    d['particles'] = list(map(int, plfile.readline().split()))
    plfile.readline()
    lay_count = int(plfile.readline())
    plfile.readline()
    d['layers'] = list(map(int, plfile.readline().split()))
    plfile.readline()  # <Particle motion in a layer
    for i in range(par_count):
        plfile.readline()
    plfile.readline()  # <Source in volume
    d['volume_src'] = []
    for i in range(par_count):
        d['volume_src'].append(list(map(int, plfile.readline().split())))

    plfile.readline()  # <Surface source
    d['surface_src'] = dict()
    for j in range(par_count):
        plfile.readline()
        p = int(plfile.readline())
        d['surface_src'][p] = []
        for i in range(lay_count):
            d['surface_src'][p].append(list(map(int, plfile.readline().split())))

    plfile.readline()  # <Current density calculation
    d['current_src'] = []
    for i in range(par_count):
        d['current_src'].append(list(map(int, plfile.readline().split())))
    plfile.close()
    return d


def read_REMP(path):
    if os.path.exists(os.path.join(path, "START_N")):
        start = read_start(os.path.join(path, "START_N"))
    else:
        try:
            start = read_start(os.path.join(path, "START"))
        except (IOError, ValueError) as e:
            return

    prj = read_prj(os.path.join(path, start["prjf"]))
    grd_ = read_grid(os.path.join(path, prj["grdf"]))
    pl = read_pl(os.path.join(path, prj["plf"]))
    par = read_par(os.path.join(path, prj["parf"]))
    lay = read_lay(os.path.join(path, prj["layf"]))

    x, y, z = [process_grid(grd_['x']), process_grid(grd_['y']), process_grid(grd_['z'])]

    prj['prjf'] = start["prjf"]
    space = read_space(os.path.join(path, prj["celf"]), (len(x['i']) + 1, len(y['i']) + 1, len(z['i']) + 1))
    return prj, [x, y, z], space, pl, par, lay
