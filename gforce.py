from scipy.integrate import ode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import math
import numpy as np
from pyqtgraph.dockarea import *

rate = 100.

G = 6.674e-11
Me = 5.972e24
Ms = 50
Re = 6.3781e6
Ds = 300e3

Meso = 80e3

alpha = 0 * math.pi / 180.
theta = 0 * math.pi / 180.
omega = 0 * math.pi / 180.

decay = 1


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


isp = 9100

s0 = [0, 0, 0, 0, 0, 0]
r0 = 0


def calc(alpha, omega, theta, isp):
    global s0, r0, Ds, Re

    rw = Re + Ds
    op = np.dot(rotation_matrix([math.cos(theta), math.sin(theta), 0], alpha), [math.cos(theta + math.pi / 2 - omega), math.sin(theta + math.pi / 2 - omega), 0])
    s0 = [
        math.cos(theta) * rw, math.sin(theta) * rw, 0,
        op[0] * isp, op[1] * isp, op[2] * isp,
    ]

    r0 = (s0[0] ** 2 + s0[1] ** 2 + s0[2] ** 2) ** 0.5


calc(alpha, omega, theta, isp)


def ut(a): return 1 if a > 0 else 0


def gforce(t, s):
    global decay, Meso, Re, Me, Ms, G
    x = s[0]
    dx0 = s[3]

    y = s[1]
    dy0 = s[4]

    z = s[2]
    dz0 = s[5]

    rx = (x ** 2 + y ** 2 + z ** 2) ** 0.5

    dcr = 1 - (math.exp(t * 1.032e-6) * decay * 1.254e-2) - ut(Re + Meso - rx) * 0.8

    dx = dx0 * dcr
    dy = dy0 * dcr
    dz = dz0 * dcr

    rxx = (rx ** 3)
    cte = - G * (Me + Ms)

    dxx = (cte * x) / rxx
    dyy = (cte * y) / rxx
    dzz = (cte * z) / rxx

    return [
        dx, dy, dz,
        dxx, dyy, dzz
    ]


app = QtGui.QApplication([])
win = QtGui.QMainWindow()

win.setWindowTitle('Newton Gravitational Law')
win.resize(700, 700)

area = DockArea()

w1 = pg.LayoutWidget()

b1 = QtGui.QPushButton("Button")

pg.setConfigOptions(antialias=True)

cw = QtGui.QWidget()
win.setCentralWidget(cw)
lzz = QtGui.QVBoxLayout()
cw.setLayout(lzz)

p4 = pg.PlotWidget()
p4.setAspectLocked(True)
ptr = 0
# p4 = win.addPlot(title="trajectory")
lzz.addWidget(p4)

rmm = Re
circle = pg.QtGui.QGraphicsEllipseItem(-rmm, -rmm, rmm * 2, rmm * 2)
circle.setPen(pg.mkPen(4))
p4.addItem(circle)

rmm = Re + Meso
circleM = pg.QtGui.QGraphicsEllipseItem(-rmm, -rmm, rmm * 2, rmm * 2)
circleM.setPen(pg.mkPen(1))
p4.addItem(circleM)

rmm = 1e5
circle = pg.QtGui.QGraphicsEllipseItem(-rmm, -rmm, rmm * 2, rmm * 2)
circle.setPen(pg.mkPen(1))
p4.addItem(circle)

p4.enableAutoRange('xy', False)

w = 1e5
cvS = pg.QtGui.QGraphicsEllipseItem(s0[0] + -w, s0[1] + -w, w * 2, w * 2)
cvS.setPen(pg.mkPen(3))
p4.addItem(cvS)

w = 1e5
cvv = pg.QtGui.QGraphicsEllipseItem(s0[0] + -w, s0[1] + -w, w * 2, w * 2)
cvv.setPen(pg.mkPen(3))
p4.addItem(cvv)

w = 1e2
cvv2 = pg.QtGui.QGraphicsEllipseItem(s0[0] + -w, s0[1] + -w, w * 2, w * 2)
cvv2.setPen(pg.mkPen(3))
p4.addItem(cvv2)

t = []
x = []
curve = p4.plot(t, x)
curve.setPen(0.7)
p4.showGrid(x=True, y=True)

r = ode(gforce)
r.set_integrator('vode', method='bdf')

dt = rate
r.set_initial_value(s0, 0)

timer = QtCore.QTimer()


def update():
    global curve, dt, r, t, x, timer

    tt = r.t + dt

    cmp = r.y
    xx = cmp[0]
    yy = cmp[1]
    zz = cmp[2]

    vxx = cmp[3]
    vyy = cmp[4]
    vzz = cmp[5]

    rx = (xx ** 2 + yy ** 2 + zz ** 2) ** 0.5

    if rx < Re:
        w = 1.5e5
        cvx = pg.QtGui.QGraphicsEllipseItem(xx + -w, yy + -w, w * 2, w * 2)
        cvx.setPen(pg.mkPen(3))
        p4.addItem(cvx)
        timer.stop()
        return

    r.integrate(tt)

    w = 3e5

    cvv2.setRect(xx + -w, yy + -w, w * 2, w * 2)
    w = 1e5

    cvv.setRect(xx + vxx * 200 + -w, yy + vyy * 200 + -w, w * 2, w * 2)
    t.append(xx)
    x.append(yy)

    if len(x) > 1000:
        del t[0]
        del x[0]
    curve.setData(t, x)


spins = [
    ("Theta",
     pg.SpinBox(value=theta, decimals=100, suffix='˚'), 'theta'),
    ("Alpha",
     pg.SpinBox(value=alpha, decimals=100, suffix='˚'), 'alpha'),
    ("Omega",
     pg.SpinBox(value=omega, decimals=100, suffix='˚'), 'omega'),
    ("Initial Velocity",
     pg.SpinBox(value=isp, decimals=100, suffix='m/s'), 'velocity'),
    ("Distance To Earth",
     pg.SpinBox(value=Ds, decimals=100, suffix='m'), 'distance'),
    ("Mesosphere Border",
     pg.SpinBox(value=Meso, decimals=100, suffix='m'), 'mesosphere'),
    ("Sattlite Mass",
     pg.SpinBox(value=Ms, decimals=100, suffix='kg'), 'mass'),
    ("Deltatime per step, step is every 50ms",
     pg.SpinBox(value=rate, decimals=100, suffix='s'), 'dt'),
    ("Orbital decay coefficient",
     pg.SpinBox(value=decay, decimals=100), 'decay'),
]


def valueChanged(sb):
    k = getattr(sb, 'slx')
    global dt, theta, omega, alpha, isp, Ds, Meso, Ms, decay, circle, s0, r0
    if k == 'dt':
        dt = sb.value()
    elif k == 'theta':
        theta = sb.value() * math.pi / 180
    elif k == 'omega':
        omega = sb.value() * math.pi / 180
    elif k == 'alpha':
        alpha = sb.value() * math.pi / 180
    elif k == 'velocity':
        isp = sb.value()
    elif k == 'distance':
        Ds = sb.value()
    elif k == 'mesosphere':
        Meso = sb.value()
    elif k == 'mass':
        Ms = sb.value()
    elif k == 'decay':
        decay = sb.value()

    rmm = Re + Meso
    circleM.setRect(-rmm, -rmm, rmm * 2, rmm * 2)

    w = 1e5
    cvS.setRect(s0[0] + -w, s0[1] + -w, w * 2, w * 2)

    calc(alpha, omega, theta, isp)
    r.set_initial_value(s0, 0)

    timer.stop()
    t.clear()
    x.clear()
    timer.start(50)
    pass


kscc = {}


def hello():
    hx = {}
    for text, spin, k in spins:
        label = QtGui.QLabel(text)
        # lzz.addWidget(label)
        lzz.addWidget(spin)

        setattr(spin, 'slx', k)

        spin.sigValueChanged.connect(valueChanged)


hello()

win.show()

timer.timeout.connect(update)
timer.start(50)  # rate * 1000)

if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
