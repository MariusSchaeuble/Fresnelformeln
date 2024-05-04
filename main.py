import matplotlib
import numpy
import numpy as np
import sympy as sym
from Helpers import identifier, isCharacter
import math
from numpy import matrix, array, mean, std, max, linspace, ones, sin, cos, tan, arctan, pi, sqrt, exp, arcsin, arccos, arctan2, sinh, cosh
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, xlabel, ylabel, legend, title, savefig, errorbar, grid
import scipy.optimize as opt
from GPII import *
from math import sqrt
pi = math.pi


plt.rcParams["text.usetex"] = True
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 20,
    "font.size": 20,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
}
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)



def gauss(term):
    ids = identifier(term)
    symbols = []
    for str1 in ids:
        symbols.append(sym.sympify(str1))
    termSymbol = sym.sympify(term)
    values = []
    for ident in ids:
        exec("values.append(" + ident + ")")

    derivatives = []
    i = 0
    while i < len(symbols):
        r = sym.diff(termSymbol, symbols[i])
        j = 0
        while j < len(symbols):
            # exec('r.evalf(subs={symbols[j]: ' + values[j] + '})')
            r = r.evalf(subs={symbols[j]: values[j]})
            j += 1
        derivatives.append(r.evalf())
        i += 1
    i = 0
    while i < len(derivatives):
        exec("derivatives[i] *= sigma_" + ids[i])
        i = i + 1
    res = 0
    for z in derivatives:
        res += z ** 2
    return math.sqrt(res)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -






U_dunkel = - 10/1000 #V
sigma_U_dunkel = 0.01/1000 + 0.008*U_dunkel

n1 = 1 #luft
n2 = np.tan(56.5*pi/180) #aus brewster winkel

I0_p = 10.45
sigma_I0_p = 0.01 + 0.008*I0_p

I0_s = 12.53
sigma_I0_s = 0.01 + 0.008*I0_s

#parallel polarisiert
#Reflektierter Strahl
#Annahme: einfall = ausfall

W_p_refl = matrix("""
10 0.91;
15 0.56;
20 0.39;
25 0.34;
30 0.29;
35 0.21;
40 0.11;
45 0.07;
50 0.02;
55 0;
60 0;
65 0.12;
70 0.31;
75 1.03;
80 2.68;
81 3.14;
82 4.12;
83 4.30;
84 5;
85 6.72;
86 6.98;
87 7.81
""") # Einfallswinkel in °, Spannung in V, messbereich 20V


U = toArray(W_p_refl[:, 1])
sigma_U = ones(len(U))
for i in range(len(U)):
    U[i] = (U[i] - U_dunkel)/I0_p
    temp1 = U[i]
    sigma_temp1 = 0.01 + 0.008*U[i]
    sigma_U[i] = gauss("(temp1 - U_dunkel)/I0_p")

alpha = toArray(W_p_refl[:, 0])
sigma_alpha = ones(len(alpha))*0.25

def fres(alpha1):
    alpha1 = alpha1*pi/180
    beta = arcsin(sin(alpha1)/n2)
    if alpha1 + beta == 0:
        return 0
    else:
        return (tan(alpha1 - beta)/tan(alpha1 + beta))**2


errorbar(alpha, U, sigma_U, sigma_alpha,'x', label='Gemessenes Spannungsverhältnis')
plot(alpha, np.vectorize(fres)(alpha), label="Theoriekurve")
xlabel('Einfallswinkel in °, p polarisiert', fontsize=20)
ylabel('Reflexionsvermögen', fontsize=20)
legend(fontsize=20)
grid()
plt.tight_layout()
savefig('pr.png')
show()




#transmitteierter strahl
W_p_trans = matrix("""
10 9.68;
15 9.70;
20 9.72;
25 9.81;
30 9.9;
35 10.00;
40 10.13;
45 10.24;
50 10.33;
55 10.31;
60 10.27;
65 10.13;
70 9.49;
75 8.35;
80 6.2;
81 5.72;
82 5.13;
83 4.35;
84 4.01;
85 3.47;
86 2.98;
87 1.6;
88 1.44;
89 0.46
""")


U = toArray(W_p_trans[:, 1])
sigma_U = ones(len(U))
for i in range(len(U)):
    U[i] = (U[i] - U_dunkel)/I0_p
    temp1 = U[i]
    sigma_temp1 = 0.01 + 0.008*U[i]
    sigma_U[i] = gauss("(temp1 - U_dunkel)/I0_p")

alpha = toArray(W_p_trans[:, 0])
sigma_alpha = ones(len(alpha))*0.25

def fres(alpha1):
    alpha1 = alpha1*pi/180
    beta = arcsin(sin(alpha1)/n2)
    return (2*sin(beta)*cos(alpha1)/(sin(alpha1 + beta)*cos(alpha1 - beta)))**2*n2*cos(beta)/cos(alpha1)


errorbar(alpha, U, sigma_U, sigma_alpha,'x', label='Gemessenes Spannungsverhältnis')
plot(alpha, np.vectorize(fres)(alpha), label="Theoriekurve")
xlabel('Einfallswinkelswinkel in °, p polarisiert', fontsize=20)
ylabel('Transmissionsvermögen', fontsize=20)
legend(fontsize=20)
grid()
plt.tight_layout()
savefig('pt.png')
show()


W_s_refl = matrix("""
10 0.44;
15 0.45;
20 0.56;
25 0.67;
30 0.72;
35 0.88;
40 1.01;
45 1.15;
50 1.49;
55 1.71;
60 2.28;
65 3.02;
70 3.94;
75 5.25;
80 6.99;
81 7.52;
82 7.89;
83 9.32;
84 9.97;
85 10.24;
86 12.20;
87 11.27
""")

U = toArray(W_s_refl[:, 1])
sigma_U = ones(len(U))
for i in range(len(U)):
    U[i] = (U[i] - U_dunkel)/I0_p
    temp1 = U[i]
    sigma_temp1 = 0.01 + 0.008*U[i]
    sigma_U[i] = gauss("(temp1 - U_dunkel)/I0_p")

alpha = toArray(W_s_refl[:, 0])
sigma_alpha = ones(len(alpha))*0.25

def fres(alpha1):
    alpha1 = alpha1*pi/180
    beta = arcsin(sin(alpha1)/n2)
    return (sin(alpha1 - beta)/sin(alpha1 + beta))**2


errorbar(alpha, U, sigma_U, sigma_alpha,'x', label='Gemessenes Spannungsverhältnis')
plot(alpha, np.vectorize(fres)(alpha), label="Theoriekurve")
xlabel('Einfallswinkel in °, s polarisiert', fontsize=20)
ylabel('Reflexionsvermögen', fontsize=20)
legend(fontsize=15)
grid()
plt.tight_layout()
savefig('sr.png')
show()

W_s_trans = matrix("""
10 09.77;
15 11.20;
20 10.79;
25 09.80;
30 10.80;
35 7.58;
40 10.34;
45 10.00;
50 9.52;
55 8.90;
60 8.14;
65 7.13;
70 5.82;
75 4.21;
80 2.41;
81 2.09;
82 1.71;
83 1.39;
84 1.08;
85 0.76;
86 0.52;
87 0.58
""")


U = toArray(W_s_trans[:, 1])
sigma_U = ones(len(U))
for i in range(len(U)):
    U[i] = (U[i] - U_dunkel)/I0_p
    temp1 = U[i]
    sigma_temp1 = 0.01 + 0.008*U[i]
    sigma_U[i] = gauss("(temp1 - U_dunkel)/I0_p")

alpha = toArray(W_s_trans[:, 0])
sigma_alpha = ones(len(alpha))*0.25

def fres(alpha1):
    alpha1 = alpha1*pi/180
    beta = arcsin(sin(alpha1)/n2)
    return (2*sin(beta)*cos(alpha1)/(sin(alpha1 + beta)))**2*n2*cos(beta)/cos(alpha1)


errorbar(alpha, U, sigma_U, sigma_alpha,'x', label='Gemessenes Spannungsverhältnis')
plot(alpha, np.vectorize(fres)(alpha), label="Theoriekurve")
xlabel('Einfallswinkelswinkel in °, s polarisiert', fontsize=20)
ylabel('Transmissionsvermögen', fontsize=20)
legend(fontsize=15)
grid()
plt.tight_layout()
savefig('st.png')
show()



