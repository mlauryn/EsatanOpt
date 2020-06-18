from pyxdsm.XDSM import XDSM

opt = 'Optimization'
solver = 'MDA'
func = 'Function'
ifunc = 'ImplicitFunction'

x = XDSM()

x.add_system('opt', opt, (r'0, 7 \rightarrow 1:', r'\text{Optimizer}'))
x.add_system('cond', func, (r'1:', r'\text{Conductors}'))
x.add_system('Qext', func, (r'2:', r'\text{HeatFlux}'))
x.add_system('Newton', solver, (r'3, 5 \rightarrow 3:', r'\text{Newton}'))
x.add_system('Power', func, (r'4:', r'\text{Power}'))
x.add_system('Temp', ifunc, (r'5:', r'\text{Temperature}'))
x.add_system('Fun', func, (r'6:', r'\text{Functions}'), stack=True)

x.connect('opt', 'cond', r'\epsilon, k')
x.connect('opt', 'Qext', r'\Phi, \alpha')
#x.connect('opt', 'Power', r'QS')
x.connect('opt', 'Temp', r'Q_i')
x.connect('opt', 'Fun', r'Q_i')

x.connect('Qext', 'Power', r'Q_{sc}, Q_{sr}')
x.connect('cond', 'Temp', r'GL, GR')
x.connect('Power', 'Temp', r'Q_s')
x.connect('Power', 'Fun', r'P_{el}')
x.connect('Temp', 'Newton', r'\mathcal{R}(T)')
x.connect('Temp', 'Fun', 'T')
x.connect('Temp', 'Power', 'T')

x.connect('Fun', 'opt', 'f(x),c(x)')
#x.connect('G', 'opt', 'g')

x.add_input('opt', r'x_0', stack=True)

x.add_output('opt', 'x^*', side='left')
x.add_output('Temp', 'T^*', side='left')
#x.add_output('Qext', r'Q_{sun}^*', side='left')
x.add_output('Fun', 'f^*, c^*', side='left')
#x.add_output('Power', r'Q_{dis}^*', side='left')

x.add_process(['opt', 'cond','Qext', 'Newton', 'Power', 'Temp', 'Newton' ], arrow=True)
x.add_process(['Newton', 'Fun', 'opt'], arrow=True)

x.write('xdsm_MAT') 