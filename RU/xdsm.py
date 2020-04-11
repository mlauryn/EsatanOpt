from pyxdsm.XDSM import XDSM

opt = 'Optimization'
solver = 'MDA'
func = 'Function'
ifunc = 'ImplicitFunction'

x = XDSM()

x.add_system('opt', opt, (r'0, 7 \rightarrow 1:', r'\text{Optimizer}'))
x.add_system('solver', solver, (r'1, 5 \rightarrow 2:', r'\text{Solver}'))
x.add_system('Qext', func, (r'2:', r'\text{HeatFlux}'))
x.add_system('Power', func, (r'3:', r'\text{Power}'))
x.add_system('Temp', ifunc, (r'4:', r'\text{Temperature}'))
x.add_system('Fun', func, (r'6:', r'\text{Functions}'), stack=True)

x.connect('opt', 'solver', r'T^{t,(0)}')
x.connect('opt', 'Qext', r'\Phi, \Psi, \Omega, \alpha')
x.connect('opt', 'Power', r'P_{out}')
x.connect('opt', 'Temp', r'\epsilon, k')
x.connect('opt', 'Fun', r'P_{out}')
x.connect('solver', 'Qext', 'T^t')
x.connect('solver', 'Power', 'T^t')
x.connect('Qext', 'Power', r'Q_{sun}')
x.connect('Qext', 'Temp', r'Q_{sun}, Q_{planet}, Q_{alb}')
x.connect('Qext', 'Temp', r'Q_{sun}, Q_{planet}, Q_{alb}')
x.connect('Power', 'Temp', r'Q_{dis}')
x.connect('Power', 'Fun', r'Q_{dis}, P_{el}')
x.connect('Temp', 'solver', r'\mathcal{R}(T)')
x.connect('Temp', 'Fun', 'T')

x.connect('Fun', 'opt', 'f,c')
#x.connect('G', 'opt', 'g')

x.add_input('opt', r'x_0', stack=True)

x.add_output('opt', 'x^*', side='left')
x.add_output('Temp', 'T^*', side='left')
x.add_output('Qext', r'Q_{sun}^*', side='left', stack=True)
x.add_output('Fun', 'f^*, c^*', side='left')
x.add_output('Power', r'Q_{dis}^*', side='left')

x.add_process(['opt', 'solver', 'Qext', 'Power', 'Temp', 'solver' ], arrow=True)
x.add_process(['solver', 'Fun', 'opt'], arrow=True)

x.write('mdf') 