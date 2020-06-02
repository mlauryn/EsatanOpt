from openmdao.api import CaseReader
from matplotlib import pyplot as plt
from plot_size import set_size
from matplotlib.ticker import FormatStrFormatter

plt.style.use('thesis')
#plt.rcParams['axes.grid'] = True
#plt.rcParams['grid.linestyle'] = '--'

x = [7,28, 63, 112, 175, 252] # number of nodes
y1 = [] # design var
y2 = [] # objectives result
y3 = [4e-04]*6 # objective Esatan
for num in range(1,7):
    
    file_name = './Cases/CUBESAT{num}.sql'.format(num=num)
    
    # load cases from recording database
    cr = CaseReader(file_name)
    #cases = cr.get_cases('driver')
    cases = cr.list_cases('driver')
    
    # open last case
    case = cr.get_case(cases[-1])
    dvs = case.get_design_vars()
    objs = case.get_objectives()
    
    """ num_cases = len(cases)
    if num_cases == 0:
        print('No data yet...')
        #quit()
    else:
        print('# cases:', num_cases) """

    y1.append(dvs['outer_surf'])
    y2.append((objs['obj']))

fig, (ax1, ax2)  = plt.subplots(2, sharex=True, figsize=set_size('thesis', subplots=(2,1)))
fig.subplots_adjust(left=0.15)

ax1.plot(x, y1, '-+')
ax1.set(title='a', ylabel=r'$\epsilon$')
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax2.plot(x, y2, '-o', label='OpenMDAO result')
ax2.plot(x, y3, '-x', label= 'Esatan-TMS result')
ax2.set(title = 'b', xlabel='Total number of thermal nodes', ylabel= r'$f(\epsilon)$', xticks=x)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.legend()

plt.show()