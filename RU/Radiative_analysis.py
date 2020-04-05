import numpy as np

n = 19
phi = np.linspace(0.0, 90.0, n, endpoint=True)

templ = open('Radiative_template.txt', 'r')
cmd = templ.read()
templ.close()

file = open("Radiative_cmd.txt", 'w')

for angle in phi:
    text = cmd.format(name='phi_%d' % int(angle), phi=angle, psi=0.0, omega=0.0, is_in_spin='TRUE', spin_axis = [0.0, 1.0, 0.0], spin_pos = 18)
    file.write(text + '\n\n')

file.close

# create commands for automatic report generation

report = """REPORT_HF_AGAINST_TIME(
orbit_times = {case}.ORBIT_TIMES,
orbit_angles = {case}.ORBIT_ANGLES,
direct_sun_hf = {case}.SDF,
intensity = TRUE,
value_min = 0.0,
value_max = 1.7976931348623158E308
);"""

file = open("Report_cmd.txt", 'w')

for angle in phi:
    text = report.format(case='phi_%d' % int(angle))
    file.write(text + '\n\n')

file.close