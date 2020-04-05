import numpy as np

templ = open('Radiative_template.txt', 'r')
cmd = templ.read()
templ.close()

file = open("Radiative_cmd.txt", 'w')

text = cmd.format(end_angle=90.0, angle_gap=5.0, is_in_spin='TRUE', spin_axis = [0.0, 1.0, 0.0], spin_pos = 18)

file.write(text)

file.close

# create command for automatic report generation

report = """REPORT_HF_AGAINST_TIME(
orbit_times = {case}.ORBIT_TIMES,
orbit_angles = {case}.ORBIT_ANGLES,
direct_sun_hf = {case}.SDF,
intensity = TRUE,
value_min = 0.0,
value_max = 1.7976931348623158E308
);"""

""" file = open("Report_cmd.txt", 'w')

for angle in phi:
    text = report.format(case='phi_%d' % int(angle))
    file.write(text + '\n\n')

file.close """